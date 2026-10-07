"""Evaluate original released Rev weights using its unchanged Scorer.encode/run_rows."""

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from experiments.action_training import log_messages
from experiments.choice_action import shuffled_orders
from experiments.large_choice_eval import make_records, metrics
from experiments.runtime import Run, ensure_tensorboard, json_write, measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_original():
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    manifest = json.loads((ROOT / "docs/original_rev_manifest.json").read_text(encoding="utf-8"))
    source = ROOT / "third_party/rev/serve_local.py"
    expected_source_hash = manifest.get(
        "serve_local_formatted_sha256", manifest["serve_local.py_sha256"]
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_source_hash:
        raise ValueError("original scorer source changed")
    spec = importlib.util.spec_from_file_location("original_rev_serve", source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    packaged_checkpoint = ROOT.parent / "weights" / "original_rev" / "checkpoint.pt"
    adapter_directory = manifest.get("adapter_directory")
    cached_checkpoint = Path(adapter_directory) / "checkpoint.pt" if adapter_directory else None
    if packaged_checkpoint.exists():
        checkpoint_path = packaged_checkpoint
    elif cached_checkpoint and cached_checkpoint.exists():
        checkpoint_path = cached_checkpoint
    else:
        from huggingface_hub import hf_hub_download

        checkpoint_path = Path(
            hf_hub_download(
                repo_id=manifest["adapter_repository"],
                filename="checkpoint.pt",
                revision=manifest["adapter_revision"],
            )
        )
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError("original weights hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if (
        metadata["model"] != manifest["base_repository"]
        or metadata["revision"] != manifest["base_revision"]
    ):
        raise ValueError("base revision mismatch")
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(814)
    born = time.perf_counter()
    path = manifest.get("base_directory")
    local_base = bool(path and Path(path).exists())
    model_source = path if local_base else manifest["base_repository"]
    revision = None if local_base else manifest["base_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, revision=revision, local_files_only=local_base, trust_remote_code=False
    )
    full = AutoModelForImageTextToText.from_pretrained(
        model_source,
        revision=revision,
        local_files_only=local_base,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    backbone = full.model.language_model
    del full
    torch.cuda.empty_cache()
    modules = dict(backbone.named_modules())
    for name, weights in checkpoint["adapters"].items():
        module = modules[name]
        if (
            not isinstance(module, torch.nn.Linear)
            or name.split(".")[-1] not in original.LORA_TARGETS
        ):
            raise ValueError("unexpected adapter " + name)
        with torch.no_grad():
            module.weight.add_(
                (2 * (weights["b"].float() @ weights["a"].float()))
                .to(module.weight.dtype)
                .to(module.weight.device)
            )
    h = backbone.config.hidden_size
    headq = torch.nn.Linear(h, 256, bias=False, device="cuda", dtype=torch.float32)
    headk = torch.nn.Linear(h, 256, bias=False, device="cuda", dtype=torch.float32)
    headq.load_state_dict(checkpoint["headq"])
    headk.load_state_dict(checkpoint["headk"])
    info = {
        **manifest,
        "model": metadata["model"],
        "state_format": metadata.get("state_format", "json"),
        "temperature": float(metadata.get("temperature", 1)),
        "merged_adapters": len(checkpoint["adapters"]),
        "backbone_parameters": sum(p.numel() for p in backbone.parameters()),
        "head_parameters": sum(p.numel() for p in headq.parameters())
        + sum(p.numel() for p in headk.parameters()),
        "precision": "BF16 merged LoRA, FP32 pointer head",
        "generated_tokens": 0,
        "unchanged_upstream_methods": ["Scorer.encode", "Scorer.run_rows", "Scorer.score"],
        "loader_differences": [
            "local pinned paths",
            "safe weights_only=True",
            "no optional kernel monkeypatch",
        ],
        "finetuning_on_Orion_examples": False,
    }
    scorer = original.Scorer(tokenizer, backbone, headq, headk, info)
    del checkpoint
    gc.collect()
    scorer.warm()
    info["load_seconds"] = time.perf_counter() - born
    print(
        "SELFTEST=" + json.dumps(scorer.score(original.SELFTEST_REQUEST), ensure_ascii=False),
        flush=True,
    )
    return scorer, info


def encode(scorer, rows, seed, corpus):
    orders = shuffled_orders(rows, seed)
    encoded, texts = [], []
    for row, order in zip(rows, orders, strict=True):
        criteria = {
            str(i + 1): corpus["candidates"][option]["description"]
            for i, option in enumerate(order)
        }
        question = {"id": row["id"], "instructions": corpus["instructions"], "criteria": criteria}
        ids, positions, _ = scorer.encode(row["command"], question)
        if len(ids) > 16384:
            raise ValueError("upstream token budget exceeded")
        encoded.append((ids, positions))
        texts.append(
            "State:\n"
            + row["command"]
            + "\nQuestion: "
            + corpus["instructions"]
            + "\nOptions:\n"
            + "".join(k + ": " + v + "\n" for k, v in criteria.items())
            + "Decision:"
        )
    return encoded, orders, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    smol = json.loads((ROOT / "docs/large_choice_results.json").read_text(encoding="utf-8"))
    if not Path(smol["run_dir"], "selection_lock.json").exists():
        raise ValueError("Smol checkpoint not locked")
    corpus = json.loads((ROOT / "docs/large_choice_dataset.json").read_text(encoding="utf-8"))
    if corpus["sha256"] != smol["dataset_hash"]:
        raise ValueError("dataset changed")
    scorer, info = load_original()
    print(ensure_tensorboard(6007), flush=True)
    with Run(
        "original-Rev-Qwen3.5-4B-same-task",
        {
            "model": info,
            "dataset_hash": corpus["sha256"],
            "protocol": "released checkpoint zero-shot; same 26 descriptions/policy/order seeds as selected Smol",
            "smol_run": smol["run_dir"],
            "batch_size": args.batch_size,
            "no_test_prompt_tuning": True,
        },
    ) as run:
        print("RUN_DIR=" + str(run.path), flush=True)
        run.save("dataset.json", corpus)
        final = {}
        started = time.perf_counter()
        # No prompt/model selection on any fold. Validation is reported for
        # diagnostics only.
        for fold in ("validation", "test", "legacy_challenge"):
            rows = corpus["splits"][fold]
            records = []
            for seed in (21001, 21002):
                # Match Smol's order seed for chunks of12 even with different
                # inference batch size.
                orders = shuffled_orders(rows, seed)
                # Smol generates random order separately for each12-row chunk;
                # reconstruct those exactly.
                orders = []
                for j in range(0, len(rows), 12):
                    orders.extend(shuffled_orders(rows[j : j + 12], seed + j))
                for i in range(0, len(rows), args.batch_size):
                    chunk = rows[i : i + args.batch_size]
                    encoded = []
                    texts = []
                    chunk_orders = orders[i : i + len(chunk)]
                    for row, order in zip(chunk, chunk_orders, strict=True):
                        criteria = {
                            str(n + 1): corpus["candidates"][option]["description"]
                            for n, option in enumerate(order)
                        }
                        ids, positions, _ = scorer.encode(
                            row["command"],
                            {"instructions": corpus["instructions"], "criteria": criteria},
                        )
                        if len(ids) > 16384:
                            raise ValueError("too many tokens")
                        encoded.append((ids, positions))
                        texts.append(
                            "State:\n"
                            + row["command"]
                            + "\nQuestion: "
                            + corpus["instructions"]
                            + "\nOptions:\n"
                            + "".join(k + ": " + v + "\n" for k, v in criteria.items())
                            + "Decision:"
                        )
                    probs, _ = scorer.run_rows(encoded)
                    # Same argmax; tiny FP32 renormalization for common
                    # softmax-based metrics.
                    logits = torch.tensor(probs).clamp_min(1e-30).log()
                    now = make_records(logits, chunk, chunk_orders, texts, corpus["candidates"])
                    for record, native in zip(now, probs, strict=True):
                        record["native_probabilities"] = native
                    log_messages(
                        run, now, fold + "_selected", 0, batch_index=(seed - 21001) * 10000 + i
                    )
                    records.extend(now)
                    if i % 40 == 0:
                        print(f"{fold} order={seed} rows={i + len(chunk)}/{len(rows)}", flush=True)
            final[fold] = metrics(records)
            run.log(final[fold], 0, fold + "_selected/")
            run.save(fold + "_selected.json", {"metrics": final[fold], "records": records})
            print(f'{fold} exact={
                    final[fold]["exact"]:.3f} scope={
                    final[fold]["scope_exact"]}', flush=True)
        sample, _, _ = encode(scorer, corpus["splits"]["test"][:1], 21001, corpus)
        latency = measure(lambda: scorer.run_rows(sample), "cuda", warmup=5, repeats=20)
        report = {
            "run_dir": str(run.path),
            "dataset_hash": corpus["sha256"],
            "model": info,
            "metrics": final,
            "latency_batch1": latency,
            "elapsed_eval_seconds": time.perf_counter() - started,
            "updates": 0,
            "processed_examples": 0,
            "training_rows": 0,
            "is_original_Rev": True,
            "comparison_limits": [
                "Different base size134.5M versus4B and pretraining/languages.",
                "Smol is Orion-task trained; Rev is published zero-shot.",
                "Native model input layout/tokenizer differ; policy and candidate meanings/orders identical.",
                "Synthetic grammar-held-out test is not production accuracy.",
            ],
        }
        run.save("report.json", report)
        json_write(ROOT / "docs/original_rev_results.json", report)
        run.writer.add_text("summary/report", json.dumps(report, ensure_ascii=False, indent=2))
        print("COMPLETE=" + str(run.path), flush=True)


if __name__ == "__main__":
    main()
