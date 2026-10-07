"""Continue released Qwen Rev LoRA+pointer on Orion; validation selects, tests never do."""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from experiments.action_training import log_messages
from experiments.choice_action import shuffled_orders, signature
from experiments.large_choice_data import build_large_choice_dataset, normalize
from experiments.large_choice_eval import make_records, metrics
from experiments.rev_finetuning import RevLoRA, paired_loss, pointer_logits
from experiments.runtime import Run, json_write, seed_all, sync

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fresh_test(corpus):
    cache = ROOT / "docs/rev_finetune_fresh_test.json"
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored["source_dataset_hash"] == corpus["sha256"]:
            return stored["rows"]
    blocked = {normalize(r["command"]) for rows in corpus["splits"].values() for r in rows}
    pools = {d: [] for d in ("EXECUTE", "CLARIFY", "UNSUPPORTED", "NO_ACTION")}
    for seed in range(91, 121):
        for row in build_large_choice_dataset(seed)["splits"]["test"]:
            key = normalize(row["command"])
            if key not in blocked:
                blocked.add(key)
                pools[row["decision"]].append(row)
        if all(len(rows) >= (225 if d == "EXECUTE" else 25) for d, rows in pools.items()):
            break
    rows = [r for d, pool in pools.items() for r in pool[: 225 if d == "EXECUTE" else 25]]
    if len(rows) != 300:
        raise ValueError("not enough unseen test strings")
    random.Random(904).shuffle(rows)
    json_write(
        cache,
        {
            "source_dataset_hash": corpus["sha256"],
            "rows": rows,
            "limitations": "unseen strings, same held-out grammar families as previous inspected test",
        },
    )
    return rows


def load_trainable():
    manifest = json.loads((ROOT / "docs/original_rev_manifest.json").read_text(encoding="utf-8"))
    source = ROOT / "third_party/rev/serve_local.py"
    expected_source_hash = manifest.get(
        "serve_local_formatted_sha256", manifest["serve_local.py_sha256"]
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_source_hash:
        raise ValueError("upstream source changed")
    spec = importlib.util.spec_from_file_location("rev_upstream", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    packaged_checkpoint = ROOT.parent / "weights" / "original_rev" / "checkpoint.pt"
    adapter_directory = manifest.get("adapter_directory")
    cached_checkpoint = Path(adapter_directory) / "checkpoint.pt" if adapter_directory else None
    if packaged_checkpoint.exists():
        path = packaged_checkpoint
    elif cached_checkpoint and cached_checkpoint.exists():
        path = cached_checkpoint
    else:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=manifest["adapter_repository"],
                filename="checkpoint.pt",
                revision=manifest["adapter_revision"],
            )
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError("source weights changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    base_directory = manifest.get("base_directory")
    local_base = bool(base_directory and Path(base_directory).exists())
    model_source = base_directory if local_base else manifest["base_repository"]
    revision = None if local_base else manifest["base_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=revision,
        local_files_only=local_base,
    )
    torch.backends.cuda.enable_cudnn_sdp(False)
    full = AutoModelForImageTextToText.from_pretrained(
        model_source,
        revision=revision,
        local_files_only=local_base,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    backbone = full.model.language_model
    del full
    backbone.requires_grad_(False)
    adapters = {}
    for name, weights in state["adapters"].items():
        parent_name, leaf = name.rsplit(".", 1)
        parent = backbone.get_submodule(parent_name)
        layer = getattr(parent, leaf)
        if not isinstance(layer, torch.nn.Linear) or leaf not in upstream.LORA_TARGETS:
            raise ValueError("unexpected adapter " + name)
        adapters[name] = RevLoRA(layer, weights)
        setattr(parent, leaf, adapters[name])
    headq = torch.nn.Linear(backbone.config.hidden_size, 256, bias=False, device="cuda")
    headk = torch.nn.Linear(backbone.config.hidden_size, 256, bias=False, device="cuda")
    headq.load_state_dict(state["headq"])
    headk.load_state_dict(state["headk"])
    info = {
        "state_format": state["metadata"].get("state_format", "raw"),
        "temperature": float(state["metadata"].get("temperature", 1)),
        "manifest": manifest,
    }
    scorer = upstream.Scorer(tokenizer, backbone, headq, headk, info)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return scorer, adapters


def encode(scorer, rows, orders, corpus):
    encoded, texts, targets = [], [], []
    for row, order in zip(rows, orders, strict=True):
        criteria = {str(i + 1): corpus["candidates"][j]["description"] for i, j in enumerate(order)}
        ids, positions, _ = scorer.encode(
            row["command"], {"instructions": corpus["instructions"], "criteria": criteria}
        )
        if len(ids) > 1536:
            raise ValueError("token budget exceeded")
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
        target = signature(row["decision"], row["plan"]["actions"])
        targets.append(
            next(
                i
                for i, j in enumerate(order)
                if signature(
                    corpus["candidates"][j]["decision"], corpus["candidates"][j]["actions"]
                )
                == target
            )
        )
    return encoded, texts, torch.tensor(targets, device="cuda")


def logits(scorer, encoded):
    return pointer_logits(
        scorer.backbone, scorer.headq, scorer.headk, encoded, scorer.pad_id, scorer.temperature
    )


def save_checkpoint(path, scorer, adapters, optimizer, epoch, step, corpus_hash):
    state = {
        "adapters": {
            name: {"a": a.a.detach().cpu(), "b": a.b.detach().cpu()} for name, a in adapters.items()
        },
        "headq": {k: v.detach().cpu() for k, v in scorer.headq.state_dict().items()},
        "headk": {k: v.detach().cpu() for k, v in scorer.headk.state_dict().items()},
        "metadata": {
            **scorer.info["manifest"]["checkpoint_metadata"],
            "temperature": scorer.temperature,
            "initial_checkpoint": scorer.info["manifest"]["checkpoint_sha256"],
            "training": "Orion continuation; unmerged FP32 LoRA+head, frozen BF16 Qwen",
            "epochs": epoch,
        },
        "examples": 1500 * epoch,
        "epoch": epoch,
        "step": step,
        "dataset_hash": corpus_hash,
        "optimizer": optimizer.state_dict(),
    }
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def restore(path, scorer, adapters):
    state = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for name, a in adapters.items():
            a.a.copy_(state["adapters"][name]["a"])
            a.b.copy_(state["adapters"][name]["b"])
    scorer.headq.load_state_dict(state["headq"])
    scorer.headk.load_state_dict(state["headk"])
    return state


@torch.inference_mode()
def evaluate(scorer, rows, corpus, run, phase, step, seeds=(19001,), batch_size=2):
    scorer.backbone.eval()
    scorer.headq.eval()
    scorer.headk.eval()
    records = []
    for view, seed in enumerate(seeds):
        orders = []
        for j in range(0, len(rows), 12):
            orders.extend(shuffled_orders(rows[j : j + 12], seed + j))
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            order = orders[i : i + len(chunk)]
            encoded, texts, _ = encode(scorer, chunk, order, corpus)
            now = make_records(logits(scorer, encoded), chunk, order, texts, corpus["candidates"])
            log_messages(run, now, phase, step, batch_index=view * 10000 + i)
            records.extend(now)
    result = metrics(records)
    run.log(result, step, phase + "/")
    run.save(phase + ".json", {"metrics": result, "records": records})
    print(f'EVAL {phase}: exact={
            result["exact"]:.4f} false_action={
            result["false_action_rate"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--max-commands",
        type=int,
        default=None,
        help="Stop at a complete optimizer update; evaluate via evaluate_stopped_rev.py",
    )
    args = parser.parse_args()
    seed_all(814)
    corpus = json.loads((ROOT / "docs/large_choice_dataset.json").read_text(encoding="utf-8"))
    novel = fresh_test(corpus)
    config = {
        "source": "released Rev Qwen3.5-4B",
        "train_rows": 1500,
        "validation_rows": 300,
        "adapter_lr": 2e-5,
        "head_lr": 1e-4,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "paired_orders": True,
        "consistency_weight": 0.1,
        "effective_commands_per_update": 8,
        "microbatch_commands": 1,
        "max_epochs": args.max_epochs,
        "warmup_updates": 20,
        "selection": "validation macro_candidate_exact, tie lower CE",
        "stop": "at least 3 epochs then 2 stale validation epochs; max6",
        "old_test": "development regression; already inspected in previous experiment",
        "fresh_test": "300 unseen strings from existing held-out grammars; not new independent grammar families",
        "all_base_weights_frozen": True,
        "base_dtype": "BF16",
        "trainable_dtype": "FP32",
        "dataset_hash": corpus["sha256"],
        "fresh_test_sha256": hashlib.sha256(
            json.dumps(novel, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    with Run(
        "Rev-Qwen3.5-4B-Orion-LoRA-finetune" + ("-smoke" if args.smoke else ""), config
    ) as run:
        print("RUN_DIR=" + str(run.path), flush=True)
        run.save("dataset.json", corpus)
        run.save("fresh_test.json", novel)
        scorer, adapters = load_trainable()
        ap = [p for a in adapters.values() for p in (a.a, a.b)]
        hp = list(scorer.headq.parameters()) + list(scorer.headk.parameters())
        params = ap + hp
        if sum(p.numel() for p in params) != 15687680:
            raise ValueError("trainable count differs from released Rev")
        optimizer = torch.optim.AdamW(
            [{"params": ap, "lr": config["adapter_lr"]}, {"params": hp, "lr": config["head_lr"]}],
            weight_decay=0.01,
        )
        run.save(
            "model.json",
            {
                "trainable_parameters": sum(p.numel() for p in params),
                "adapters": len(adapters),
                "source_manifest": scorer.info["manifest"],
            },
        )
        train = corpus["splits"]["train"]
        val = corpus["splits"]["validation"]
        step = 0
        history = []
        if not args.smoke:
            initial = evaluate(scorer, val, corpus, run, "validation_initial", 0)
        else:
            initial = {}
        best = (
            (initial["macro_candidate_exact"], -initial["loss"])
            if initial
            else (-1.0, -float("inf"))
        )
        if initial:
            save_checkpoint(
                run.path / "best.pt", scorer, adapters, optimizer, 0, 0, corpus["sha256"]
            )
        stale = 0
        started = time.perf_counter()
        window = []
        stop = "maximum epochs reached; no claim of convergence"
        for epoch in range(1, (1 if args.smoke else args.max_epochs) + 1):
            indices = list(range(1 if args.smoke else len(train)))
            rng = random.Random(814 + epoch)
            rng.shuffle(indices)
            epoch_values = []
            optimizer.zero_grad(set_to_none=True)
            for offset, index in enumerate(indices):
                row = train[index]
                orders = shuffled_orders([row, row], 814 + epoch * 100000 + offset)
                rows = [row, row]
                encoded, texts, targets = encode(scorer, rows, orders, corpus)
                scorer.backbone.train()
                scorer.headq.train()
                scorer.headk.train()
                then = time.perf_counter()
                output = logits(scorer, encoded)
                loss, ce, kl = paired_loss(output, targets, orders, 0.1)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                group_start = (offset // 8) * 8
                group_size = min(8, len(indices) - group_start)
                (loss / group_size).backward()
                records = make_records(output, rows, orders, texts, corpus["candidates"])
                log_messages(run, records, "train_preupdate", step, batch_index=offset)
                item = {
                    "loss/total": float(loss.detach()),
                    "loss/ce": float(ce.detach()),
                    "loss/consistency": float(kl.detach()),
                }
                epoch_values.append(item)
                window.append(item)
                if (offset + 1) % 8 == 0 or offset + 1 == len(indices):
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                    if args.smoke:
                        nonzero_adapters = sum(
                            a.a.grad is not None and a.a.grad.abs().max().item() > 0
                            for a in adapters.values()
                        )
                        print("SMOKE_GRADIENT_ADAPTERS=" + str(nonzero_adapters), flush=True)
                        if nonzero_adapters != len(adapters):
                            raise ValueError("missing adapter gradients")
                    step += 1
                    factor = min(1.0, step / 20)
                    for g, lr in zip(
                        optimizer.param_groups, (config["adapter_lr"], config["head_lr"])
                    ):
                        g["lr"] = lr * factor
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    run.log(
                        {"gradient_norm": float(norm), "epoch": epoch, "commands_seen": offset + 1},
                        step,
                        "train/",
                    )
                    if (run.path / "STOP_REQUESTED").exists() or (
                        args.max_commands is not None
                        and (epoch - 1) * 1500 + offset + 1 >= args.max_commands
                    ):
                        run.log(
                            {
                                k + "_mean": sum(v[k] for v in epoch_values) / len(epoch_values)
                                for k in epoch_values[0]
                            },
                            step,
                            "train_partial_epoch/",
                        )
                        if window:
                            run.log(
                                {k: sum(v[k] for v in window) / len(window) for k in window[0]},
                                step,
                                "train_window_partial/",
                            )
                        save_checkpoint(
                            run.path / "last.pt",
                            scorer,
                            adapters,
                            optimizer,
                            epoch - 1,
                            step,
                            corpus["sha256"],
                        )
                        run.save(
                            "progress.json",
                            {
                                "epoch_in_progress": epoch,
                                "commands_in_epoch": offset + 1,
                                "step": step,
                                "history": history,
                                "stopped": True,
                            },
                        )
                        run.save(
                            "stop_status.json",
                            {
                                "reason": "stop marker or explicit command budget",
                                "commands_in_epoch": offset + 1,
                                "step": step,
                                "training_stopped": True,
                            },
                        )
                        print("STOPPED_CHECKPOINT=" + str(run.path / "last.pt"), flush=True)
                        return
                sync("cuda")
                if len(window) == 25 or offset + 1 == len(indices):
                    average = {k: sum(v[k] for v in window) / len(window) for k in item}
                    run.log(average, step, "train_window/")
                    window = []
                    print(f'epoch={epoch} commands={
                            offset + 1}/{
                            len(indices)} step={step} loss={
                            average["loss/total"]:.4f} micro_seconds={
                            time.perf_counter() - then:.2f}', flush=True)
                if not args.smoke and (offset + 1) % 250 == 0:
                    save_checkpoint(
                        run.path / "last.pt",
                        scorer,
                        adapters,
                        optimizer,
                        epoch - 1,
                        step,
                        corpus["sha256"],
                    )
                    run.save(
                        "progress.json",
                        {
                            "epoch_in_progress": epoch,
                            "commands_in_epoch": offset + 1,
                            "step": step,
                            "history": history,
                        },
                    )
            run.log(
                {
                    k + "_mean": sum(v[k] for v in epoch_values) / len(epoch_values)
                    for k in epoch_values[0]
                },
                epoch,
                "train_epoch/",
            )
            validation = evaluate(
                scorer, val[:4] if args.smoke else val, corpus, run, "validation", step
            )
            score = (validation["macro_candidate_exact"], -validation["loss"])
            improved = score > best
            if improved:
                best = score
                stale = 0
            else:
                stale += 1
            save_checkpoint(
                run.path / "last.pt", scorer, adapters, optimizer, epoch, step, corpus["sha256"]
            )
            if improved:
                save_checkpoint(
                    run.path / "best.pt", scorer, adapters, optimizer, epoch, step, corpus["sha256"]
                )
            # Verify serialization on real forward, not only dictionary shapes.
            scorer.backbone.eval()
            with torch.inference_mode():
                before = logits(scorer, encoded).clone()
            restored = restore(run.path / "last.pt", scorer, adapters)
            with torch.inference_mode():
                torch.testing.assert_close(before, logits(scorer, encoded), rtol=0, atol=0)
            del restored
            history.append(
                {"epoch": epoch, "step": step, "validation": validation, "improved": improved}
            )
            run.save(
                "progress.json",
                {
                    "completed_epoch": epoch,
                    "step": step,
                    "history": history,
                    "reload_logits_bit_exact": True,
                },
            )
            print(f'EPOCH_COMPLETE {epoch} validation={
                    validation["exact"]:.4f} stale={stale}', flush=True)
            if epoch >= 3 and stale >= 2:
                stop = "validation plateau2 checks after at least3 epochs"
                break
        selected = restore(run.path / "best.pt", scorer, adapters)
        lock = {
            "selected_epoch": selected["epoch"],
            "selected_step": selected["step"],
            "checkpoint_sha256": hashlib.sha256((run.path / "best.pt").read_bytes()).hexdigest(),
            "test_used_for_selection": False,
        }
        del selected
        run.save("selection_lock.json", lock)
        final = {}
        if not args.smoke:
            for fold, rows in [
                ("validation_selected", val),
                ("test_regression_selected", corpus["splits"]["test"]),
                ("fresh_test_selected", novel),
                ("train_recall_sample", train[:300]),
            ]:
                final[fold] = evaluate(
                    scorer, rows, corpus, run, fold, lock["selected_step"], seeds=(21001, 21002)
                )
            # Baseline on fresh strings only after selection is irreversibly
            # locked.
            original_path = Path(scorer.info["manifest"]["adapter_directory"]) / "checkpoint.pt"
            source = torch.load(original_path, map_location="cpu", weights_only=True)
            with torch.no_grad():
                for name, a in adapters.items():
                    a.a.copy_(source["adapters"][name]["a"])
                    a.b.copy_(source["adapters"][name]["b"])
            scorer.headq.load_state_dict(source["headq"])
            scorer.headk.load_state_dict(source["headk"])
            del source
            final["fresh_test_original"] = evaluate(
                scorer, novel, corpus, run, "fresh_test_original", 0, seeds=(21001, 21002)
            )
            restore(run.path / "best.pt", scorer, adapters)
        report = {
            **config,
            **lock,
            "run_dir": str(run.path),
            "initial_validation": initial,
            "history": history,
            "metrics": final,
            "completed_epochs": epoch,
            "updates": step,
            "stop_reason": stop,
            "processed_training_commands": epoch * (1 if args.smoke else 1500),
            "processed_training_views": epoch * (2 if args.smoke else 3000),
            "seconds": time.perf_counter() - started,
            "checkpoint_reload_bit_exact": True,
        }
        run.save("report.json", report)
        json_write(
            ROOT
            / "docs"
            / ("rev_finetune_smoke.json" if args.smoke else "rev_finetune_results.json"),
            report,
        )
        print("COMPLETE=" + str(run.path), flush=True)


if __name__ == "__main__":
    main()
