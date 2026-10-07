"""Full FP32-master Smol tuning with the verified Rev pointer and BF16 activations."""

import argparse
import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch

from experiments.action_training import log_messages, module_fingerprint
from experiments.choice_action import shuffled_orders
from experiments.large_choice_eval import make_records, metrics
from experiments.rev_finetuning import paired_loss
from experiments.runtime import Run, json_write, load_baseline, seed_all
from scripts.finetune_original_rev import encode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_model():
    source = json.loads((ROOT / "docs/smol_native_rev_results.json").read_text(encoding="utf-8"))
    source_checkpoint = Path(source["checkpoint"])
    if not source_checkpoint.is_absolute():
        source_checkpoint = (ROOT / source_checkpoint).resolve()
    if not source_checkpoint.exists():
        source_checkpoint = ROOT.parent / "weights" / "native_rev_initial.pt"
    state = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    full, tokenizer, meta = load_baseline()
    backbone = full.model
    del full
    backbone.float()
    with torch.no_grad():
        for name, weights in state["adapters"].items():
            layer = backbone.get_submodule(name)
            layer.weight.add_((2 * weights["b"].float() @ weights["a"].float()).to(layer.weight))
    q = torch.nn.Linear(576, 256, bias=False, device="cuda")
    k = torch.nn.Linear(576, 256, bias=False, device="cuda")
    q.load_state_dict(state["headq"])
    k.load_state_dict(state["headk"])
    spec = importlib.util.spec_from_file_location(
        "full_native_rev", ROOT / "third_party/rev/serve_local.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    scorer = upstream.Scorer(tokenizer, backbone, q, k, {"state_format": "raw", "temperature": 1.0})
    backbone.requires_grad_(True)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    meta.update(
        source_checkpoint=str(source_checkpoint),
        source_sha256=hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
        initialization="native Smol Rev; merge existing LoRA in FP32, then full decoder tuning",
        precision_note="FP32 masters/residuals change numerical forward from original BF16 source; evaluate initial state anew",
    )
    return scorer, meta


def forward(scorer, encoded):
    length = ((max(len(ids) for ids, _ in encoded) + 15) // 16) * 16
    inputs = torch.full((len(encoded), length), scorer.pad_id, dtype=torch.long, device="cuda")
    for i, (ids, _) in enumerate(encoded):
        inputs[i, : len(ids)] = torch.tensor(ids, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = scorer.backbone(
            input_ids=inputs, attention_mask=None, use_cache=False
        ).last_hidden_state
    return torch.stack(
        [
            (
                scorer.headk(hidden[i, pos].float()) * scorer.headq(hidden[i, len(ids) - 1].float())
            ).sum(-1)
            / 16
            for i, (ids, pos) in enumerate(encoded)
        ]
    )


@torch.inference_mode()
def evaluate(scorer, rows, corpus, run, phase, step, seeds=(19001,), batch=8):
    scorer.backbone.eval()
    scorer.headq.eval()
    scorer.headk.eval()
    records = []
    for view, seed in enumerate(seeds):
        orders = []
        for j in range(0, len(rows), 12):
            orders.extend(shuffled_orders(rows[j : j + 12], seed + j))
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            order = orders[start : start + len(chunk)]
            inputs, texts, _ = encode(scorer, chunk, order, corpus)
            now = make_records(forward(scorer, inputs), chunk, order, texts, corpus["candidates"])
            log_messages(run, now, phase, step, batch_index=view * 10000 + start)
            records.extend(now)
    result = metrics(records)
    run.log(result, step, phase + "/")
    run.save(phase + ".json", {"metrics": result, "records": records})
    print(f'EVAL {phase} step={step} exact={
            result["exact"]:.4f} false_action={
            result["false_action_rate"]}', flush=True)
    return result


def save(path, scorer, optimizer, config, epoch, offset, step, commands, best, history):
    state = {
        "format": "smol-full-rev-v1",
        "backbone": {n: p.detach().cpu() for n, p in scorer.backbone.state_dict().items()},
        "headq": {n: p.detach().cpu() for n, p in scorer.headq.state_dict().items()},
        "headk": {n: p.detach().cpu() for n, p in scorer.headk.state_dict().items()},
        "config": config,
        "epoch": epoch,
        "next_offset": offset,
        "step": step,
        "commands": commands,
        "best": best,
        "history": history,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(".partial")
    torch.save(state, tmp)
    tmp.replace(path)


def restore(path, scorer, optimizer=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["format"] == "smol-full-rev-v1"
    scorer.backbone.load_state_dict(state["backbone"])
    scorer.headq.load_state_dict(state["headq"])
    scorer.headk.load_state_dict(state["headk"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return {
        k: v
        for k, v in state.items()
        if k not in ("backbone", "headq", "headk", "optimizer", "rng", "cuda_rng")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--body-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--micro-commands", type=int, default=4, choices=(1, 2, 4, 8))
    args = parser.parse_args()
    seed_all(2027)
    corpus = json.loads((ROOT / "docs/large_choice_dataset.json").read_text(encoding="utf-8"))
    scorer, meta = load_model()
    body = list(scorer.backbone.parameters())
    hp = list(scorer.headq.parameters()) + list(scorer.headk.parameters())
    params = body + hp
    assert all(p.dtype == torch.float32 and p.requires_grad for p in params)
    optimizer = torch.optim.AdamW(
        [{"params": body, "lr": args.body_lr}, {"params": hp, "lr": args.head_lr}],
        weight_decay=0.01,
    )
    config = {
        "model": meta,
        "dataset_hash": corpus["sha256"],
        "train_rows": 1500,
        "validation_rows": 300,
        "body_lr": args.body_lr,
        "head_lr": args.head_lr,
        "full_body_parameters": sum(p.numel() for p in body),
        "head_parameters": sum(p.numel() for p in hp),
        "fp32_masters": True,
        "bf16_activations": True,
        "effective_commands": 8,
        "microbatch_commands": args.micro_commands,
        "paired_orders": True,
        "KL_weight": 0.1,
        "max_epochs": args.max_epochs,
        "train_fit_gate": 0.99,
        "selection": "validation macro26 exact, tie lower CE",
        "test_opened": False,
        "dataset_expansion_after_fit": True,
        "seed": 2027,
        "goal": "useful accurately trained Smol; train fit is only the first gate, not goal completion",
    }
    name = "Smol-full-Rev-smoke" if args.smoke else "Smol-full-Rev-fit1500"
    with Run(name, config) as run:
        print("RUN_DIR=" + str(run.path), flush=True)
        run.save("dataset.json", corpus)
        sample = corpus["splits"]["train"][: args.micro_commands]
        paired = [r for r in sample for _ in range(2)]
        inputs, texts, targets = encode(scorer, paired, shuffled_orders(paired, 2027), corpus)
        if args.smoke:
            scorer.backbone.train()
            output = forward(scorer, inputs)
            loss, _, _ = paired_loss(output, targets, shuffled_orders(paired, 2027), 0.1)
            loss.backward()
            grads = {
                n: float(p.grad.abs().sum()) if p.grad is not None else 0
                for n, p in scorer.backbone.named_parameters()
            }
            blocks = [
                any(
                    value > 0
                    for n, value in grads.items()
                    if n.startswith("layers." + str(i) + ".")
                )
                for i in range(30)
            ]
            assert all(blocks) and grads["embed_tokens.weight"] > 0
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in hp)
            probe = scorer.backbone.layers[0].self_attn.q_proj.weight.detach().clone()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            assert not torch.equal(probe, scorer.backbone.layers[0].self_attn.q_proj.weight)
            scorer.backbone.eval()
            with torch.inference_mode():
                before = forward(scorer, inputs).clone()
            save(run.path / "last.pt", scorer, optimizer, config, 1, 2, 1, 2, (-1.0, 0.0), [])
            restore(run.path / "last.pt", scorer, optimizer)
            with torch.inference_mode():
                after = forward(scorer, inputs)
            torch.testing.assert_close(before, after, rtol=0, atol=0)

            def next_update():
                scorer.backbone.train()
                optimizer.zero_grad(set_to_none=True)
                next_loss, _, _ = paired_loss(
                    forward(scorer, inputs), targets, shuffled_orders(paired, 2027), 0.1
                )
                next_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                return tuple(
                    module_fingerprint(m) for m in (scorer.backbone, scorer.headq, scorer.headk)
                )

            expected_update = next_update()
            restore(run.path / "last.pt", scorer, optimizer)
            assert next_update() == expected_update
            report = {
                "all30blocks_have_gradients": True,
                "embedding_has_gradient": True,
                "both_pointer_projections_have_gradients": True,
                "fp32_body_weight_changed": True,
                "checkpoint_optimizer_reload_logits_bit_exact": True,
                "resumed_next_optimizer_update_all_weights_bit_exact": True,
                "trainable_parameters": sum(p.numel() for p in params),
                "loss": float(loss.detach()),
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "run_dir": str(run.path),
            }
            run.save("report.json", report)
            json_write(ROOT / "docs/smol_full_rev_smoke.json", report)
            print("SMOKE_COMPLETE=" + json.dumps(report), flush=True)
            return
        initial = evaluate(
            scorer, corpus["splits"]["validation"], corpus, run, "validation_initial", 0
        )
        best = (initial["macro_candidate_exact"], -initial["loss"])
        history = []
        step = 0
        commands = 0
        start_epoch = 1
        start_offset = 0
        if args.resume_run:
            prior = args.resume_run / "last.pt"
            state = restore(prior, scorer, optimizer)
            if state["config"]["dataset_hash"] != config["dataset_hash"]:
                raise ValueError("resume dataset mismatch")
            best = tuple(state["best"])
            history = state["history"]
            step = state["step"]
            commands = state["commands"]
            start_epoch = state["epoch"]
            start_offset = state["next_offset"]
            previous_best = args.resume_run / "best.pt"
            import shutil

            shutil.copy2(previous_best, run.path / "best.pt")
        else:
            save(run.path / "best.pt", scorer, None, config, 0, 0, 0, 0, best, history)
        stop = "phase budget reached; inspect fit and adjust learning/data before continuing goal"
        started = time.perf_counter()
        window = []
        finished = False
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, args.max_epochs + 1):
            indices = list(range(1500))
            random.Random(2027 + epoch).shuffle(indices)
            for offset in range(start_offset if epoch == start_epoch else 0, 1500, 8):
                chosen = indices[offset : offset + 8]
                n = len(chosen)
                orders_done = []
                scorer.backbone.train()
                for micro in range(0, n, args.micro_commands):
                    rows = [
                        corpus["splits"]["train"][i]
                        for i in chosen[micro : micro + args.micro_commands]
                    ]
                    paired = [r for r in rows for _ in range(2)]
                    order = shuffled_orders(paired, 2027 + epoch * 100000 + offset + micro)
                    encoded, texts, targets = encode(scorer, paired, order, corpus)
                    output = forward(scorer, encoded)
                    loss, ce, kl = paired_loss(output, targets, order, 0.1)
                    (loss * len(rows) / n).backward()
                    log_messages(
                        run,
                        make_records(output, paired, order, texts, corpus["candidates"]),
                        "train_preupdate",
                        step + 1,
                        batch_index=offset + micro,
                    )
                    window.append(
                        (
                            len(rows),
                            {
                                "loss/total": float(loss.detach()),
                                "loss/ce": float(ce.detach()),
                                "loss/consistency": float(kl.detach()),
                            },
                        )
                    )
                step += 1
                commands += n
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                for group, lr in zip(optimizer.param_groups, (args.body_lr, args.head_lr)):
                    group["lr"] = lr * min(1.0, step / 30)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step % 25 == 0 or offset + n == 1500:
                    size = sum(size for size, _ in window)
                    values = {
                        k + "_mean": sum(size * v[k] for size, v in window) / size
                        for k in window[0][1]
                    }
                    run.log(
                        {
                            **values,
                            "gradient_norm": float(norm),
                            "commands_seen": commands,
                            "epoch": epoch,
                        },
                        step,
                        "train_window/",
                    )
                    window = []
                    print(f'TRAIN epoch={epoch} step={step} commands={commands} loss={
                            values["loss/total_mean"]:.4f}', flush=True)
                stopped = (run.path / "STOP_REQUESTED").exists()
                if step % 100 == 0 or offset + n == 1500 or stopped:
                    val = evaluate(
                        scorer, corpus["splits"]["validation"], corpus, run, "validation", step
                    )
                    recall = evaluate(
                        scorer,
                        corpus["splits"]["train"][:300],
                        corpus,
                        run,
                        "train_probe",
                        step,
                        seeds=(51001,),
                    )
                    full_recall = None
                    if recall["exact"] >= 0.99 or offset + n == 1500:
                        full_recall = evaluate(
                            scorer,
                            corpus["splits"]["train"],
                            corpus,
                            run,
                            "train_full",
                            step,
                            seeds=(51001,),
                        )
                    score = (val["macro_candidate_exact"], -val["loss"])
                    history.append(
                        {
                            "epoch": epoch,
                            "step": step,
                            "commands": commands,
                            "validation": val,
                            "train_probe": recall,
                            "train_full": full_recall,
                        }
                    )
                    if score > best:
                        best = score
                        save(
                            run.path / "best.pt",
                            scorer,
                            None,
                            config,
                            epoch,
                            offset + n,
                            step,
                            commands,
                            best,
                            history,
                        )
                    save(
                        run.path / "last.pt",
                        scorer,
                        optimizer,
                        config,
                        epoch,
                        offset + n,
                        step,
                        commands,
                        best,
                        history,
                    )
                    progress = {
                        "state": "training",
                        "run_dir": str(run.path),
                        "epoch": epoch,
                        "step": step,
                        "commands": commands,
                        "elapsed_seconds": time.perf_counter() - started,
                        "history": history,
                        "checkpoint": str(run.path / "last.pt"),
                    }
                    run.save("progress.json", progress)
                    json_write(ROOT / "docs/smol_full_rev_live.json", progress)
                    if full_recall and full_recall["exact"] >= 0.99:
                        stop = "full train fit>=99%; advance to data/generalization stage"
                        finished = True
                    if stopped:
                        stop = "STOP_REQUESTED; saved resumable optimizer checkpoint"
                        finished = True
                    if finished:
                        break
            start_offset = 0
            if finished:
                break
        last = history[-1] if history else None
        selected = restore(run.path / "best.pt", scorer)
        selected_val = evaluate(
            scorer,
            corpus["splits"]["validation"],
            corpus,
            run,
            "validation_selected",
            selected["step"],
            seeds=(19001, 19002),
        )
        selected_train = evaluate(
            scorer,
            corpus["splits"]["train"],
            corpus,
            run,
            "train_selected",
            selected["step"],
            seeds=(51001, 51002),
        )
        report = {
            **config,
            "run_dir": str(run.path),
            "updates": step,
            "commands": commands,
            "selected_step": selected["step"],
            "history": history,
            "last_observation": last,
            "validation": selected_val,
            "train": selected_train,
            "stop_reason": stop,
            "checkpoint": str(run.path / "best.pt"),
            "resume_checkpoint": str(run.path / "last.pt"),
            "elapsed_seconds": time.perf_counter() - started,
            "status": "phase complete; full research goal remains active",
        }
        run.save("report.json", report)
        json_write(ROOT / "docs/smol_full_rev_results.json", report)
        json_write(ROOT / "docs/smol_full_rev_live.json", {"state": "phase_complete", **report})
        print("PHASE_COMPLETE=" + str(run.path), flush=True)


if __name__ == "__main__":
    main()
