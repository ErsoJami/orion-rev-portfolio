"""Full Smol training on diverse development families; never opens sealed test."""

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch

from experiments.action_quality import quality
from experiments.action_risk import execute_mask, expected_decision_cost
from experiments.action_training import log_messages, module_fingerprint
from experiments.choice_action import shuffled_orders
from experiments.large_choice_eval import make_records
from experiments.rev_finetuning import paired_loss
from experiments.runtime import Run, json_write, seed_all
from scripts.finetune_original_rev import encode
from scripts.train_smol_full_rev import evaluate, forward, load_model, restore, save

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate_quality(scorer, rows, corpus, run, phase, step, seeds=(29001,)):
    result = evaluate(scorer, rows, corpus, run, phase, step, seeds=seeds)
    stored = json.loads((run.path / (phase + ".json")).read_text(encoding="utf-8"))
    result.update(quality(stored["records"], corpus["candidates"]))
    run.log(result, step, phase + "_quality/")
    stored["metrics"] = result
    run.save(phase + ".json", stored)
    return result


def training_objective(logits, targets, orders, corpus, risk_weight):
    loss, ce, kl = paired_loss(logits, targets, orders, 0.1)
    # Log the actual expected cost even in the CE control. Detach only that
    # diagnostic when disabled, preserving the original CE/KL gradient graph.
    risk = expected_decision_cost(
        logits if risk_weight else logits.detach(),
        targets,
        execute_mask(corpus["candidates"], orders, logits.device),
    )
    return loss + risk_weight * risk if risk_weight else loss, ce, kl, risk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume-run", type=Path)
    p.add_argument("--max-updates", type=int, default=1200)
    p.add_argument("--body-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=3081)
    p.add_argument(
        "--dataset", type=Path, default=ROOT / "docs/smol_diverse_v2_train_validation.json"
    )
    p.add_argument("--initial-checkpoint", type=Path)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--run-name", default="Smol-diverse-Rev-8000")
    p.add_argument("--artifact-prefix", default="smol_diverse")
    p.add_argument(
        "--risk-weight",
        type=float,
        default=0.0,
        help="CE+pairedKL plus exact expected0/1/5decisioncost; not publishedRLCD",
    )
    p.add_argument(
        "--final-train-evaluation",
        choices=("full", "probe"),
        default="full",
        help="Optional short objective-ablation report; never label probe as wholetrain",
    )
    args = p.parse_args()
    seed_all(args.seed)
    assert args.eval_every > 0 and not (args.resume_run and args.initial_checkpoint)
    assert math.isfinite(args.risk_weight) and args.risk_weight >= 0
    assert all(c.isalnum() or c == "_" for c in args.artifact_prefix)
    corpus_path = args.dataset.resolve()
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    digest = corpus.pop("sha256")
    assert (
        hashlib.sha256(json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        == digest
    )
    corpus["sha256"] = digest
    assert set(corpus["splits"]) == {"train", "validation"}, "sealed test must not enter trainer"
    initial = json.loads((ROOT / "docs/smol_full_rev_results.json").read_text(encoding="utf-8"))
    initial_path = args.initial_checkpoint or Path(initial["checkpoint"])
    scorer, meta = load_model()
    restore(initial_path, scorer)
    meta.update(
        full_initial_checkpoint=str(initial_path.resolve()),
        full_initial_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
    )
    body = list(scorer.backbone.parameters())
    heads = list(scorer.headq.parameters()) + list(scorer.headk.parameters())
    params = body + heads
    optimizer = torch.optim.AdamW(
        [{"params": body, "lr": args.body_lr}, {"params": heads, "lr": args.head_lr}],
        weight_decay=0.01,
    )
    train = corpus["splits"]["train"]
    validation = corpus["splits"]["validation"]
    probe = train[:600]
    config = {
        "stage": "new semantic families, full Smol+Rev",
        "dataset_path": str(corpus_path),
        "dataset_hash": digest,
        "model": meta,
        "seed": args.seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "full_body_parameters": sum(p.numel() for p in body),
        "head_parameters": sum(p.numel() for p in heads),
        "body_lr": args.body_lr,
        "head_lr": args.head_lr,
        "effective_commands": 8,
        "micro_commands": 4,
        "precision": "FP32 master/AdamW/head; BF16 body activations",
        "paired_CE_KL": 0.1,
        "selection": "minimum validation error_cost=wrong_rate+4*false_execute_rate_over_all_rows; tie lower CE",
        "max_updates": args.max_updates,
        "scheduler": "20 update warmup then cosine down to 0.25 initial LR",
        "eval_every": args.eval_every,
        "artifact_prefix": args.artifact_prefix,
        "risk_weight": args.risk_weight,
        "risk_objective": "exact finite-action expected0/1/5cost, in addition to supervisedCE; not JevRLCD",
        "final_train_evaluation": args.final_train_evaluation,
        "final_train_evaluation_rows": (
            len(train) if args.final_train_evaluation == "full" else len(probe)
        ),
        "test_opened": False,
        "quality_gate": "validation >=98% exact and <=1% false action; independent test still required",
        "goal_complete_on_phase_end": False,
    }
    if args.resume_run:
        config["resume_run"] = str(args.resume_run.resolve())
    sources = (
        Path(__file__),
        ROOT / "scripts/train_smol_full_rev.py",
        ROOT / "experiments/diverse_action_data.py",
        ROOT / "experiments/action_quality.py",
        ROOT / "experiments/action_risk.py",
    )
    is_broad = any(r.get("data_kind") == "broad_synthetic" for r in train)
    if is_broad:
        sources += (
            ROOT / "experiments/broad_action_data.py",
            ROOT / "scripts/create_broad_action_dataset.py",
        )
    with Run((args.run_name + "-smoke") if args.smoke else args.run_name, config) as run:
        for source in sources:
            shutil.copy2(source, run.path / source.name)
        run.save(
            "source_manifest.json",
            {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
        )
        print("RUN=" + str(run.path), flush=True)
        if args.smoke:
            # Independent fresh process checks the saved first-stage checkpoint
            # against the actual300 old validation predictions, before updates.
            if args.initial_checkpoint:
                assert (
                    args.initial_checkpoint.name == "best.pt"
                ), "custom smoke expects saved selected validation of best.pt"
                reference = json.loads(
                    (args.initial_checkpoint.parent / "validation_selected.json").read_text(
                        encoding="utf-8"
                    )
                )["records"][:8]
                assert [r["id"] for r in reference] == [r["id"] for r in validation[:8]]
                evaluate(
                    scorer, validation[:8], corpus, run, "source_cold_reload", 0, seeds=(29001,)
                )
            else:
                old = json.loads(
                    (ROOT / "docs/large_choice_dataset.json").read_text(encoding="utf-8")
                )
                evaluate(
                    scorer,
                    old["splits"]["validation"],
                    old,
                    run,
                    "source_cold_reload",
                    0,
                    seeds=(19001,),
                )
                reference = json.loads(
                    (Path(initial["run_dir"]) / "validation_selected.json").read_text(
                        encoding="utf-8"
                    )
                )["records"][:300]
            reloaded = json.loads(
                (run.path / "source_cold_reload.json").read_text(encoding="utf-8")
            )["records"]
            assert all(
                a["id"] == b["id"] and a["returned"] == b["returned"]
                for a, b in zip(reference, reloaded, strict=True)
            )
            max_prob_difference = max(
                abs(p - q)
                for a, b in zip(reference, reloaded)
                for p, q in zip(a["probabilities"], b["probabilities"])
            )
            # Derivation categories include empty, requested and scope
            # decisions.
            rows = []
            for d in ("EXECUTE", "CLARIFY", "UNSUPPORTED", "NO_ACTION"):
                kind = "broad_synthetic" if is_broad else "synthetic_semantic"
                available = [r for r in train if r["decision"] == d and r.get("data_kind") == kind]
                rows.append(
                    max(available, key=lambda r: len(r["command"]))
                    if kind == "broad_synthetic"
                    else available[0]
                )
            paired = [r for r in rows for _ in range(2)]
            orders = shuffled_orders(paired, args.seed)
            inputs, texts, targets = encode(scorer, paired, orders, corpus)
            scorer.backbone.eval()
            with torch.inference_mode():
                before = forward(scorer, inputs).clone()
            save(
                run.path / "last.pt",
                scorer,
                optimizer,
                config,
                1,
                0,
                0,
                0,
                (float("-inf"), float("-inf")),
                [],
            )
            restore(run.path / "last.pt", scorer, optimizer)
            with torch.inference_mode():
                torch.testing.assert_close(before, forward(scorer, inputs), rtol=0, atol=0)
            scorer.backbone.train()
            optimizer.zero_grad(set_to_none=True)
            logits = forward(scorer, inputs)
            loss, _, _, risk = training_objective(logits, targets, orders, corpus, args.risk_weight)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            log_messages(
                run,
                make_records(logits, paired, orders, texts, corpus["candidates"]),
                "smoke_preupdate",
                1,
                batch_index=0,
            )
            expected = tuple(
                module_fingerprint(m) for m in (scorer.backbone, scorer.headq, scorer.headk)
            )
            save(run.path / "updated.pt", scorer, optimizer, config, 1, 4, 1, 4, (-1.0, 0.0), [])
            restore(run.path / "last.pt", scorer, optimizer)
            scorer.backbone.train()
            logits = forward(scorer, inputs)
            repeat, _, _, _ = training_objective(logits, targets, orders, corpus, args.risk_weight)
            repeat.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            assert (
                tuple(module_fingerprint(m) for m in (scorer.backbone, scorer.headq, scorer.headk))
                == expected
            )
            outcome = {
                "initial_reload_logits_bit_exact": True,
                "resumed_update_all_weights_bit_exact": True,
                "all_trainable_gradients_finite": True,
                "source_cold_reload_choices_identical": True,
                "source_cold_reload_cases": len(reference),
                "source_cold_reload_max_probability_difference": max_prob_difference,
                "loss": float(loss.detach()),
                "expected_decision_cost": float(risk.detach()),
                "risk_weight": args.risk_weight,
                "run_dir": str(run.path),
                "max_encoded_tokens": max(len(x[0]) for x in inputs),
            }
            if len(reference) == 300:
                outcome["source_cold_reload_300_choices_identical"] = True
            run.save("report.json", outcome)
            json_write(ROOT / "docs" / f"{args.artifact_prefix}_smoke.json", outcome)
            print("SMOKE_COMPLETE", json.dumps(outcome), flush=True)
            return
        step = 0
        commands = 0
        epoch = 1
        start_offset = 0
        history = []
        best = (-math.inf, -math.inf)
        if args.resume_run:
            state = restore(args.resume_run / "last.pt", scorer, optimizer)
            assert (
                state["config"]["dataset_hash"] == digest and state["config"]["seed"] == args.seed
            )
            assert (
                state["config"]["body_lr"] == args.body_lr
                and state["config"]["head_lr"] == args.head_lr
            )
            assert (
                state["config"]["max_updates"] == args.max_updates
            ), "resume must preserve the cosine schedule budget"
            assert state["config"].get("eval_every", 200) == args.eval_every
            assert (
                state["config"].get("risk_weight", 0.0) == args.risk_weight
            ), "Resume must preserve the training objective"
            step = state["step"]
            commands = state["commands"]
            epoch = state["epoch"]
            start_offset = state["next_offset"]
            history = state["history"]
            best = tuple(state["best"])
            shutil.copy2(args.resume_run / "best.pt", run.path / "best.pt")
            # Preserve the original measured baseline for reports, without
            # rerunning inference or adding duplicate TensorBoard messages.
            shutil.copy2(
                args.resume_run / "validation_initial.json", run.path / "validation_initial.json"
            )
            run.save(
                "resume_provenance.json",
                {
                    "source_run": str(args.resume_run.resolve()),
                    "restored_step": step,
                    "restored_commands": commands,
                    "epoch": epoch,
                    "next_offset": start_offset,
                    "baseline_copied_from": str(args.resume_run / "validation_initial.json"),
                    "baseline_remeasured": False,
                },
            )
        else:
            val = evaluate_quality(scorer, validation, corpus, run, "validation_initial", 0)
            best = (-val["error_cost"], -val["loss"])
            history.append(
                {"step": 0, "commands": 0, "epoch": 0, "validation": val, "train_probe": None}
            )
            save(run.path / "best.pt", scorer, None, config, 0, 0, 0, 0, best, history)
            save(run.path / "last.pt", scorer, optimizer, config, 1, 0, 0, 0, best, history)
        started = time.perf_counter()
        window = []
        finished = False
        stop = "phase update budget reached; goal remains active"

        def progress(state="training"):
            result = {
                "state": state,
                "run_dir": str(run.path),
                "epoch": epoch,
                "step": step,
                "commands": commands,
                "history": history,
                "checkpoint": str(run.path / "last.pt"),
                "elapsed_seconds": time.perf_counter() - started,
                "config": config,
            }
            run.save("progress.json", result)
            json_write(ROOT / "docs" / f"{args.artifact_prefix}_live.json", result)

        progress()
        while step < args.max_updates and not finished:
            indices = list(range(len(train)))
            random.Random(args.seed + epoch).shuffle(indices)
            for offset in range(start_offset, len(train), 8):
                chosen = indices[offset : offset + 8]
                n = len(chosen)
                scorer.backbone.train()
                for micro in range(0, n, 4):
                    rows = [train[i] for i in chosen[micro : micro + 4]]
                    paired = [r for r in rows for _ in range(2)]
                    orders = shuffled_orders(paired, args.seed + epoch * 100000 + offset + micro)
                    inputs, texts, targets = encode(scorer, paired, orders, corpus)
                    output = forward(scorer, inputs)
                    loss, ce, kl, risk = training_objective(
                        output, targets, orders, corpus, args.risk_weight
                    )
                    (loss * len(rows) / n).backward()
                    log_messages(
                        run,
                        make_records(output, paired, orders, texts, corpus["candidates"]),
                        "train_preupdate",
                        step + 1,
                        batch_index=offset + micro,
                    )
                    window.append(
                        (
                            len(rows),
                            float(loss.detach()),
                            float(ce.detach()),
                            float(kl.detach()),
                            float(risk.detach()),
                        )
                    )
                step += 1
                commands += n
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                factor = min(1.0, step / 20) * (
                    0.25
                    + 0.75
                    * 0.5
                    * (1 + math.cos(math.pi * min(step, args.max_updates) / args.max_updates))
                )
                for group, lr in zip(optimizer.param_groups, (args.body_lr, args.head_lr)):
                    group["lr"] = lr * factor
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step % 25 == 0 or offset + n == len(train):
                    size = sum(w[0] for w in window)
                    vals = {
                        name: sum(w[0] * w[i] for w in window) / size
                        for i, name in enumerate(
                            (
                                "loss/total_mean",
                                "loss/ce_mean",
                                "loss/consistency_mean",
                                "loss/expected_cost_mean",
                            ),
                            1,
                        )
                    }
                    run.log(
                        {
                            **vals,
                            "gradient_norm": float(norm),
                            "lr_factor": factor,
                            "commands": commands,
                            "epoch": epoch,
                        },
                        step,
                        "train_window/",
                    )
                    window = []
                    progress()
                    print(f'TRAIN epoch={epoch} step={step} commands={commands} loss={
                            vals["loss/total_mean"]:.4f}', flush=True)
                requested = (run.path / "STOP_REQUESTED").exists()
                if (
                    step % args.eval_every == 0
                    or offset + n == len(train)
                    or step == args.max_updates
                    or requested
                ):
                    val = evaluate_quality(scorer, validation, corpus, run, "validation", step)
                    recall = evaluate_quality(
                        scorer, probe, corpus, run, "train_probe", step, seeds=(61001,)
                    )
                    history.append(
                        {
                            "epoch": epoch,
                            "step": step,
                            "commands": commands,
                            "validation": val,
                            "train_probe": recall,
                        }
                    )
                    score = (-val["error_cost"], -val["loss"])
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
                    progress()
                    if val["exact"] >= 0.98 and val["false_action_rate"] <= 0.01:
                        stop = "development quality gate passed; next is sealed test"
                        finished = True
                    if requested:
                        stop = "STOP_REQUESTED saved checkpoint"
                        finished = True
                if finished or step >= args.max_updates:
                    break
            epoch += 1
            start_offset = 0
        selected = restore(run.path / "best.pt", scorer)
        val = evaluate_quality(
            scorer,
            validation,
            corpus,
            run,
            "validation_selected",
            selected["step"],
            seeds=(29001, 29002),
        )
        train_phase = (
            "train_selected" if args.final_train_evaluation == "full" else "train_selected_probe"
        )
        final_train_rows = train if args.final_train_evaluation == "full" else probe
        recall = evaluate_quality(
            scorer, final_train_rows, corpus, run, train_phase, selected["step"], seeds=(61001,)
        )
        report = {
            **config,
            "run_dir": str(run.path),
            "updates": step,
            "commands": commands,
            "selected_step": selected["step"],
            "validation": val,
            "train": recall,
            "train_evaluation_phase": train_phase,
            "history": history,
            "stop_reason": stop,
            "checkpoint": str(run.path / "best.pt"),
            "resume_checkpoint": str(run.path / "last.pt"),
            "elapsed_seconds": time.perf_counter() - started,
            "status": "phase complete; goal not complete",
        }
        run.save("report.json", report)
        json_write(ROOT / "docs" / f"{args.artifact_prefix}_results.json", report)
        json_write(ROOT / "docs" / f"{
                args.artifact_prefix}_live.json", {"state": "phase_complete", **report})
        print("PHASE_COMPLETE=" + str(run.path), flush=True)


if __name__ == "__main__":
    main()
