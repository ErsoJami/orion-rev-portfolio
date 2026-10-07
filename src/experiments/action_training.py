"""Action-only pilot: FP32 optimizer parameters, full message logs, closed test gate."""

from __future__ import annotations

import hashlib
import html
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .action_data import DECISIONS, REASONS
from .heads import ActionHead, action_loss
from .loss_curves import LossCurves
from .runtime import MODEL_REVISION, Run, load_baseline, seed_all, sync
from .tasks import FIELDS, TOOL_FIELD, TOOLS, encode_batch, make_plan, plan_match
from .token_action_head import TokenActionHead
from .training import select_trainable


class DecisionActionHead(ActionHead):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        self.decision = torch.nn.Linear(hidden_size, len(DECISIONS))

    def forward(self, hidden):
        result = super().forward(hidden)
        result["decision"] = self.decision(hidden.float())
        return result


@dataclass
class ActionConfig:
    steps: int = 500
    batch_size: int = 8
    backbone_lr: float = 2e-4
    head_lr: float = 1e-3
    train_last_layers: int = 2
    eval_every: int = 100
    log_every: int = 10
    max_tokens: int = 512
    seed: int = 41
    device: str = "cuda"
    dtype: str = "bfloat16"
    head_kind: str = "pooled"
    head_width: int = 192
    head_query_std: float = 1.0
    loss_curve: str = "both"
    loss_interval: int = 25


def action_variant(config):
    """Keep artifacts for distinct backbone training policies separate."""
    suffix = "-attention" if config.head_kind == "attention" else ""
    if config.train_last_layers == 0:
        suffix += "-frozen"
    elif config.train_last_layers != 2:
        suffix += f"-layers{config.train_last_layers}"
    return suffix


def module_fingerprint(module):
    """Hash every unique parameter and buffer, including names, shapes and dtypes."""
    digest = hashlib.sha256()
    for name, tensor in (*module.named_parameters(), *module.named_buffers()):
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        digest.update(
            tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        )
    return digest.hexdigest()


def loss_for(logits, rows, batch):
    action, terms = action_loss(logits, batch["action_labels"])
    status = torch.tensor(
        [DECISIONS.index(r["decision"]) for r in rows], device=logits["decision"].device
    )
    decision = F.cross_entropy(logits["decision"].float(), status)
    return action + decision, {**terms, "loss/action_decision": float(decision.detach())}


def decode(logits, rows):
    values = {k: v.detach().argmax(-1).cpu().tolist() for k, v in logits.items()}
    predictions = []
    for b, row in enumerate(rows):
        decision = DECISIONS[values["decision"][b]]
        raw_count = values["count"][b]
        # Typed adapter guarantees a valid shape, not semantic correctness.
        count = max(1, raw_count) if decision == "EXECUTE" else 0
        actions = []
        for i in range(count):
            tool = TOOLS[values["tool"][b][i]]
            field = TOOL_FIELD.get(tool)
            actions.append(
                {
                    "tool": tool,
                    "arguments": {field: FIELDS[field][values[field][b][i]]} if field else {},
                }
            )
        plan = make_plan(row["command"], actions)
        if count == 0:
            plan["clarification"] = REASONS[decision]
        predictions.append(
            {
                "decision": decision,
                "plan": plan,
                "raw_count": raw_count,
                "raw_status_count_consistent": (raw_count > 0) == (decision == "EXECUTE"),
            }
        )
    return predictions


def record_for(row, prediction):
    expected_actions = row["plan"]["actions"]
    actual_actions = prediction["plan"]["actions"]
    status_correct = row["decision"] == prediction["decision"]
    plan_correct = plan_match(prediction["plan"], row["plan"])
    return {
        "id": row["id"],
        "split": row["split"],
        "input": row["command"],
        "expected": {"decision": row["decision"], "plan": row["plan"]},
        "returned": prediction,
        "correct": {
            "decision": status_correct,
            "count": len(expected_actions) == len(actual_actions),
            "tools": [a["tool"] for a in expected_actions] == [a["tool"] for a in actual_actions],
            "arguments": [a["arguments"] for a in expected_actions]
            == [a["arguments"] for a in actual_actions],
            "plan": plan_correct,
            "all": status_correct and plan_correct,
        },
    }


def log_messages(run, records, phase, step, *, batch_index=None):
    """Every processed row, untruncated. One TB text event contains an entire batch."""
    with (run.path / "messages.jsonl").open("a", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps({"phase": phase, "step": step, **record}, ensure_ascii=False) + "\n"
            )
    blocks = []
    for record in records:
        body = json.dumps(record, ensure_ascii=False, indent=2)
        blocks.append("<pre>" + html.escape(body) + "</pre>")
    tag = "messages/" + phase + (f"/batch_{batch_index:03d}" if batch_index is not None else "")
    run.writer.add_text(tag, "\n\n".join(blocks), global_step=step)


def _amp(config):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if config.device == "cuda" and config.dtype == "bfloat16"
        else nullcontext()
    )


def forward(model, head, batch, config):
    with _amp(config):
        hidden = model.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
        ).last_hidden_state
    pooled = hidden[
        torch.arange(hidden.shape[0], device=hidden.device), batch["prompt_lengths"] - 1
    ]
    # Outside autocast: count, status and argument logits / CE remain FP32.
    return (
        head(hidden, batch["attention_mask"]) if config.head_kind == "attention" else head(pooled)
    )


@torch.inference_mode()
def evaluate(model, tokenizer, head, rows, config, run=None, phase="validation", step=0):
    model.eval()
    head.eval()
    records, loss_sum = [], 0.0
    for start in range(0, len(rows), config.batch_size):
        chunk = rows[start : start + config.batch_size]
        batch = encode_batch(tokenizer, chunk, "action", config.device, config.max_tokens)
        logits = forward(model, head, batch, config)
        loss, _ = loss_for(logits, chunk, batch)
        loss_sum += float(loss) * len(chunk)
        predictions = decode(logits, chunk)
        current = [record_for(r, p) for r, p in zip(chunk, predictions, strict=True)]
        records.extend(current)
        if run:
            log_messages(run, current, phase, step, batch_index=start // config.batch_size)
    executable = [r for r in records if r["expected"]["decision"] == "EXECUTE"]
    empty = [r for r in records if not r["expected"]["plan"]["actions"]]
    metrics = {
        "rows": len(rows),
        "loss": loss_sum / len(rows),
        "exact": sum(r["correct"]["all"] for r in records) / len(rows),
        "executable_exact": (
            sum(r["correct"]["all"] for r in executable) / len(executable) if executable else 0.0
        ),
        "decision_accuracy": sum(r["correct"]["decision"] for r in records) / len(rows),
        "count_accuracy": sum(r["correct"]["count"] for r in records) / len(rows),
        "false_action_rate": (
            sum(bool(r["returned"]["plan"]["actions"]) for r in empty) / len(empty)
            if empty
            else 0.0
        ),
        "missed_action_rate": (
            sum(not r["returned"]["plan"]["actions"] for r in executable) / len(executable)
            if executable
            else 0.0
        ),
        "raw_status_count_consistency": sum(
            r["returned"]["raw_status_count_consistent"] for r in records
        )
        / len(rows),
    }
    for n in range(6):
        subset = [r for r in records if len(r["expected"]["plan"]["actions"]) == n]
        if subset:
            metrics[f"exact_count_{n}"] = sum(r["correct"]["all"] for r in subset) / len(subset)
    if run:
        run.log(metrics, step, phase + "/")
        run.writer.flush()
    return {"metrics": metrics, "records": records}


def save(
    path, model, head, optimizer, config, dataset_hash, step, order, cursor, rng, best, curves=None
):
    state = {
        "format": "action-v2",
        "revision": MODEL_REVISION,
        "config": asdict(config),
        "dataset_hash": dataset_hash,
        "step": step,
        "best": best,
        "adapted": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "head": {n: p.detach().cpu() for n, p in head.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "order": order,
        "cursor": cursor,
        "sampler_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loss_curves": curves.state_dict() if curves else None,
    }
    temporary = Path(str(path) + ".partial")
    torch.save(state, temporary)
    os.replace(temporary, path)


def setup(config):
    model, tokenizer, metadata = load_baseline(device=config.device, dtype=config.dtype)
    if config.head_kind not in ("pooled", "attention"):
        raise ValueError("unknown action head kind")
    head = (
        TokenActionHead(model.config.hidden_size, config.head_width, config.head_query_std)
        if config.head_kind == "attention"
        else DecisionActionHead(model.config.hidden_size)
    ).to(config.device)
    select_trainable(model, head, config.train_last_layers, "action")
    # AdamW must update full-precision master parameters. BF16 weight-only AdamW
    # can quantize small changes away, especially during tiny-set overfitting.
    for layer in (
        model.model.layers[-config.train_last_layers :] if config.train_last_layers else []
    ):
        layer.float()
    return model, tokenizer, head, metadata


def restore(checkpoint, model, head):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if state["format"] != "action-v2" or state["revision"] != MODEL_REVISION:
        raise ValueError("incompatible action checkpoint")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["adapted"].items():
            parameters[name].copy_(value.to(parameters[name]))
    head.load_state_dict(state["head"])
    return state


def train_action(
    train_rows, validation_rows, config, dataset_hash, *, name="action-pilot", resume_from=None
):
    if config.steps < 1 or config.batch_size < 1 or config.eval_every < 1 or config.log_every < 1:
        raise ValueError("invalid action training configuration")
    if not train_rows or not validation_rows:
        raise ValueError("training and validation must be nonempty")
    seed_all(config.seed)
    model, tokenizer, head, metadata = setup(config)
    backbone_parameters = [p for p in model.parameters() if p.requires_grad]
    parameters = backbone_parameters + list(head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.backbone_lr},
            {"params": head.parameters(), "lr": config.head_lr},
        ],
        weight_decay=0.01,
    )
    rng = random.Random(config.seed)
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    cursor, start_step, best = 0, 0, -1.0
    curves = LossCurves(config.loss_curve, config.loss_interval)
    if resume_from:
        state = restore(resume_from, model, head)
        if state["dataset_hash"] != dataset_hash:
            raise ValueError("resume dataset mismatch")
        for key, value in asdict(config).items():
            if (
                key not in ("steps", "log_every", "eval_every", "loss_curve", "loss_interval")
                and state["config"].get(key, asdict(ActionConfig())[key]) != value
            ):
                raise ValueError("resume configuration mismatch: " + key)
        if config.steps <= state["step"]:
            raise ValueError("resume requires additional steps")
        optimizer.load_state_dict(state["optimizer"])
        order, cursor, start_step, best = (
            state["order"],
            state["cursor"],
            state["step"],
            state["best"],
        )
        rng.setstate(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state.get("loss_curves"):
            curves.load_state_dict(state["loss_curves"])
            if state["config"].get("loss_interval", 25) != config.loss_interval:
                curves.window, curves.window_steps, curves.previous_window = {}, 0, None
    backbone_before = module_fingerprint(model) if config.train_last_layers == 0 else None
    head_before = module_fingerprint(head) if backbone_before else None
    if backbone_before:
        assert not backbone_parameters
    timings, visited = [], set()
    started = time.perf_counter()
    with Run(
        name,
        {
            **asdict(config),
            **metadata,
            "dataset_hash": dataset_hash,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "head_parameters": sum(p.numel() for p in head.parameters()),
            "backbone_trainable_parameters": sum(p.numel() for p in backbone_parameters),
            "backbone_fingerprint_before": backbone_before,
            "precision": "frozen BF16, trainable FP32 masters, BF16 CUDA autocast, FP32 head",
            "objective": "count CE + tool CE + mean applicable argument CE + decision CE",
            "message_log": "all processed rows; pre-update training predictions; no sampling or truncation",
            "resume_from": str(resume_from) if resume_from else None,
        },
    ) as run:
        layout = {}
        if config.loss_curve in ("window", "both"):
            layout["Mean over updates"] = [
                "Multiline",
                ["train_window/loss/total_mean", "validation/loss"],
            ]
        if config.loss_curve in ("epoch", "both"):
            layout["Mean over epoch"] = ["Multiline", ["train_epoch/loss/total_mean"]]
        if layout:
            run.writer.add_custom_scalars({"Loss trend": layout})
        if resume_from:
            import shutil

            previous = Path(resume_from).parent / "best.pt"
            if previous.exists():
                shutil.copyfile(previous, run.path / "best.pt")
            else:
                best = -1.0
        initial = evaluate(model, tokenizer, head, validation_rows, config, run, step=start_step)
        run.save("validation_initial.json", initial)
        for step in range(start_step + 1, config.steps + 1):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            indices = order[cursor : cursor + config.batch_size]
            cursor += len(indices)
            visited.update(indices)
            rows = [train_rows[i] for i in indices]
            batch = encode_batch(tokenizer, rows, "action", config.device, config.max_tokens)
            model.train(config.train_last_layers > 0)
            head.train()
            optimizer.zero_grad(set_to_none=True)
            sync(config.device)
            timer = time.perf_counter()
            logits = forward(model, head, batch, config)
            loss, terms = loss_for(logits, rows, batch)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            warmup = min(1.0, step / 20)
            for group, rate in zip(
                optimizer.param_groups, (config.backbone_lr, config.head_lr), strict=True
            ):
                group["lr"] = rate * warmup
            optimizer.step()
            sync(config.device)
            timings.append((time.perf_counter() - timer) * 1000)
            predictions = decode(logits, rows)
            records = [record_for(r, p) for r, p in zip(rows, predictions, strict=True)]
            log_messages(run, records, "train_preupdate", step)
            metrics = {
                "loss/total": float(loss.detach()),
                **terms,
                "gradient_norm": float(norm),
                "step_ms": timings[-1],
                "exact": sum(r["correct"]["all"] for r in records) / len(records),
                "backbone_lr": optimizer.param_groups[0]["lr"] if backbone_parameters else 0.0,
                "head_lr": optimizer.param_groups[1]["lr"],
                "tokens": int(batch["attention_mask"].sum()),
                "examples": len(rows),
            }
            run.log(
                metrics,
                step,
                "train/",
                scalar_keys=(
                    None
                    if config.loss_curve == "raw"
                    else {k for k in metrics if not k.startswith("loss/")}
                ),
            )
            curves.update(
                run,
                {k: v for k, v in metrics.items() if k.startswith("loss/")},
                step,
                len(rows),
                epoch_end=cursor >= len(order),
                epoch_number=(step - 1)
                // ((len(order) + config.batch_size - 1) // config.batch_size)
                + 1,
                final=step == config.steps,
            )
            if step == 1 or step % config.log_every == 0:
                print(
                    f"{name}: {step}/{config.steps} loss={float(loss.detach()):.4f} update_ms={timings[-1]:.1f}",
                    flush=True,
                )
            if step % config.eval_every == 0 or step == config.steps:
                result = evaluate(model, tokenizer, head, validation_rows, config, run, step=step)
                score = result["metrics"]["exact"]
                print(f"{name}: validation exact={
                        score:.3f}, executable={
                        result['metrics']['executable_exact']:.3f}, false_actions={
                        result['metrics']['false_action_rate']:.3f}", flush=True)
                run.save(f"validation_step_{step}.json", result)
                if score > best:
                    best = score
                    save(
                        run.path / "best.pt",
                        model,
                        head,
                        optimizer,
                        config,
                        dataset_hash,
                        step,
                        order,
                        cursor,
                        rng,
                        best,
                        curves,
                    )
                save(
                    run.path / "last.pt",
                    model,
                    head,
                    optimizer,
                    config,
                    dataset_hash,
                    step,
                    order,
                    cursor,
                    rng,
                    best,
                    curves,
                )
        last_metrics = result["metrics"]
        state = restore(run.path / "best.pt", model, head)
        selected = evaluate(
            model,
            tokenizer,
            head,
            validation_rows,
            config,
            run,
            phase="validation_selected",
            step=state["step"],
        )
        # An explicit full train evaluation logs every source row at least once,
        # and separates memorization from validation generalization.
        train_result = evaluate(
            model,
            tokenizer,
            head,
            train_rows,
            config,
            run,
            phase="train_selected",
            step=state["step"],
        )
        integrity = None
        if backbone_before:
            backbone_after = module_fingerprint(model)
            head_after = module_fingerprint(head)
            assert backbone_before == backbone_after, "frozen backbone changed during training"
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            assert not state["adapted"], "frozen checkpoint contains adapted backbone parameters"
            assert head_before != head_after, "action head did not change"
            integrity = {
                "backbone_before_sha256": backbone_before,
                "backbone_after_sha256": backbone_after,
                "backbone_unchanged": True,
                "backbone_gradients_absent": True,
                "checkpoint_backbone_parameters": 0,
                "head_changed": True,
            }
        report = {
            "run_dir": str(run.path),
            "best_checkpoint": str(run.path / "best.pt"),
            "last_checkpoint": str(run.path / "last.pt"),
            "selected_step": state["step"],
            "start_step": start_step,
            "total_steps": config.steps,
            "initial": initial["metrics"],
            "last": last_metrics,
            "selected": selected["metrics"],
            "train_selected": train_result["metrics"],
            "wall_seconds": time.perf_counter() - started,
            "mean_update_ms": sum(timings) / len(timings),
            "unique_training_rows_updated": len(visited),
            "processed_training_examples": 0,
            "frozen_integrity": integrity,
            "test_evaluated": False,
        }
        # Read the authoritative per-message log rather than infer
        # partial-batch counts.
        report["processed_training_examples"] = sum(
            json.loads(line)["phase"] == "train_preupdate"
            for line in (run.path / "messages.jsonl").read_text(encoding="utf-8").splitlines()
        )
        run.save("validation_final.json", selected)
        run.save("train_final.json", train_result)
        run.save("report.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    del optimizer, parameters, backbone_parameters, head, model
    torch.cuda.empty_cache()
    return report


def test_checkpoint(checkpoint, rows):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    config = ActionConfig(**state["config"])
    model, tokenizer, head, _ = setup(config)
    restore(checkpoint, model, head)
    with Run(
        "action-heldout", {"checkpoint": str(checkpoint), "split": "test", "selection_locked": True}
    ) as run:
        result = evaluate(
            model, tokenizer, head, rows, config, run, phase="test", step=state["step"]
        )
        result["run_dir"] = str(run.path)
        run.save("test.json", result)
    del model, head
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def infer_checkpoint(checkpoint, commands):
    from .tasks import prompt

    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    config = ActionConfig(**state["config"])
    model, tokenizer, head, _ = setup(config)
    restore(checkpoint, model, head)
    model.eval()
    head.eval()
    rows = [
        {"command": command, "prompt": prompt(command), "plan": make_plan(command, [])}
        for command in commands
    ]
    results = []
    with Run(
        "action-inference",
        {"checkpoint": str(checkpoint), "source": "user probes, no known labels"},
    ) as run:
        for start in range(0, len(rows), config.batch_size):
            chunk = rows[start : start + config.batch_size]
            batch = encode_batch(tokenizer, chunk, "action", config.device, config.max_tokens)
            predictions = decode(forward(model, head, batch, config), chunk)
            records = [
                {"input": row["command"], "expected": None, "returned": prediction, "correct": None}
                for row, prediction in zip(chunk, predictions, strict=True)
            ]
            log_messages(
                run, records, "inference_unlabelled", 0, batch_index=start // config.batch_size
            )
            results.extend(records)
        run.save("predictions.json", results)
    del model, head
    torch.cuda.empty_cache()
    return results
