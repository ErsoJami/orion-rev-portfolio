"""Small supervised comparisons with validation-only checkpoint selection."""

from __future__ import annotations

import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .heads import ActionHead, action_loss, lm_loss, predict_plans
from .runtime import MODEL_REVISION, Run, load_baseline, seed_all, sync
from .tasks import audit_suite, encode_batch, parse_json_prediction, plan_match


@dataclass
class TrainConfig:
    objective: str = "hybrid"  # action, hybrid, json
    steps: int = 200
    batch_size: int = 4
    learning_rate: float = 2e-4
    train_last_layers: int = 2
    lambda_action: float = 1.0
    lambda_text: float = 1.0
    max_tokens: int = 768
    max_new_tokens: int = 384
    log_every: int = 5
    eval_every: int = 25
    eval_rows: int = 12
    seed: int = 17
    device: str = "cuda"
    dtype: str = "bfloat16"


def select_trainable(model, head, last_layers: int, objective: str) -> list:
    if not 0 <= last_layers <= len(model.model.layers):
        raise ValueError("invalid train_last_layers")
    model.requires_grad_(False)
    if last_layers:
        for layer in model.model.layers[-last_layers:]:
            layer.requires_grad_(True)
    # Keep tied embeddings and LM projection unchanged in every objective.
    head.requires_grad_(objective != "json")
    return [p for p in (*model.parameters(), *head.parameters()) if p.requires_grad]


def forward_losses(model, head, batch, objective):
    hidden = model.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
    ).last_hidden_state
    pooled = hidden[
        torch.arange(hidden.shape[0], device=hidden.device), batch["prompt_lengths"] - 1
    ]
    if objective == "json":
        action, terms = hidden.sum() * 0, {}
    else:
        action, terms = action_loss(head(pooled), batch["action_labels"])
    text = lm_loss(model, hidden, batch["lm_labels"]) if objective != "action" else action * 0
    return action, text, terms


@torch.inference_mode()
def evaluate(
    model, tokenizer, head, rows, config: TrainConfig, *, generate_json: bool = False
) -> dict:
    exact, valid, tool_correct, tool_total = 0, 0, 0, 0
    losses_a, losses_t, predictions = [], [], []
    for row in rows:
        batch = encode_batch(tokenizer, [row], config.objective, config.device, config.max_tokens)
        action, text, _ = forward_losses(model, head, batch, config.objective)
        losses_a.append(float(action))
        if config.objective != "action":
            losses_t.append(float(text))
        prompt_batch = encode_batch(tokenizer, [row], "action", config.device, config.max_tokens)
        hidden = model.model(
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            use_cache=False,
        ).last_hidden_state[:, -1]
        if config.objective == "json":
            if not generate_json:
                continue
            generated = model.generate(
                prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            answer = tokenizer.decode(
                generated[0, prompt_batch["input_ids"].shape[1] :], skip_special_tokens=True
            )
            prediction = parse_json_prediction(answer, row)
        else:
            prediction = predict_plans(head(hidden), [row])[0]
            answer = None
        valid += prediction is not None
        exact += plan_match(prediction, row["plan"])
        expected_tools = [a["tool"] for a in row["plan"]["actions"]]
        predicted_tools = [a["tool"] for a in prediction["actions"]] if prediction else []
        tool_total += max(len(expected_tools), len(predicted_tools))
        tool_correct += sum(a == b for a, b in zip(expected_tools, predicted_tools))
        predictions.append(
            {
                "id": row["id"],
                "command": row["command"],
                "expected": row["plan"],
                "prediction": prediction,
                "generated_text": answer,
            }
        )
    result = {
        "rows": len(rows),
        "predictions": predictions,
        "generation_evaluated": config.objective != "json" or generate_json,
    }
    if config.objective != "json":
        result["loss_action"] = sum(losses_a) / len(rows)
    if losses_t:
        result["loss_lm"] = sum(losses_t) / len(losses_t)
    if result["generation_evaluated"]:
        result.update(
            plan_exact=exact / len(rows),
            plan_valid=valid / len(rows),
            tool_slot_accuracy=tool_correct / tool_total if tool_total else 0.0,
        )
    return result


def save_checkpoint(path, model, head, optimizer, step, config, suite_hash, sampler, best):
    payload = {
        "format": 1,
        "revision": MODEL_REVISION,
        "config": asdict(config),
        "suite_hash": suite_hash,
        "step": step,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler_rng": sampler.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "best": best,
    }
    temporary = Path(str(path) + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, head, optimizer=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["revision"] != MODEL_REVISION:
        raise ValueError("checkpoint backbone revision mismatch")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in payload["trainable"].items():
            parameters[name].copy_(value.to(parameters[name]))
    head.load_state_dict(payload["head"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def train_comparison(suite, config: TrainConfig, resume_from: Path | None = None) -> dict:
    if (
        config.objective not in ("action", "hybrid", "json")
        or config.steps < 1
        or config.batch_size < 1
    ):
        raise ValueError("invalid training configuration")
    seed_all(config.seed)
    model, tokenizer, metadata = load_baseline(device=config.device, dtype=config.dtype)
    head = ActionHead(model.config.hidden_size).to(config.device)
    parameters = select_trainable(model, head, config.train_last_layers, config.objective)
    if not parameters:
        raise ValueError("no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.01)
    sampler = random.Random(config.seed)
    dataset = audit_suite(suite)
    start_step, best = 0, float("inf")
    if resume_from:
        state = load_checkpoint(resume_from, model, head, optimizer)
        if state["suite_hash"] != dataset["sha256"]:
            raise ValueError("resume dataset mismatch")
        previous = state["config"]
        current = asdict(config)
        for key in current:
            if (
                key not in ("steps", "log_every", "eval_every", "eval_rows")
                and current[key] != previous[key]
            ):
                raise ValueError(f"resume config mismatch: {key}")
        start_step, best = state["step"], state["best"]
        sampler.setstate(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if config.steps <= start_step:
            raise ValueError("resume steps must exceed the saved step")
    # Cover the split, including executable commands even in a two-row smoke.
    source = suite["validation"]
    n = min(config.eval_rows, len(source))
    validation = [source[round(i * (len(source) - 1) / max(1, n - 1))] for i in range(n)]
    with Run(
        config.objective,
        {
            **asdict(config),
            **metadata,
            "dataset": dataset,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "action_head_parameters": sum(p.numel() for p in head.parameters()),
            "resume_from": str(resume_from) if resume_from else None,
        },
    ) as run:
        if resume_from:
            previous_best = Path(resume_from).parent / "best.pt"
            if previous_best.exists():
                shutil.copyfile(previous_best, run.path / "best.pt")
            else:
                best = float("inf")
        initial = evaluate(model, tokenizer, head, validation, config)
        run.save("validation_initial.json", initial)
        run.log({k: v for k, v in initial.items() if k != "predictions"}, start_step, "validation/")
        for step in range(start_step + 1, config.steps + 1):
            rows = sampler.sample(suite["train"], min(config.batch_size, len(suite["train"])))
            batch = encode_batch(
                tokenizer, rows, config.objective, config.device, config.max_tokens
            )
            optimizer.zero_grad(set_to_none=True)
            sync(config.device)
            started = time.perf_counter()
            action, text, terms = forward_losses(model, head, batch, config.objective)
            loss = (
                config.lambda_text * text
                if config.objective == "json"
                else config.lambda_action * action
            )
            if config.objective == "hybrid":
                loss = loss + config.lambda_text * text
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            warmup = min(1.0, step / min(20, max(1, config.steps // 10)))
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * warmup
            optimizer.step()
            sync(config.device)
            if step % config.log_every == 0 or step == 1 or step == config.steps:
                metrics = {
                    "loss/total": float(loss.detach()),
                    "loss/lm": float(text.detach()),
                    "loss/action": float(action.detach()),
                    "gradient_norm": float(norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "step_ms": (time.perf_counter() - started) * 1000,
                    "tokens": int(batch["attention_mask"].sum()),
                    **terms,
                }
                if config.device == "cuda":
                    metrics["allocated_mib"] = torch.cuda.memory_allocated() / 2**20
                run.log(metrics, step, "train/")
                print(
                    f"{config.objective}: {step}/{config.steps}, loss={metrics['loss/total']:.4f}",
                    flush=True,
                )
            if step % config.eval_every == 0 or step == config.steps:
                result = evaluate(model, tokenizer, head, validation, config)
                run.log(
                    {k: v for k, v in result.items() if k != "predictions"}, step, "validation/"
                )
                selection = (
                    result["loss_lm"] if config.objective == "json" else result["loss_action"]
                )
                if selection < best:
                    best = selection
                    save_checkpoint(
                        run.path / "best.pt",
                        model,
                        head,
                        optimizer,
                        step,
                        config,
                        dataset["sha256"],
                        sampler,
                        best,
                    )
                save_checkpoint(
                    run.path / "last.pt",
                    model,
                    head,
                    optimizer,
                    step,
                    config,
                    dataset["sha256"],
                    sampler,
                    best,
                )
        # Final report is validation. The held-out test is an explicit later
        # call.
        result = evaluate(model, tokenizer, head, validation, config, generate_json=True)
        result["run_dir"] = str(run.path)
        result["checkpoint"] = str(run.path / "last.pt")
        result["best_checkpoint"] = str(run.path / "best.pt")
        from .benchmarks import language_probe

        result["language_probe"] = language_probe(model, tokenizer)
        run.log({"token_ce": result["language_probe"]["token_ce"]}, config.steps, "language_final/")
        run.save("validation_final.json", result)
        run.log(
            {k: v for k, v in result.items() if k != "predictions"},
            config.steps,
            "validation_final/",
        )
    del optimizer, parameters, head, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_checkpoint(checkpoint, rows, *, max_new_tokens=None) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = TrainConfig(**state["config"])
    if max_new_tokens:
        config.max_new_tokens = max_new_tokens
    model, tokenizer, _ = load_baseline(device=config.device, dtype=config.dtype)
    head = ActionHead(model.config.hidden_size).to(config.device)
    load_checkpoint(checkpoint, model, head)
    with Run(
        "heldout-" + config.objective, {"checkpoint": str(checkpoint), "split": "test"}
    ) as run:
        result = evaluate(model, tokenizer, head, rows, config, generate_json=True)
        run.save("test.json", result)
        run.log({k: v for k, v in result.items() if k != "predictions"}, 0, "test/")
    return result
