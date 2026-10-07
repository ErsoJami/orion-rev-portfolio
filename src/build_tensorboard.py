"""Build a compact TensorBoard dashboard from audited aggregate artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

PACKAGE = Path(__file__).resolve().parents[1]
RESULTS = PACKAGE / "results"
LOGDIR = RESULTS / "tensorboard"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_scalar(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def log_training(writer: SummaryWriter) -> int:
    lines = (RESULTS / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    scalar_count = 0
    train_tags = {
        "loss/total_mean",
        "loss/ce_mean",
        "loss/consistency_mean",
        "loss/expected_cost_mean",
        "gradient_norm",
        "lr_factor",
    }
    validation_tags = {
        "exact",
        "macro_candidate_exact",
        "loss",
        "false_action_rate",
        "error_cost",
        "brier",
        "ece15",
        "scope_exact",
        "scope_quartet_all_correct",
    }
    for line in lines:
        item = json.loads(line)
        prefix = item["prefix"].strip("/")
        step = int(item["step"])
        allowed = train_tags if prefix == "train_window" else validation_tags
        if (
            prefix == "train_window"
            or prefix.startswith("validation")
            or prefix.startswith("train_probe")
            or prefix == "train_selected"
        ):
            for key in allowed:
                value = item.get(key)
                if finite_scalar(value):
                    writer.add_scalar(f"training/{prefix}/{key}", float(value), step)
                    scalar_count += 1
    report = load_json(RESULTS / "full_train_metrics.json")
    train = report["train"]
    for key in ("exact", "false_action_rate", "error_cost", "macro_candidate_exact", "loss"):
        writer.add_scalar(f"training/full_train_last/{key}", float(train[key]), int(report["step"]))
        scalar_count += 1
    return scalar_count


def log_dataset(writer: SummaryWriter) -> None:
    audit = load_json(RESULTS / "dataset_audit.json")
    writer.add_scalar("dataset/rows/train", audit["train_rows"], 0)
    writer.add_scalar("dataset/rows/validation", audit["validation_rows"], 0)
    for category, count in audit["added_categories"].items():
        writer.add_scalar(f"dataset/new_train_rows/category/{category}", count, 0)
    for decision, count in audit["whole_train_decisions"].items():
        writer.add_scalar(f"dataset/train_rows/decision/{decision}", count, 0)
    writer.add_text(
        "dataset/provenance",
        """### Synthetic corpus v3

15,000 train rows = 8,000 retained from diverse v2 + 7,000 new unique normalized core phrases. The 1,000-row validation split is byte-for-byte preserved at the row level. Labels are rule-derived from a fixed 26-plan catalogue; no independent human adjudication or real-user collection is claimed.

New rows use fixed quotas across direct requests, question-wrapped requests, argument corrections, negation scope, cancellation, explanation, quotation, past report, hypothetical requests, missing arguments and unavailable tools. The audit used only normalized SHA-256 core hashes to exclude collisions with prior/held-out examples. It recorded zero collisions; sealed test text and labels were not scored by the model.

See `DATASET.md` and `results/dataset_audit.json` for the exact counts, derivation and limits.""",
        0,
    )


def log_comparison(writer: SummaryWriter) -> None:
    result = load_json(RESULTS / "matched_comparison.json")
    pretty = {"orion": "SmolLM2-135M + Rev head", "original_rev": "Published Qwen3.5-4B Rev"}
    for key, model in result["models"].items():
        name = pretty[key]
        metrics = model["metrics"]
        latency = model["latency_batch1"]
        for metric in ("exact", "false_action_rate", "macro_candidate_exact"):
            writer.add_scalar(f"comparison/{metric}/{name}", float(metrics[metric]), 0)
        for percentile in ("p50", "p95"):
            value = float(latency[f"{percentile}_ms"])
            writer.add_scalar(f"comparison/batch1_latency_ms/{percentile}/{name}", value, 0)
            writer.add_scalar(
                f"comparison/serial_queries_per_second_from_{percentile}/{name}", 1000 / value, 0
            )
        writer.add_scalar(
            f"comparison/peak_allocated_memory_mib/{name}",
            float(latency["peak_allocated_mib"]),
            0,
        )
        writer.add_histogram(
            f"comparison/batch1_latency_samples_ms/{name}",
            torch.tensor(latency["samples_ms"], dtype=torch.float32),
            0,
        )
    writer.add_text(
        "comparison/protocol",
        """### Matched development comparison

Both systems scored the same 1,000 commands, candidate descriptions and option order (seed 29001). Latency: batch 1, 5 warmups, 30 synchronized repeats, after tokenization; includes tensor packing and model forward, excludes load/tokenization/HTTP/application overhead.

The Original Rev weights are the published Qwen3.5-4B adapter, with no Orion-command fine-tuning. Our weights are the final step-1875 Smol checkpoint. Model size, tokenizer, pretraining and adaptation history differ, so this is an end-system comparison, not an isolated architecture experiment. The synthetic development set is not an independent test or production accuracy.

Serial queries/second tags are the reciprocal of measured p50/p95 latency, not a separate saturated-throughput benchmark. Hardware: NVIDIA GeForce RTX 5060 Ti.""",
        0,
    )
    writer.add_text(
        "comparison/paired_correctness",
        json.dumps(result["paired_correctness"], ensure_ascii=False, indent=2),
        0,
    )


def main() -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(LOGDIR / "smol_training") as writer:
        count = log_training(writer)
        writer.add_text(
            "run/about",
            "Audited full-pass scalar history reconstructed from `results/training_metrics.jsonl`; no raw example messages copied.",
            0,
        )
        writer.flush()
    with SummaryWriter(LOGDIR / "dataset_profile") as writer:
        log_dataset(writer)
    with SummaryWriter(LOGDIR / "matched_rev_comparison") as writer:
        log_comparison(writer)
    summary = {
        "logdir": str(LOGDIR.relative_to(PACKAGE)),
        "runs": ["smol_training", "dataset_profile", "matched_rev_comparison"],
        "training_scalar_points": count,
        "source": "aggregate JSON/JSONL artifacts only; no command texts or TensorBoard message logs",
    }
    (RESULTS / "tensorboard_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
