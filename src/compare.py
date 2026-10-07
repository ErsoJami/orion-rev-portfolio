"""Matched development and latency comparison for the portfolio checkpoints."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

from experiments.choice_action import shuffled_orders
from experiments.large_choice_eval import make_records, metrics
from experiments.runtime import measure
from scripts.finetune_original_rev import encode as encode_smol
from scripts.train_smol_full_rev import forward as smol_forward
from scripts.train_smol_full_rev import load_model as load_smol
from scripts.train_smol_full_rev import restore as restore_smol

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_SRC = Path(__file__).resolve().parents[0]
sys.path[:0] = [str(PACKAGE_SRC), str(ROOT)]


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train_validation.json"
CHECKPOINT = Path(__file__).resolve().parents[1] / "weights" / "last.pt"
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "matched_comparison.json"


def load_original_rev():
    spec = importlib.util.spec_from_file_location(
        "portfolio_original_rev", PACKAGE_SRC / "scripts" / "evaluate_original_rev.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.load_original()


def make_orders(rows: list[dict], seed: int) -> list[list[int]]:
    orders: list[list[int]] = []
    for start in range(0, len(rows), 12):
        orders.extend(shuffled_orders(rows[start : start + 12], seed + start))
    return orders


def evaluate_smol(rows: list[dict], corpus: dict, orders: list[list[int]]):
    scorer, model_info = load_smol()
    restore_smol(CHECKPOINT, scorer)
    scorer.backbone.eval()
    scorer.headq.eval()
    scorer.headk.eval()
    records: list[dict] = []

    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            batch = rows[start : start + 8]
            batch_orders = orders[start : start + len(batch)]
            encoded, texts, _ = encode_smol(scorer, batch, batch_orders, corpus)
            logits = smol_forward(scorer, encoded)
            records.extend(make_records(logits, batch, batch_orders, texts, corpus["candidates"]))

    sample, _, _ = encode_smol(scorer, rows[:1], orders[:1], corpus)
    latency = measure(lambda: smol_forward(scorer, sample), "cuda", warmup=5, repeats=30)
    info = {
        "name": "Orion SmolLM2-135M + trained Rev pointer head",
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "parameters": sum(p.numel() for p in scorer.backbone.parameters())
        + sum(p.numel() for p in scorer.headq.parameters())
        + sum(p.numel() for p in scorer.headk.parameters()),
        "base_model": model_info["model_id"],
        "base_revision": model_info["revision"],
        "latency_batch1": latency,
    }
    del scorer
    gc.collect()
    torch.cuda.empty_cache()
    return info, records


def encode_original(scorer, rows: list[dict], orders: list[list[int]], corpus: dict):
    encoded: list[tuple[list[int], list[int]]] = []
    texts: list[str] = []
    for row, order in zip(rows, orders, strict=True):
        criteria = {
            str(index + 1): corpus["candidates"][candidate]["description"]
            for index, candidate in enumerate(order)
        }
        ids, positions, _ = scorer.encode(
            row["command"], {"instructions": corpus["instructions"], "criteria": criteria}
        )
        if len(ids) > 16384:
            raise ValueError("Original Rev token limit exceeded")
        encoded.append((ids, positions))
        texts.append(
            "State:\n"
            + row["command"]
            + "\nQuestion: "
            + corpus["instructions"]
            + "\nOptions:\n"
            + "".join(f"{key}: {description}\n" for key, description in criteria.items())
            + "Decision:"
        )
    return encoded, texts


def evaluate_original(rows: list[dict], corpus: dict, orders: list[list[int]]):
    scorer, model_info = load_original_rev()
    records: list[dict] = []

    with torch.inference_mode():
        for start in range(0, len(rows), 4):
            batch = rows[start : start + 4]
            batch_orders = orders[start : start + len(batch)]
            encoded, texts = encode_original(scorer, batch, batch_orders, corpus)
            probabilities, _ = scorer.run_rows(encoded)
            logits = torch.tensor(probabilities, dtype=torch.float32).clamp_min(1e-30).log()
            records.extend(make_records(logits, batch, batch_orders, texts, corpus["candidates"]))

    sample, _ = encode_original(scorer, rows[:1], orders[:1], corpus)
    latency = measure(lambda: scorer.run_rows(sample), "cuda", warmup=5, repeats=30)
    model_info = {
        "name": "Published Original Rev on Qwen3.5-4B",
        "checkpoint_sha256": model_info["checkpoint_sha256"],
        "base_revision": model_info["base_revision"],
        "parameters": model_info["backbone_parameters"] + model_info["head_parameters"],
        "latency_batch1": latency,
    }
    del scorer
    gc.collect()
    torch.cuda.empty_cache()
    return model_info, records


def pairwise(ours: list[dict], original: list[dict]) -> dict:
    outcomes = {"both_correct": 0, "ours_only": 0, "original_only": 0, "both_wrong": 0}
    for mine, upstream in zip(ours, original, strict=True):
        mine_ok = mine["correct"]["all"]
        original_ok = upstream["correct"]["all"]
        if mine_ok and original_ok:
            outcomes["both_correct"] += 1
        elif mine_ok:
            outcomes["ours_only"] += 1
        elif original_ok:
            outcomes["original_only"] += 1
        else:
            outcomes["both_wrong"] += 1
    return outcomes


def main() -> None:
    corpus = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    digest = corpus["sha256"]
    rows = corpus["splits"]["validation"]
    orders = make_orders(rows, seed=29001)
    smol_info, smol_records = evaluate_smol(rows, corpus, orders)
    original_info, original_records = evaluate_original(rows, corpus, orders)

    result = {
        "experiment": "matched development comparison; no training and no sealed test access",
        "dataset_sha256": digest,
        "rows": len(rows),
        "candidate_order_seed": 29001,
        "hardware": torch.cuda.get_device_name(0),
        "models": {
            "orion": {**smol_info, "metrics": metrics(smol_records)},
            "original_rev": {**original_info, "metrics": metrics(original_records)},
        },
        "paired_correctness": pairwise(smol_records, original_records),
        "limitations": [
            "Both models receive the same commands, policy, candidates and option order; tokenizers and native model input processing differ.",
            "The models have different sizes and pretraining/adaptation histories, so this is a system comparison rather than a causal architecture ablation.",
            "The 1000-row synthetic development split is not production accuracy or an independent sealed test.",
            "Latency is batch-1 model forward only after tokenization; it excludes load, tokenization, HTTP and application overhead.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
