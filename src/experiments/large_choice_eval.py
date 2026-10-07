"""Shared metrics for Smol training and the unchanged published Rev scorer."""

import math
from collections import defaultdict

from .action_training import log_messages, record_for
from .choice_action import decode_choices, signature


def make_records(logits, rows, orders, texts, catalog):
    predictions = decode_choices(logits, rows, orders, catalog=catalog)
    probabilities = logits.detach().float().softmax(-1).cpu().tolist()
    records = []
    for row, prediction, order, text, probs in zip(
        rows, predictions, orders, texts, probabilities, strict=True
    ):
        record = record_for(row, prediction)
        ranked = sorted(probs, reverse=True)
        record.update(
            model_input=text,
            candidate_order=order,
            probabilities=probs,
            confidence=ranked[0],
            margin=ranked[0] - ranked[1],
            target_signature=signature(row["decision"], row["plan"]["actions"]),
            family=row.get("family"),
            data_kind=row.get("data_kind"),
            scope_group=row.get("scope_group"),
            first_tool=row.get("first_tool"),
        )
        target = next(
            i
            for i, index in enumerate(order)
            if signature(catalog[index]["decision"], catalog[index]["actions"])
            == record["target_signature"]
        )
        record["cross_entropy"] = -math.log(max(probs[target], 1e-30))
        for field in ("semantic_category", "core", "derivation"):
            if field in row:
                record[field] = row[field]
        records.append(record)
    return records


def metrics(records):
    by_target, by_id, by_scope = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        by_target[r["target_signature"]].append(r)
        by_id[r["id"]].append(r)
        if r.get("scope_group"):
            by_scope[r["scope_group"]].append(r)

    def exact(group):
        return sum(r["correct"]["all"] for r in group) / len(group)

    empty = [r for r in records if not r["expected"]["plan"]["actions"]]
    scope = [r for r in records if r.get("scope_group")]
    decisions = {
        d: [r for r in records if r["expected"]["decision"] == d]
        for d in ("EXECUTE", "CLARIFY", "UNSUPPORTED", "NO_ACTION")
    }
    return {
        "exact": exact(records),
        "macro_candidate_exact": sum(exact(g) for g in by_target.values()) / len(by_target),
        "loss": sum(r["cross_entropy"] for r in records) / len(records),
        "rows": len(records),
        "unique_rows": len(by_id),
        "per_decision_exact": {d: exact(g) for d, g in decisions.items() if g},
        "per_candidate_exact": {d: exact(g) for d, g in by_target.items()},
        "false_action_rate": (
            sum(bool(r["returned"]["plan"]["actions"]) for r in empty) / len(empty)
            if empty
            else None
        ),
        "scope_exact": exact(scope) if scope else None,
        "scope_quartet_all_correct": (
            sum(all(r["correct"]["all"] for r in g) for g in by_scope.values()) / len(by_scope)
            if by_scope
            else None
        ),
        "order_consistency": (
            sum(
                len(
                    {
                        signature(r["returned"]["decision"], r["returned"]["plan"]["actions"])
                        for r in g
                    }
                )
                == 1
                for g in by_id.values()
            )
            / len(by_id)
            if all(len(g) >= 2 for g in by_id.values())
            else None
        ),
    }


def log_result(run, records, phase, step, batch_index):
    log_messages(run, records, phase, step, batch_index=batch_index)
