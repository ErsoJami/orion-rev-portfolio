"""Decision quality and uncertainty; repeated option orders are not new samples."""

from collections import defaultdict
from math import sqrt

from .choice_action import signature


def wilson(successes, count, z=1.96):
    if count == 0:
        return None
    p = successes / count
    den = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / den
    width = z * sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
    return [max(0.0, centre - width), min(1.0, centre + width)]


def quality(records, catalog):
    """Supplement existing exact metrics with explicit error cost and calibration."""
    first = {}
    brier = []
    bins = [[] for _ in range(15)]
    false_actions = 0
    wrong = 0
    argument_errors = 0
    categories = defaultdict(list)
    for r in records:
        first.setdefault(r["id"], r)
        if r.get("semantic_category"):
            categories[r["semantic_category"]].append(r)
        correct = r["correct"]["all"]
        wrong += not correct
        false = not r["expected"]["plan"]["actions"] and bool(r["returned"]["plan"]["actions"])
        false_actions += false
        argument_errors += r["correct"]["tools"] and not r["correct"]["arguments"]
        target = next(
            i
            for i, j in enumerate(r["candidate_order"])
            if signature(catalog[j]["decision"], catalog[j]["actions"]) == r["target_signature"]
        )
        probs = r["probabilities"]
        brier.append(1 + sum(p * p for p in probs) - 2 * probs[target])
        bins[min(14, int(r["confidence"] * 15))].append((r["confidence"], correct))
    unique = list(first.values())
    empty = [r for r in unique if not r["expected"]["plan"]["actions"]]
    count_false = sum(bool(r["returned"]["plan"]["actions"]) for r in empty)
    count_correct = sum(r["correct"]["all"] for r in unique)
    return {
        "error_cost": (wrong + 4 * false_actions) / len(records),
        "cost_definition": "wrong decision costs 1; false EXECUTE for empty gold costs 5 total",
        "brier": sum(brier) / len(brier),
        "ece15": sum(
            len(b)
            / len(records)
            * abs(sum(c for c, _ in b) / len(b) - sum(y for _, y in b) / len(b))
            for b in bins
            if b
        ),
        "argument_error_views": argument_errors,
        "per_category_exact": {
            c: sum(r["correct"]["all"] for r in group) / len(group)
            for c, group in categories.items()
        },
        "first_order_unique_rows": len(unique),
        "first_order_correct": count_correct,
        "first_order_exact_ci95": wilson(count_correct, len(unique)),
        "first_order_empty_rows": len(empty),
        "first_order_false_actions": count_false,
        "first_order_false_action_ci95": wilson(count_false, len(empty)),
        "interval_scope": "first permutation per unique command; correlated grammar siblings still limit population interpretation",
    }
