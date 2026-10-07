"""Build train-only expansion; consult sealed text only for hashed duplicate exclusion."""

import hashlib
import json
import sys
from pathlib import Path

from experiments.broad_action_data import build
from experiments.large_choice_data import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


parent = load(ROOT / "docs/smol_diverse_v2_train_validation.json")
assert set(parent["splits"]) == {"train", "validation"}


def digest(row):
    return hashlib.sha256(normalize(row.get("core", row["command"])).encode()).hexdigest()


blocked = {digest(row) for rows in parent["splits"].values() for row in rows}
old = load(ROOT / "docs/large_choice_dataset.json")
blocked.update(digest(row) for rows in old["splits"].values() for row in rows)
blocked.update(digest(row) for row in load(ROOT / "docs/rev_finetune_fresh_test.json")["rows"])
# This integrity-only pass gives the generator opaque hashes, not test labels,
# text examples, predictions or metrics. No model is involved.
sealed = load(ROOT / "data/evaluation/smol_diverse_v2_sealed_test.json")
seal_hash = sealed.pop("sha256")
assert (
    hashlib.sha256(json.dumps(sealed, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    == seal_hash
)
sealed_hashes = {digest(row) for row in sealed["rows"]}
blocked.update(sealed_hashes)
del sealed
data, audit = build(parent, blocked)
assert data["splits"]["validation"] == parent["splits"]["validation"]
new = [r for r in data["splits"]["train"] if r["data_kind"] == "broad_synthetic"]
assert not any(digest(r) in blocked for r in new)
assert len({r["id"] for r in data["splits"]["train"]}) == 15000
assert not any(digest(r) in sealed_hashes for r in data["splits"]["train"])
audit.update(
    sealed_test_sha256=seal_hash,
    sealed_access="Integrity-only normalized core hashing; no labels used or model predictions",
    heldout_or_old_core_collisions=0,
    test_scored=False,
    validation_unchanged=True,
    train_classification="15k engineered synthetic commands; not15k independent real utterances",
)
path = ROOT / "docs/smol_broad_v3_train_validation.json"
if path.exists():
    assert load(path) == data, "Do not silently mutate a training corpus"
else:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(ROOT / "docs/smol_broad_v3_dataset_audit.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(audit, ensure_ascii=False))
