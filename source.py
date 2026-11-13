"""Read sealed materials; evaluate a specified world, without inference."""
import csv
import json
from pathlib import Path
from fractions import Fraction

ROOT = Path(__file__).resolve().parent

def names(rule):
    if isinstance(rule, str):
        return {rule}
    if not isinstance(rule, list) or not rule:
        raise ValueError("BAD_RULE")
    if rule[0] in ("all", "any") and len(rule) > 1:
        children = rule[1:]
    elif rule[0] == "atleast" and len(rule) >= 3 and type(rule[1]) is int and 1 <= rule[1] <= len(rule)-2:
        children = rule[2:]
    else:
        raise ValueError("BAD_RULE")
    return set().union(*(names(x) for x in children))

def good(rule, faults, repaired=()):
    if isinstance(rule, str):
        return rule in repaired or rule not in faults
    if rule[0] == "all":
        return all(good(x, faults, repaired) for x in rule[1:])
    if rule[0] == "any":
        return any(good(x, faults, repaired) for x in rule[1:])
    return sum(good(x, faults, repaired) for x in rule[2:]) >= rule[1]

def latest(records):
    seen, heads = {}, {}
    for row in records:
        key = (row["id"], row["rev"])
        if type(row["rev"]) is not int or row["rev"] < 1 or type(row["deleted"]) is not bool:
            raise ValueError("BAD_REVISION")
        if key in seen and seen[key] != row:
            raise ValueError("REVISION_CONFLICT")
        seen[key] = row
        if row["id"] not in heads or row["rev"] > heads[row["id"]]["rev"]:
            heads[row["id"]] = row
    return [heads[k] for k in sorted(heads) if not heads[k]["deleted"]]

def load(root=ROOT):
    root = Path(root)
    with (root / "data/batches.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    priors = {}
    for row in rows:
        name, n, d = row["id"], int(row["bad_n"]), int(row["den"])
        if name in priors or not 0 < n < d:
            raise ValueError("BAD_PRIOR")
        priors[name] = (n, d)
    records = [json.loads(line) for line in (root / "data/pools.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in records:
        ids = row["batches"]
        if not row["id"] or len(set(ids)) != len(ids) or not set(ids) <= priors.keys():
            raise ValueError("BAD_POOL")
        if type(row["lo"]) is not int or type(row["hi"]) is not int or not 0 <= row["lo"] <= row["hi"] <= len(ids):
            raise ValueError("BAD_POOL")
    pools = latest(records)
    mosaic = json.loads((root / "data/mosaic.json").read_text(encoding="utf-8"))
    tiles = mosaic["tiles"]
    if len({t["id"] for t in tiles}) != len(tiles):
        raise ValueError("DUP_TILE")
    for tile in tiles:
        if type(tile["area"]) is not int or tile["area"] <= 0 or not names(tile["rule"]) <= priors.keys():
            raise ValueError("BAD_TILE")
    rework = json.loads((root / "data/rework.json").read_text(encoding="utf-8"))
    jobs = rework["jobs"]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("DUP_JOB")
    for job in jobs:
        if type(job["cost"]) is not int or job["cost"] <= 0 or not set(job["repairs"]) <= priors.keys():
            raise ValueError("BAD_JOB")
    inspections = rework["inspections"]
    if len({x["id"] for x in inspections}) != len(inspections) or len({x["batch"] for x in inspections}) != len(inspections):
        raise ValueError("DUP_INSPECTION")
    for item in inspections:
        if item["batch"] not in priors or type(item["cost"]) is not int or item["cost"] <= 0:
            raise ValueError("BAD_INSPECTION")
    return {"priors": priors, "pools": pools, "tiles": tiles, "jobs": jobs, "inspections": inspections, "policy": mosaic["policy"]}

def evidence_ok(model, faults):
    return all(p["lo"] <= len(set(p["batches"]) & set(faults)) <= p["hi"] for p in model["pools"])

def weight(model, faults):
    out = Fraction(1)
    for name, (n, d) in model["priors"].items():
        out *= Fraction(n if name in faults else d-n, d)
    return out

def evaluate(model, faults, selected=()):
    """Replacements affect output quality, not the original world's likelihood."""
    if set(selected) - {j["id"] for j in model["jobs"]} or not set(faults) <= model["priors"].keys():
        raise ValueError("UNKNOWN_ID")
    repaired = {name for j in model["jobs"] if j["id"] in selected for name in j["repairs"]}
    usable = [t["id"] for t in model["tiles"] if good(t["rule"], faults, repaired)]
    return {"usable_tiles": usable, "area": sum(t["area"] for t in model["tiles"] if t["id"] in usable)}

def inspect():
    model = load()
    print(json.dumps({
        "batches": len(model["priors"]),
        "active_pools": [p["id"] for p in model["pools"]],
        "tiles": len(model["tiles"]),
        "total_area": sum(t["area"] for t in model["tiles"]),
        "jobs": model["jobs"],
        "inspections": model["inspections"],
        "policy": model["policy"],
        "all_good_world": evaluate(model, set()),
        "all_bad_world": evaluate(model, set(model["priors"]))
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    inspect()
