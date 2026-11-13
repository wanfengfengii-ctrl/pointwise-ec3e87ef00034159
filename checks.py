"""Regress supplied semantics; this does not solve the sealed batch."""
from fractions import Fraction
from itertools import combinations
import json
import source

def fixture():
    return {
        "priors": {"a": (1, 2), "b": (1, 2), "c": (1, 2)},
        "pools": [{"id": "p", "rev": 1, "deleted": False, "batches": ["a", "b"], "lo": 1, "hi": 1}],
        "tiles": [
            {"id": "x", "area": 2, "rule": ["all", "a", "b"]},
            {"id": "y", "area": 3, "rule": ["any", "b", "c"]}
        ],
        "jobs": [{"id": "r", "cost": 3, "repairs": ["a"]}, {"id": "s", "cost": 3, "repairs": ["b"]}],
        "inspections": [{"id": "i", "batch": "a", "cost": 1}],
        "policy": {"area_at_least": 5, "probability_at_least": "19/20"}
    }
def normal():
    model = fixture()
    worlds = [set(c) for n in range(4) for c in combinations(model["priors"], n)]
    worlds = [w for w in worlds if source.evidence_ok(model, w)]
    mass = sum((source.weight(model, w) for w in worlds), Fraction())
    assert mass == Fraction(1, 2)
    dist = {}
    for w in worlds:
        area = source.evaluate(model, w)["area"]
        dist[area] = dist.get(area, Fraction()) + source.weight(model, w) / mass
    assert dist == {0: Fraction(1, 4), 3: Fraction(3, 4)}
    assert source.evaluate(model, {"a", "c"}, ["r"])["area"] == 5
    assert source.evidence_ok(model, {"a", "c"})
    assert not source.evidence_ok(model, {"c"})
    v1 = {"id": "p", "rev": 1, "deleted": False, "batches": ["a"], "lo": 0, "hi": 0}
    v2 = dict(v1, rev=2, deleted=True)
    assert source.latest([v2, v1, v1]) == []
    try:
        source.latest([v1, dict(v1, hi=1)])
    except ValueError as exc:
        assert str(exc) == "REVISION_CONFLICT"
    else:
        raise AssertionError("conflict accepted")
    real = source.load()
    assert len(real["priors"]) == 53
    assert len(real["pools"]) == 16
    assert len(real["tiles"]) == 24
    assert sum(t["area"] for t in real["tiles"]) == 60
    assert len(real["jobs"]) == 6
    assert [x["batch"] for x in real["inspections"]] == ["A0", "A1", "A2"]
    assert source.evaluate(real, set(real["priors"]), [j["id"] for j in real["jobs"]])["area"] == 60
    print(json.dumps({"fixture_mass": str(mass), "fixture_area_pmf": {str(k): str(v) for k, v in sorted(dist.items())}}, sort_keys=True))
    print("NORMAL_REGRESSION_OK")

if __name__ == "__main__":
    normal()
