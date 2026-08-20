"""Runner orchestration tests (job enumeration and config resolution; no parallel run)."""
from __future__ import annotations

from src.run import runner


def test_index_covers_all_67_instances():
    assert len(runner.INDEX) == 67
    assert runner.INDEX["ft06"][0] == "jssp"
    assert runner.INDEX["mk01"][0] == "fjsp"
    assert runner.INDEX["edata_la01"][0] == "fjsp"


def test_all_methods_and_determinism_partition():
    assert len(runner.ALL_METHODS) == 17
    assert runner.DETERMINISTIC <= set(runner.ALL_METHODS)
    # only the two constructive methods are deterministic; the exact solver takes a seed
    assert runner.DETERMINISTIC == {"dispatching", "greedy"}
    stochastic = set(runner.ALL_METHODS) - runner.DETERMINISTIC
    assert {"cpsat", "tabu", "sa", "brkga", "cmaes", "mde", "csa", "ga", "rime",
            "imode"} <= stochastic
    # every decoded method is in the panel and none of the non-decoded ones leaked in
    assert runner.DECODED <= set(runner.ALL_METHODS)
    assert not (runner.DECODED & {"cpsat", "tabu", "sa", "dispatching", "greedy"})


def test_method_ids_handles_flat_dict_and_alias():
    assert runner._method_ids({"methods": ["cpsat", "tabu", "ga"]}, None) == ["cpsat", "tabu", "ga"]
    # detailed panel (list of dicts): greedy_gt and recent_extra are aliased
    cfg = {"methods": [{"id": "cpsat"}, {"id": "greedy_gt"}, {"id": "recent_extra"},
                       {"id": "cmaes"}, {"id": "mde"}]}
    assert runner._method_ids(cfg, None) == ["cpsat", "greedy", "rime", "cmaes", "mde"]
    # explicit override wins
    assert runner._method_ids(cfg, ["de", "pso"]) == ["de", "pso"]


def test_instance_ids_list_and_all():
    assert runner._instance_ids({"instances": ["ft06", "mk01", "nope"]}, None) == ["ft06", "mk01"]
    assert len(runner._instance_ids({"instances": "all"}, None)) == 67
    assert runner._instance_ids({}, ["abz5", "mk06"]) == ["abz5", "mk06"]


def test_result_path_format():
    p = runner.result_path("ft06", "tabu", 11, "pilot")
    assert p.name == "ft06__tabu__seed011.json"
    assert p.parent.name == "pilot"  # results are namespaced per run
