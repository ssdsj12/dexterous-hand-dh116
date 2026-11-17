"""Pure comparison contract, not native execution."""
import importlib.util
from copy import deepcopy
import pytest
MODULE="scripts.replay_dh116_measurement_origin"
def api():
    assert importlib.util.find_spec(MODULE) is not None,"same-ledger equivalence checker missing"
    import importlib as lib
    return lib.import_module(MODULE)

def test_only_explicit_source_fields_are_mapped():
    mod=api()
    row=dict(terms={"a":dict(raw=3.,valid=True,source="raw1|constant")},
        term_stats={"a":dict(mean=3.,source_identities=["raw1|constant","raw2|constant"])},
        body_net_lift_m=.03,training_reward=None)
    mapped=mod.reward_sources(row,{"raw1":"origin","raw2":"origin"})
    assert mapped["terms"]["a"]["source"]=="origin|constant"
    assert mapped["term_stats"]["a"]["source_identities"]==["origin|constant"]
    assert mapped["body_net_lift_m"]==.03 and mapped["training_reward"] is None
    assert row["terms"]["a"]["source"]=="raw1|constant"

@pytest.mark.parametrize("mutation",["reward","valid","clock","blocker","guard","value","pair"])
def test_equivalence_rejects_non_provenance_change(mutation):
    mod=api()
    row=dict(measurements={"a":{"value":1,"source":None}},pair_diagnostics={},
        r0={"terms":{"a":{"raw":2.,"valid":True,"source":None}},
            "term_stats":{},"physics_step":3,"blocked_terms":["slip"]},
        physical_guard_failure=None)
    changed=deepcopy(row)
    if mutation=="reward":changed["r0"]["terms"]["a"]["raw"]=0
    elif mutation=="valid":changed["r0"]["terms"]["a"]["valid"]=False
    elif mutation=="clock":changed["r0"]["physics_step"]=4
    elif mutation=="blocker":changed["r0"]["blocked_terms"]=[]
    elif mutation=="guard":changed["physical_guard_failure"]="joint_limit"
    elif mutation=="value":changed["measurements"]["a"]["value"]=2
    else:changed["pair_diagnostics"]["invented"]=True
    with pytest.raises(ValueError):mod.equal(row,changed,"test")
