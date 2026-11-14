"""CPU-only provenance tests; synthetic packet fixtures confer no native credit."""
from copy import deepcopy
import importlib.util
import pytest
from test_live_effective_pad_source import ROOT,recorded,source,synthetic_complete_fixture
from go2_pvcnn.control.dh116.tabletop_live_effective_pad_source import CompletePairLiveR0Observer
MODULE="go2_pvcnn.control.dh116.tabletop_r0_measurement_origin"

def api():
    assert importlib.util.find_spec(MODULE) is not None, "stable measurement origin adapter missing"
    return importlib.import_module(MODULE)

def sample(recorded,source):
    x=synthetic_complete_fixture(recorded[1])
    observer=object.__new__(CompletePairLiveR0Observer)
    observer._grasp="pinch";observer.effective_pad_source=source
    c=x["action_readback"]["post"]["identity"][0]
    m,d,na=observer._measurements({"exchange":x},c,None)
    return x,m,d,na

def test_origins_stable_samples_preserved_and_values_untouched(recorded,source):
    module=api();x,m,d,_=sample(recorded,source)
    adapter=module.MeasurementOrigin(recorded[0])
    baseline=deepcopy((x,m,d))
    out,p=adapter.adapt(x,m,d)
    assert out["sample_identity"]==m["sample_identity"]
    for name in module.FIELDS:
        assert out[name]["value"]==m[name]["value"]
        assert p["samples"][name]["original_source"]==m[name]["source"]
        assert out[name]["source"]==p["samples"][name]["origin"]
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    assert _digest((x,m,d))==_digest(baseline)
    # Same validated sample replay has identical IDs; origin never accumulates samples.
    again,p2=adapter.adapt(x,m,d)
    assert again==out and p2==p
    assert len(adapter.origins)==4
    assert len(set(adapter.origins.values()))==4
    assert p["identity"]==x["action_readback"]["post"]["identity"][0]
    assert p["raw_capture_sha256"]==d["complete_effective_pad_source"]["raw_capture_sha256"]

@pytest.mark.parametrize("mutation",["episode","reset","rod","bindings","producer","source_family","side_hash","table_hash","joint_hash"])
def test_changed_origin_or_sample_binding_is_rejected(recorded,source,mutation):
    module=api();x,m,d,_=sample(recorded,source)
    adapter=module.MeasurementOrigin(recorded[0]);adapter.adapt(x,m,d)
    if mutation in ("episode","reset","rod"):
        key={"episode":"episode_uid","reset":"reset_id","rod":"rod_uid"}[mutation]
        x["action_readback"]["post"]["identity"][0][key]="wrong"
    elif mutation=="bindings":x["rod"]["mesh_bindings"]["right.finger14_link"]["side"]="left"
    elif mutation=="producer":d["complete_effective_pad_source"]["producer_source_sha256"]="0"*64
    elif mutation=="source_family":m["joint_limit"]["source"]="guessed_force:123"
    elif mutation=="side_hash":m["left"]["source"]=m["right"]["source"]
    else:m["table_contact" if mutation=="table_hash" else "joint_limit"]["source"]+="tampered"
    with pytest.raises(ValueError):adapter.adapt(x,m,d)

def test_missing_value_stays_unknown_without_source(recorded,source):
    module=api();x,m,d,_=sample(recorded,source)
    m["left"]={"value":None,"source":None}
    d["complete_effective_pad_source"]["hands"]["left"]["value"]=None
    out,p=module.MeasurementOrigin(recorded[0]).adapt(x,m,d)
    assert out["left"]==m["left"] and p["samples"]["left"]["origin"] is None
    assert out["stable"]==m["stable"] and out["stable"]["value"] is None

def test_new_reset_cannot_reuse_old_epoch_origin(recorded,source):
    module=api();x,m,d,_=sample(recorded,source)
    reset=deepcopy(recorded[0]);reset["episode_uid"]="another"
    with pytest.raises(ValueError):module.MeasurementOrigin(reset).adapt(x,m,d)

def test_provenance_is_reversible_and_tampering_rejected(recorded,source):
    module=api();x,m,d,_=sample(recorded,source)
    adapter=module.MeasurementOrigin(recorded[0]);out,p=adapter.adapt(x,m,d)
    assert module.restore_measurements(out,p)==m
    for key in ("origin","original_source"):
        bad=deepcopy(p);bad["samples"]["right"][key]="tampered"
        with pytest.raises(ValueError):module.restore_measurements(out,bad)

def test_per_frame_raw_digest_is_not_origin_identity(recorded,source):
    module=api();x,m,d,_=sample(recorded,source)
    a=module.MeasurementOrigin(recorded[0]);out,p=a.adapt(x,m,d)
    # Different raw fixture and geometry packet, same physical producer.
    x2=deepcopy(x)
    x2["rod"]["mesh_wrenches_com_world"]={"test_owned_evidence":1}
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
    m2=deepcopy(m);m2["table_contact"]["source"]="validated_solver_pair_capture:"+_digest(x2["rod"])+":any_table_SDF_pair_not_PhysX_hand_table"
    out2,p2=a.adapt(x2,m2,d)
    assert out2["table_contact"]["source"]==out["table_contact"]["source"]
    assert p2["samples"]["table_contact"]["original_source"]!=p["samples"]["table_contact"]["original_source"]
