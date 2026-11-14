"""Read-only replay checks, not new Isaac runs; protocol tests use owned CPU doubles."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from scripts import replay_dh116_native_parameter_r0 as replay
from tests.test_dh116_native_parameter_endpoints import endpoint
from tests.test_dh116_material_director_readback import case

def test_source_hash_and_symlink_are_not_silently_accepted(tmp_path):
    p=tmp_path/"source";p.write_bytes(b"owned")
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert replay.read_pinned(p,sha)==b"owned"
    with pytest.raises(ValueError,match="SHA"):replay.read_pinned(p,"0"*64)
    link=tmp_path/"alias";link.symlink_to(p)
    with pytest.raises(ValueError,match="regular"):replay.read_pinned(link,sha)

@pytest.mark.parametrize("fault",["clock","value","source","hardware","derived","nodes"])
def test_radius_requires_actual_matching_packet_and_raw_parameters(endpoint,fault):
    c=endpoint;rod=c.row()
    packet=c.owner.capture_reset(rod,material_reference_uid="material1",**c.args)
    if fault=="clock":packet["identity"]["physics_step"]+=1
    elif fault=="value":packet["reward_radius_sample"]["value"]*=2
    elif fault=="source":packet["reward_radius_sample"]["source"]["provenance"]="configured_nominal_radius"
    elif fault=="hardware":packet["reward_radius_sample"]["source"]["hardware_measured"]=True
    elif fault=="derived":packet["native_parameters"]["fixed_node_indices"]=[0]
    elif fault=="nodes":packet["node_positions_m"][0,0,0]+=.01
    with pytest.raises(ValueError):replay.check_radius_packet(packet,rod)

def test_same_clock_packet_uses_native_radius_without_making_hardware_claim(endpoint):
    c=endpoint;rod=c.row()
    packet=c.owner.capture_reset(rod,material_reference_uid="material1",**c.args)
    value=replay.check_radius_packet(packet,rod)
    assert value["value"]==pytest.approx(.005)
    assert value["source"]["hardware_measured"] is False

@pytest.mark.skipif(not replay.RECEIPT.exists(),reason="server-only read-only historical evidence")
def test_existing_native_16_samples_remain_blocked_without_geometry_and_stability_sources():
    result=replay.replay()
    assert result["new_physics_steps"]==result["optimizer_updates"]==0
    assert result["teacher_admission"] is False and result["native_qualification"] is False
    assert result["parameter_packets_validated"]==17 and result["measured_physical_rows"]==16
    assert result["raw_hand_rod_pair_samples"]==0 and result["raw_table_pair_samples"]==7872
    for mode,controls in result["controls"].items():
        assert len(controls)==2
        for c in controls:
            assert c["diagnostic_reward"] is c["training_reward"] is None
            assert c["terms"]["lift_progress"]["raw_sum"]==0
            assert c["terms"]["material_anchor_slip"]["raw_sum"] is None
            assert set(c["blocked_terms"])==({"approach_progress","hold_progress","instability"} |
                ({"material_anchor_slip"} if mode=="original" else set()))
    for row in result["per_step"]:
        slip=row["terms"]["material_anchor_slip"]
        assert slip["raw"] is None and not slip["valid"] and slip["applicable"] is False
