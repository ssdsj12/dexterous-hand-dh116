"""CPU ownership and serialized provenance audit; no native qualification."""
from copy import deepcopy
import numpy as np
import pytest
from go2_pvcnn.control.dh116 import tabletop_flexible_parameter_readback as reader
from test_flexible_parameter_readback import case,capture

def validate(c,packet):
    return reader.validate_packet(packet,identity=c.expected,constructor_nodes=c.base.nodes)

def test_owned_packet_roundtrip_and_separate_provenance(case):
    packet=capture(case)
    out=validate(case,packet)
    assert out["field_count"]==len(packet["fields"])
    assert out["parameter_readback_valid"] is True
    assert out["teacher_admission"] is out["material_identified"] is False
    assert out["configured_radius_m"]==.005
    assert out["mass_kg"]==pytest.approx(.0325,rel=2e-5)

@pytest.mark.parametrize("damage",[
    "identity","sources","nodes","rest","mass","pin","configured","getter",
    "native_radius","force_material","calibration","missing_unknown","unknown_zero",
    "global_friction","pair_friction","unit","new_steps","extra","native_qualification",
    "reward_sample","getter_bool","mass_summary"])
def test_tampered_packet_rejected(case,damage):
    packet=deepcopy(capture(case))
    if damage=="identity":packet["identity"]["physics_step"]+=1
    elif damage=="sources":packet["sources"]["backend_sha256"]="0"*64
    elif damage=="nodes":packet["constructor_nodes_m"][0][0]+=.001
    elif damage=="rest":packet["constructor_rest_lengths_m"][0]+=.001
    elif damage=="mass":packet["node_inverse_masses_kg_inv"][0][0]*=2
    elif damage=="pin":packet["node_inverse_masses_kg_inv"][0][0]=0
    elif damage=="configured":packet["configured"]["rod"]["radius_m"]*=2
    elif damage=="getter":packet["current_native_parameters"]["damping"]=100.
    elif damage=="native_radius":packet["fields"]["radius_m"]["native_measured"]=True
    elif damage=="force_material":packet["material_identified"]=True
    elif damage=="calibration":packet["fields"]["radius_m"]["uncertainty_calibrated"]=True
    elif damage=="missing_unknown":del packet["fields"]["young_modulus_pa"]
    elif damage=="unknown_zero":packet["fields"]["young_modulus_pa"]["value"]=0
    elif damage=="global_friction":packet["fields"]["global_friction_static"]["value"]=1.
    elif damage=="pair_friction":packet["configured_contact_pairs"]["left.finger14_link"]["mu_s"]=.5
    elif damage=="unit":packet["fields"]["radius_m"]["unit"]="mm"
    elif damage=="new_steps":packet["new_solver_steps"]=1
    elif damage=="extra":packet["unregistered_field"]=0
    elif damage=="native_qualification":packet["native_qualification"]=True
    elif damage=="reward_sample":packet["reward_radius_sample"]=.005
    elif damage=="getter_bool":packet["current_native_parameters"]["num_itr"]=True
    elif damage=="mass_summary":packet["mass_readback"]["mass_kg"]=100.
    with pytest.raises((ValueError,TypeError,KeyError)):
        validate(case,packet)
