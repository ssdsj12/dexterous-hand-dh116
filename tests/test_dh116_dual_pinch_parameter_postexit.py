"""Disk tampering and composition checks; synthetic CPU ledger, not native."""
from copy import deepcopy
import json
import numpy as np
import pytest
from test_dh116_dual_pinch_director_postexit import evidence,case,write,journal
from scripts import validate_dh116_dual_pinch_parameter_interval as m
from go2_pvcnn.control.dh116 import tabletop_flexible_parameter_readback as reader
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import lumped_masses,verify_mass_readback
from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import _clock

@pytest.fixture
def packets(evidence):
    e=evidence
    nodes=e.reset["payload"]["node_positions_m"][0]
    rest=np.linalg.norm(np.diff(nodes,axis=0),axis=1)
    cfg=json.loads((reader.ROOT/reader.CONFIG).read_text())
    inv=(1/lumped_masses(nodes,cfg["rod"]["linear_density_kg_m"])).astype(np.float32)[None]
    mass=verify_mass_readback(inv,lumped_masses(nodes,cfg["rod"]["linear_density_kg_m"]))
    parameters={k:(cfg["rod"]["solver_iterations"] if k=="num_itr" else float(cfg["rod"][k])) for k in reader.GETTERS}
    def packet(identity):
        return reader._owned_packet(identity,reader.source_basis(),cfg,nodes,rest,inv,mass,parameters,41)
    e.reset["payload"]["parameter_readback"]=packet(e.reset["identity"])
    for row in e.rows[2:]:
        x=row["payload"]["exchange"]
        x["material_director"]["parameter_readback"]=packet(_clock(lambda:x["rod"]))
    write(e.out/"director_reset.json",e.reset);journal(e.out/"physical.jsonl",e.rows)
    return e

def test_all_seventeen_packets_replayed(packets):
    r=m.inspect_parameters(packets.out/"physical.jsonl")
    assert r["valid_packets"]==17 and r["complete_endpoint_sequence"]
    assert r["material_identified"] is r["teacher_admission"] is False

@pytest.mark.parametrize("damage",["missing","identity","config","native","extra"])
def test_invalid_or_missing_parameter_packet_cannot_pass(packets,damage):
    e=packets;p=e.rows[3]["payload"]["exchange"]["material_director"]
    if damage=="missing":del p["parameter_readback"]
    elif damage=="identity":p["parameter_readback"]["identity"]["physics_step"]+=1
    elif damage=="config":p["parameter_readback"]["configured"]["rod"]["radius_m"]*=2
    elif damage=="native":p["parameter_readback"]["fields"]["radius_m"]["native_measured"]=True
    elif damage=="extra":p["parameter_readback"]["extra"]=1
    journal(e.out/"physical.jsonl",e.rows)
    with pytest.raises((ValueError,KeyError,TypeError)):m.inspect_parameters(e.out/"physical.jsonl")

@pytest.mark.parametrize("parent_pass",[False,True])
def test_parameter_audit_cannot_bypass_parent_failure(monkeypatch,packets,parent_pass):
    e=packets;checked=m.inspect_parameters(e.out/"physical.jsonl")
    calls=[]
    def old(*args,**kwargs):
        calls.append(True)
        return dict(lifecycle_qualified=parent_pass,journal={"parameter_audit":checked},
                    reason=None if parent_pass else "original joint guard failed")
    monkeypatch.setattr(m.previous,"audit_outcome",old)
    r=m.audit_outcome(e.out,e.out/"unused-CPU-outer","explicit")
    assert calls==[True] and r["parameter_readback_qualified"] is parent_pass
    assert r["teacher_admission"] is r["grasp_qualified"] is False

def test_invalid_packet_revokes_qualified_parent(monkeypatch,packets):
    e=packets;p=e.rows[3]["payload"]["exchange"]["material_director"]["parameter_readback"]
    p["reward_radius_sample"]=.005
    journal(e.out/"physical.jsonl",e.rows)
    monkeypatch.setattr(m.previous,"audit_outcome",lambda *a,**k:dict(lifecycle_qualified=True,journal={},reason=None))
    r=m.audit_outcome(e.out,e.out/"unused-CPU-outer","explicit")
    assert r["lifecycle_qualified"] is r["parameter_readback_qualified"] is False
    assert r["parameter_audit_error"]
