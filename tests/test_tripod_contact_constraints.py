import numpy as np
from scripts import solve_dh116_tripod_hard_constraints as m
from tests.test_tripod_hard_constraints import row

def test_positive_sdf_is_near_contact_but_not_contact_candidate():
    r=row();r["node_sdf_gap_m"]=[.00001]*3
    assert min(m.contract(r)["constraints"])>0
    q=m.contract(r,require_contact=True)
    assert min(q["constraints"])<0
    assert q["require_node_contact"] is True

def test_three_negative_sdf_candidates_keep_original_penetration_floor():
    r=row();q=m.contract(r,require_contact=True)
    assert min(q["constraints"])>0
    r["node_sdf_gap_m"][2]=-.00006
    assert min(m.contract(r,require_contact=True)["constraints"])<0

def test_contact_target_never_changes_native_qualification():
    q=m.contract(row(),require_contact=True)
    assert q["native_ready"] is False
    assert q["node_upper_search_target_m"]==-.000005
