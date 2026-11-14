"""Historical pre-action source reconstruction is CPU replay, not a new run."""
from copy import deepcopy
import importlib,importlib.util
from functools import lru_cache
import pytest

def api():
    name="scripts.audit_dh116_left_pinch_preprojection"
    assert importlib.util.find_spec(name) is not None,"pre-action native source replay missing"
    return importlib.import_module(name)

@lru_cache(maxsize=1)
def source():
    m=api()
    return m,m.load_case()

def test_previous_native_links_reconstruct_safe_z_exactly():
    m,c=source();r=m.evaluate(*c)
    assert r["sides"]["left"]["limiting_source_shape"]=="finger13_link"
    assert r["sides"]["right"]["limiting_source_shape"]=="finger22_link"
    assert r["replay_max_error_m"]<1e-12
    assert r["sides"]["left"]["projection_delta_mm"]==pytest.approx(.000809032891944)
    assert r["pre_identity"]["physics_step"]==175
    assert r["action_identity"]["physics_step"]==176
    assert not r["teacher_admission"]

@pytest.mark.parametrize("bad",["clock","episode","pose","projection","request"])
def test_wrong_time_or_tampering_rejected(bad):
    m,c=source();previous,current,g,pair=c
    previous,current=deepcopy(previous),deepcopy(current)
    if bad=="clock":previous["physics_step"]-=1
    if bad=="episode":previous["closure_geometry"]["raw"]["identity"]["episode_uid"]="other"
    if bad=="pose":previous["closure_geometry"]["raw"]["hands"]["left"]["link_poses_xyzw"][0,:,2]+=.001
    if bad=="projection":current["wrist_projection"]["palm_offsets_m"][0,0,2]+=.001
    if bad=="request":current["action_readback"]["request"]["requested_palm_offsets_m"][0,0,2]+=.001
    with pytest.raises(ValueError):m.evaluate(previous,current,g,pair)
