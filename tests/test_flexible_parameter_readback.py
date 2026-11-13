"""CPU native-API doubles; no Isaac/CUDA import or physical qualification."""
import copy
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref
import numpy as np
import pytest
import torch
from go2_pvcnn.control.dh116 import tabletop_flexible_parameter_readback as m
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import FreeRodRuntime,lumped_masses,verify_mass_readback
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import DualHandRodRuntime

@pytest.fixture
def case(monkeypatch):
    cfg=json.loads((m.ROOT/m.CONFIG).read_text())
    nodes=np.zeros((41,3),np.float32);nodes[:,0]=np.linspace(0,.5,41)
    events=[]
    class Rope:
        num_envs=1;verts_per_rope=41;fixed_root_orientation=False;time=0.;last_error=0
        def __init__(self):
            self.inv=(1/lumped_masses(nodes,cfg["rod"]["linear_density_kg_m"])).astype(np.float32)[None]
            for name in m.GETTERS:setattr(self,name,cfg["rod"]["solver_iterations" if name=="num_itr" else name])
        def inv_masses(self):events.append("inv_masses");return self.inv
    rope=Rope()
    base=FreeRodRuntime.__new__(FreeRodRuntime)
    base.cfg=copy.deepcopy(cfg);base.nodes=nodes.copy();base.rest=np.linalg.norm(np.diff(nodes,axis=0),axis=1)
    base.rope=rope;base.physics_step=2;base.rod_tick=0;base.episode_uid="episode-1";base.reset_id=1
    base.mass_readback=verify_mass_readback(rope.inv,lumped_masses(nodes,cfg["rod"]["linear_density_kg_m"]))
    base.parameter_readback={name:float(getattr(rope,name)) for name in m.GETTERS}
    runtime=DualHandRodRuntime.__new__(DualHandRodRuntime)
    runtime._base=base;runtime._needs_reset=False;runtime.run_uuid="run-1";runtime.env_id=0;runtime.rod_uid="rod-0"
    sim=SimpleNamespace(device="cuda:0",current_time_step_index=2,current_time=2/480)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda device:events.append("sync"))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda device:0)
    monkeypatch.setattr(torch.cuda,"default_stream",lambda device:0)
    backend=SimpleNamespace(__file__=str(m.BACKEND),RopeBatch=Rope)
    return SimpleNamespace(runtime=runtime,base=base,rope=rope,sim=sim,backend=backend,events=events,
        expected=runtime.clock(),device="cuda:0")


def capture(c):
    return m.capture_parameters(c.runtime,sim=c.sim,backend=c.backend,expected_clock=c.expected,device=c.device)


def test_source_basis_matches_real_getters_and_no_invented_setter_readback():
    basis=m.source_basis()
    assert basis["getters"]==list(m.GETTERS)
    assert basis["native_getters_unavailable"]==["radius","gravity","EA","EI","friction","rest_lengths"]
    assert basis["material_identified"] is False


def test_actual_getters_configuration_and_unknowns_stay_separate(case):
    c=case;r=capture(c)
    assert c.events==["sync","inv_masses","sync"]
    assert r["identity"]==c.expected
    f=r["fields"]
    assert f["mass_kg"]["provenance"]=="native_getter_derived"
    assert f["mass_kg"]["value"]==pytest.approx(.0325,rel=2e-5)
    assert f["radius_m"]["value"]==.005
    assert f["radius_m"]["provenance"]=="source_locked_configuration_not_native_readback"
    assert f["radius_m"]["native_measured"] is False
    assert f["damping"]["native_measured"] is True
    assert f["node_count"]["value"]==41
    assert f["young_modulus_pa"]["value"] is f["shear_modulus_pa"]["value"] is None
    assert f["density_kg_m3"]["value"] is None
    assert all(not x["hardware_measured"] for x in f.values())
    assert r["new_physics_steps"]==r["new_solver_steps"]==0
    assert r["teacher_admission"] is r["native_qualification"] is False
    assert r["reward_radius_sample"] is None
    c.rope.inv[:]=0;c.base.cfg["rod"]["radius_m"]=20
    assert r["configured"]["rod"]["radius_m"]==.005 and np.all(np.asarray(r["node_inverse_masses_kg_inv"])>0)


@pytest.mark.parametrize("fault",["config","backend","type","reset","clock","bool_clock","sdk","default_stream",
    "mass","pin","shape","nan","damping","iterations","solver_error","root_fixed","node_count","rest"])
def test_bad_sources_or_readbacks_rejected(case,monkeypatch,fault):
    c=case
    if fault=="config":c.base.cfg["rod"]["radius_m"]*=2
    elif fault=="backend":c.backend.__file__="/wrong/backend.so"
    elif fault=="type":c.backend.RopeBatch=type("Different",(),{})
    elif fault=="reset":c.runtime._needs_reset=True
    elif fault=="clock":c.expected["rod_tick"]=12
    elif fault=="bool_clock":c.expected["env_id"]=False
    elif fault=="sdk":c.sim.current_time_step_index=3
    elif fault=="default_stream":monkeypatch.setattr(torch.cuda,"current_stream",lambda device:1)
    elif fault=="mass":c.rope.inv*=2
    elif fault=="pin":c.rope.inv[0,0]=0
    elif fault=="shape":c.rope.inv=c.rope.inv[0]
    elif fault=="nan":c.rope.inv[0,0]=np.nan
    elif fault=="damping":c.rope.damping=1
    elif fault=="iterations":c.rope.num_itr=True
    elif fault=="solver_error":c.rope.last_error=1
    elif fault=="root_fixed":c.rope.fixed_root_orientation=True
    elif fault=="node_count":c.rope.verts_per_rope=40
    elif fault=="rest":c.base.rest[0]+=.001
    with pytest.raises((ValueError,RuntimeError)):capture(c)


def test_clock_change_inside_getter_rejected(case):
    c=case;old=c.rope.inv_masses
    def drift():
        value=old();c.base.physics_step+=1;return value
    c.rope.inv_masses=drift
    with pytest.raises(ValueError,match="clock"):capture(c)


def test_no_retained_native_owner_even_through_error_traceback(case):
    c=case;ref=weakref.ref(c.rope);runtime_ref=weakref.ref(c.runtime)
    c.rope.inv[0,0]=0
    with pytest.raises(ValueError) as error:capture(c)
    # The test helper/caller intentionally retains c; the actual reader's frame
    # must explicitly release all native parameters before exception propagation.
    tb=error.value.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_name=="capture_parameters":
            assert all(tb.tb_frame.f_locals.get(k) is None for k in ("runtime","sim","backend","base","rope"))
        tb=tb.tb_next
    c.runtime=None;c.base=None;c.rope=None
    gc.collect()
    assert ref() is runtime_ref() is None

def test_global_friction_never_masquerades_as_hand_pair_friction(case):
    r=capture(case)
    f=r["fields"]
    assert "friction_static" not in f and "friction_dynamic" not in f
    assert f["global_friction_static"]["value"]==.5
    pairs=r["configured_contact_pairs"]
    assert set(pairs)=={"table"}|{side+"."+name for side in ("left","right")
        for name in ("finger14_link","finger23_link","finger33_link","finger43_link","finger53_link")}
    for name,v in pairs.items():
        assert v["native_readback"] is False
        assert v["mu_s"]==(.5 if name=="table" else 1.)
        assert v["mu_k"]==(.4 if name=="table" else .8)
    assert f["global_friction_static"]["native_measured"] is False


def test_config_mutation_during_getter_rejected(case):
    c=case;getter=c.rope.inv_masses
    def mutate():
        value=getter();c.base.cfg["rod"]["radius_m"]*=2;return value
    c.rope.inv_masses=mutate
    with pytest.raises(ValueError,match="source changed"):capture(c)


def test_stream_change_during_getter_rejected(case,monkeypatch):
    c=case;getter=c.rope.inv_masses
    def mutate():
        value=getter();monkeypatch.setattr(torch.cuda,"current_stream",lambda device:1);return value
    c.rope.inv_masses=mutate
    with pytest.raises(ValueError,match="stream"):capture(c)
