import ast
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/go2_pvcnn/control/dh116/tabletop_deformable_grasp_runtime.py"


def runtime():
    assert MODULE.exists(), "real native runtime mapping has not been implemented"
    from go2_pvcnn.control.dh116 import tabletop_deformable_grasp_runtime
    return tabletop_deformable_grasp_runtime


def test_nonuniform_lumped_masses_preserve_linear_density_and_free_ends():
    m = runtime().lumped_masses(np.array([[0, 0, 0], [.1, 0, 0], [.3, 0, 0]]), .065)
    np.testing.assert_allclose(m, [.00325, .00975, .0065])
    assert m.sum() == pytest.approx(.065 * .3)
    assert (m > 0).all()


def test_pinch_action_retains_opposite_hand_and_rejects_wrong_action_channels():
    r = runtime()
    q = np.array([.3, .4, .5, .2, .2, .2])
    a = np.zeros(24); a[1:3] = [.5, -.5]
    target = r.integrate_pinch_action(q, q, a, side="left", dt=1/480)
    np.testing.assert_allclose(target-q, [0, .15/480, -.15/480, 0, 0, 0])
    a[7] = .1
    with pytest.raises(ValueError, match="inactive"):
        r.integrate_pinch_action(q, q, a, side="left", dt=1/480)


def test_distinct_named_wrenches_not_aggregate_node_force_fabrication():
    r = runtime()
    w = np.array([[[1, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0], [0, 0, -.1, 0, 0, 0]]])
    names = ["finger14_link", "finger23_link", "table"]
    mapping = r.named_wrenches(w, names)
    assert mapping["finger14_link"][0] == 1
    assert mapping["finger23_link"][0] == -2
    assert mapping["table"][2] == -.1
    with pytest.raises(ValueError, match="distinct"):
        r.named_wrenches(w, ["finger14_link", "finger14_link", "table"])


def test_free_mass_readback_rejects_pinned_or_wrong_density():
    r = runtime()
    masses = np.array([.001, .002, .001])
    assert r.verify_mass_readback(1/masses, masses)["pinned_node_ids"] == []
    with pytest.raises(ValueError, match="pinned"):
        r.verify_mass_readback([0, 500, 1000], masses)
    with pytest.raises(ValueError, match="mass"):
        r.verify_mass_readback([1000, 1000, 1000], masses)


def test_r0_contact_counts_only_actual_named_required_pad_wrenches():
    r = runtime()
    w = {"finger14_link": [1,0,0,0,0,0], "finger23_link": [0,0,0,0,0,0],
         "finger33_link": [-1,0,0,0,0,0], "table": [0,0,-.1,0,0,0]}
    result = r.contact_metrics(w)
    assert result["required_contact_fraction"] == .5
    assert result["wrench_opposition_candidate"] is False
    assert result["table_contact"] is True


def test_box_distance_has_finite_table_edges():
    r = runtime()
    values = r.box_sdf(np.array([[0,0,.05], [0,0,0], [.6,0,.05]]), [.8,1.2,.1])
    np.testing.assert_allclose(values, [0,-.05,.2])


def test_material_mapping_does_not_invent_shear_or_volumetric_density():
    r = runtime()
    cfg = json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_text())
    params = r.material_mapping(cfg)
    assert params["density_unit"] == "kg/m"
    assert params["shear_modulus"]["status"] == "unsupported_independent_parameter"
    assert params["young_modulus"]["status"] == "not_identifiable_from_independent_EA_EI"
    assert params["native_getters"]["gravity"] is False
    assert params["native_getters"]["mass"] is True


def test_real_runtime_calls_locked_api_and_never_root_driving():
    runtime()
    tree = ast.parse(MODULE.read_text())
    methods = {n.func.attr for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"from_positions", "step_free", "inv_masses", "reset_envs",
            "add_mesh", "set_mesh_poses", "mesh_wrenches", "contact_forces"} <= methods
    assert not {"step_program", "apply_roots", "set_root"} & methods

def test_native_preflight_is_cpu_only_and_binds_all_pads_and_candidate_bytes():
    path = ROOT/"scripts/verify_dh116_deformx_pinch_contact.py"
    assert path.exists(), "bounded Pinch native entrypoint is missing"
    from scripts.verify_dh116_deformx_pinch_contact import preflight, default_arguments
    args=default_arguments(ROOT)
    report, _ = preflight(args,ROOT)
    assert report["physics_budget"]["total_physics_steps"] == 240
    assert report["physics_budget"]["measurement_physics_steps"] == 238
    assert len(report["pad_bindings"]) == 5
    assert report["native_execution_performed"] is False
    assert len(report["code_config_sha256"]) == 64
    assert report["grasp_qualified"] is False

def test_bounded_ledger_refuses_empty_or_forged_clock():
    r=runtime()
    assert hasattr(r,"validate_pinch_ledger"), "multi-pad evidence validator is missing"
    with pytest.raises(ValueError,match="238"):
        r.validate_pinch_ledger([])
    with pytest.raises(ValueError,match="identity"):
        r.validate_pinch_ledger([{"phase_step":i+1,"sdk_step":i+4,"rod_substep":(i+1)*12}
                                 for i in range(238)])

def test_disk_validator_requires_independent_outer_process_exit(tmp_path):
    from scripts import verify_dh116_deformx_pinch_contact as entry
    assert hasattr(entry,"validate_pinch_artifacts"), "disk/outer-exit validation is missing"
    outer=tmp_path/"outer_exit.json"
    outer.write_text(json.dumps({"outer_exit_code":139,"native_lock_observed":True}))
    with pytest.raises(ValueError,match="outer"):
        entry.validate_pinch_artifacts(tmp_path,outer,"a"*64)

def test_candidate_binds_actual_local_original_usd_material_on_cpu():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics, UsdShade, Sdf
    from scripts.verify_dh116_cooked_convex_hulls_native import _sources
    from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract, bind_hand_materials
    stage=Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Hand","Xform").GetReferences().AddReference(_sources(ROOT)["left"]["path"])
    spec=hand_material_contract("grip_compliant")["hand_contact_material"]
    mats=[]
    for label,soft in (("Hard",False),("Soft",True)):
        mat=UsdShade.Material.Define(stage,"/"+label)
        physics=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        physics.CreateStaticFrictionAttr(1.)
        physics.CreateDynamicFrictionAttr(.8)
        physics.CreateRestitutionAttr(0.)
        prim=mat.GetPrim()
        for field,value in (("frictionCombineMode","max"),("restitutionCombineMode","min")):
            prim.CreateAttribute("physxMaterial:"+field,Sdf.ValueTypeNames.Token).Set(value)
        for field,value in (("compliantContactStiffness",10000. if soft else 0.),
                            ("compliantContactDamping",30. if soft else 0.)):
            prim.CreateAttribute("physxMaterial:"+field,Sdf.ValueTypeNames.Float).Set(value)
        mats.append(mat)
    paths,report=bind_hand_materials(stage,"/Hand",mats[0],spec,mats[1])
    assert report["compliant_pad_colliders"] == 5
    assert report["colliders"] == len(paths) and len(paths) > 5

@pytest.mark.parametrize("failure",["step","provider","validator"])
def test_native_owners_release_on_fault_before_sdk_stop(failure):
    import weakref
    r=runtime()
    assert hasattr(r,"run_owned_call"), "exception-safe owned lifetime is missing"
    events,refs=[],[]
    class Owned:
        def reset(self): events.append("composer.reset")
    class Trace:
        def checkpoint(self,phase): events.append(phase)
    def operation(owners):
        hand=owners.keep("hand",Owned())
        view=owners.keep("view",Owned())
        composer=owners.keep("composer",Owned())
        runtime_owner=owners.keep("runtime",Owned())
        provider=owners.keep("provider",Owned())
        provider.runtime=runtime_owner
        snapshot=lambda:hand
        refs.extend(weakref.ref(v) for v in (hand,view,composer,runtime_owner,provider))
        events.append(failure)
        raise RuntimeError("injected "+failure)
    with pytest.raises(RuntimeError,match="injected "+failure):
        r.run_owned_call(operation,synchronize=lambda:events.append("sync"),ownership=Trace())
    events.append("SDK STOP")
    assert all(ref() is None for ref in refs)
    assert events.index("composer.reset") < events.index("sync") < events.index("SDK STOP")
    assert events.index("owned hand and solver references dropped") < events.index("SDK STOP")

def test_validate_cli_actually_reads_outer_exit_without_native_import(tmp_path, monkeypatch):
    from scripts import verify_dh116_deformx_pinch_contact as entry
    outer=tmp_path/"outer_exit.json"
    outer.write_text(json.dumps({"outer_exit_code":139,"native_lock_observed":True}))
    monkeypatch.setattr("sys.argv",["pinch","--validate-output",str(tmp_path),"--outer-exit",str(outer)])
    with pytest.raises(ValueError,match="outer"):
        entry.main()


def test_run_app_fault_drops_simulation_before_app_close(monkeypatch,tmp_path):
    import sys,types,weakref
    from scripts import verify_dh116_deformx_pinch_contact as entry
    from go2_pvcnn.control.dh116 import tabletop_stereo_contact_calibration as close_module
    events,refs=[],[]
    class Settings:
        def set_bool(self,*args): pass
        def get(self,*args): return False
    settings=Settings()
    class Sim:
        def __init__(self,*args,**kwargs):
            self.current_time_step_index=4
            refs.append(weakref.ref(self))
        @classmethod
        def clear_instance(cls): events.append("clear")
    def module(name,**attrs):
        obj=types.ModuleType(name)
        obj.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,obj)
        return obj
    settings_module=module("carb.settings",get_settings=lambda:settings)
    module("carb",settings=settings_module)
    sim_module=module("isaaclab.sim",SimulationContext=Sim,SimulationCfg=lambda **k:k,PhysxCfg=lambda **k:k)
    module("isaaclab",sim=sim_module)
    module("isaacsim");module("isaacsim.core")
    module("isaacsim.core.simulation_manager",SimulationManager=types.SimpleNamespace(get_physics_sim_view=lambda:None))
    timeline=types.SimpleNamespace(stop=lambda:events.append("STOP"))
    timeline_module=module("omni.timeline",get_timeline_interface=lambda:timeline)
    module("omni",timeline=timeline_module)
    def fail(sim,*args):
        snapshot=lambda:sim
        raise RuntimeError("injected provider boundary")
    monkeypatch.setattr(entry,"_run",fail)
    def close(app,output):
        assert all(r() is None for r in refs), "SimulationContext retained by original traceback"
        events.append("close")
    monkeypatch.setattr(close_module,"close_calibration_app",close)
    trace=types.SimpleNamespace(watch=lambda *a,**k:None,event=lambda *a,**k:None,checkpoint=lambda *a:None)
    app=types.SimpleNamespace(update=lambda:events.append("update"))
    args=types.SimpleNamespace(device="cuda:0",output=tmp_path)
    with pytest.raises(RuntimeError,match="injected provider boundary"):
        entry.run_app(app,args,{},{"rod":{"gravity_world_m_s2":[0,0,-9.81]}},trace)
    assert events.index("STOP") < events.index("close")
