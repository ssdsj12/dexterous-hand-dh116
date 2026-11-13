"""Human-readable v2 acquisition scope; original setup body is unchanged."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

ORIGINAL_SOURCE_SHA256 = "daf43e49428f1d7c8e43efa34607404b0ed83c29cfb2c83d73066b64d421b2e5"
NATIVE_LOCALS = ('sim', 'hand', 'view', 'composer', 'runtime', 'provider', 'asset_sim_view', 'manager_sim_view', 'snapshot', 'mesh_poses', 'host', 'materials', 'matcfg', 'full', 'grids', 'packet', 'reset', 'sdk_properties', 'binding', 'geometry', 'bodies')

def record_detachment(output, operation_raised):
    path=Path(output)/"build_frame_detached.json"
    with path.open("x",encoding="utf-8") as stream:
        json.dump({"version":"dh116-build-frame-detachment-v2",
            "clearing_completed":True,"operation_raised":operation_raised,
            "cleared_locals":list(NATIVE_LOCALS),"native_release_verified":False},stream,indent=2)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())

def build_native_detached(sim, args, contract, cfg, project):
    hand = view = composer = runtime = provider = None
    asset_sim_view = manager_sim_view = snapshot = mesh_poses = host = None
    materials = matcfg = full = grids = packet = reset = None
    sdk_properties = binding = geometry = bodies = None
    try:
        import numpy as np
        from isaacsim.core.simulation_manager import SimulationManager
        from scripts.probe_dh116_physx_lifetime import record_reset
        from scripts.verify_dh116_deformx_pinch_contact import serial
        from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import FreeRodRuntime
        from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
        from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from pxr import UsdShade
        from ropesim import sdf
        from go2_pvcnn.control.dh116.deformx_dh116_colliders import read_pad_triangles
        from go2_pvcnn.control.dh116.runtime_mimic import install_hand_mimic, configure_mimic_compliance
        from go2_pvcnn.control.dh116.tabletop_material import (
            hand_material_contract, material_config_fields, bind_hand_materials)
        from go2_pvcnn.control.dh116.tabletop_stereo_rod_state import DeformableRodStateProvider
        from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
        from go2_pvcnn.control.dh116.tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S

        def host(v): return v.detach().cpu().numpy().astype(float,copy=True)
        project=Path(__file__).resolve().parents[1]
        reference=json.loads((project/"configs/dh116_training/pose_reference.json").read_text())
        side=cfg["side"]
        start=np.asarray(reference["pose_contract"][side]["hand_active_q"])
        pose=reference["resolved_poses"][side]["dh116_base_pose_w"]
        path=f"/World/DeformXPinch/{side}Hand"
        h=cfg["hand"]
        hand=Articulation(ArticulationCfg(
            prim_path=path,
            spawn=sim_utils.UsdFileCfg(usd_path=contract["source_usd"]["path"],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=h["fixed_base"],enabled_self_collisions=h["self_collision"],
                    solver_position_iteration_count=64,solver_velocity_iteration_count=16)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pose[:3]),rot=tuple(pose[3:]),
                     joint_pos=dict(zip(JOINTS,map(float,expand(start))))),
            actuators={
                "active":ImplicitActuatorCfg(joint_names_expr=list(ACTIVE),stiffness=h["active_stiffness"],
                    damping=h["active_damping"],effort_limit_sim=h["active_effort_limit_nm"],
                    velocity_limit_sim=ACTIVE_VELOCITY_LIMIT_RAD_S),
                "passive":ImplicitActuatorCfg(joint_names_expr=[j for j in JOINTS if j not in ACTIVE],
                    stiffness=0.,damping=0.,effort_limit_sim=0.,velocity_limit_sim=3.)}))
        install_hand_mimic(sim.stage,path)
        configure_mimic_compliance(sim.stage,path,h["mimic_stiffness"],h["mimic_damping"])
        spec=hand_material_contract(h["material_profile"])["hand_contact_material"]
        materials=[]
        for label,soft in (("hard",False),("soft",True)):
            matpath=f"/World/PinchMaterials/{label}"
            matcfg=sim_utils.RigidBodyMaterialCfg(**material_config_fields(spec,compliant=soft))
            matcfg.func(matpath,matcfg)
            materials.append(UsdShade.Material(sim.stage.GetPrimAtPath(matpath)))
        _,material_report=bind_hand_materials(sim.stage,path,materials[0],spec,materials[1])
        layout=reference["reference_layout"]
        _, after_reset = record_reset(sim, args.output)
        sdk_setup = after_reset["sdk_step"]
        hand.reset()
        view=hand.root_physx_view
        composer=hand.permanent_wrench_composer
        if set(hand.joint_names) != set(JOINTS) or not hand.is_fixed_base:
            raise RuntimeError("original articulation joint/base identity changed")
        active=[hand.joint_names.index(n) for n in ACTIVE]
        bodies=[hand.body_names.index(n) for n in DISTAL_PAD_LINKS]
        binding={}
        for name,body in zip(DISTAL_PAD_LINKS,bodies):
            bodypath=view.link_paths[0][body]
            if bodypath != f"{path}/{name}":
                raise RuntimeError("actual PhysX body identity differs")
            _,_,geometry=read_pad_triangles(sim.stage,bodypath)
            if geometry["geometry_sha256"] != contract["pad_bindings"][name]["geometry_sha256"]:
                raise RuntimeError("runtime original collider differs from locked SDF")
            binding[name]={"body_path":bodypath,"body_index":body,"mesh_index":len(binding)}
        sdk_properties={name:host(getattr(view,method)()[0]) for name,method in (
            ("limits","get_dof_limits"),("stiffness","get_dof_stiffnesses"),
            ("damping","get_dof_dampings"),("effort_limit","get_dof_max_forces"),
            ("velocity_limit","get_dof_max_velocities"))}
        for name,expected in (("stiffness",h["active_stiffness"]),("damping",h["active_damping"]),
                              ("effort_limit",h["active_effort_limit_nm"]),("velocity_limit",.3)):
            if not np.allclose(sdk_properties[name][active],expected,atol=1e-6,rtol=0):
                raise RuntimeError(f"SDK active actuator {name} mismatch")
        full=hand.data.default_joint_pos.clone()
        for name,value in zip(JOINTS,expand(start)):
            full[:,hand.joint_names.index(name)]=float(value)
        hand.write_joint_state_to_sim(full,torch.zeros_like(full))
        hand.set_joint_position_target(full[:,active],joint_ids=active)
        hand.update(0.)
        if not np.allclose(host(hand.data.joint_pos[0])[active],start,atol=1e-5,rtol=0):
            raise RuntimeError("actual reset joint state mismatch")

        def snapshot():
            d=hand.data
            return {"q":host(d.joint_pos[0]),"qd":host(d.joint_vel[0]),
                    "link_pos":host(d.body_link_pos_w[0,bodies]),
                    "link_quat":host(d.body_link_quat_w[0,bodies]),
                    "com_pos":host(d.body_com_pos_w[0,bodies])}
        def mesh_poses(state):
            return {name:(state["link_pos"][i].astype(np.float32),
                          state["link_quat"][i][[1,2,3,0]].astype(np.float32))
                    for i,name in enumerate(DISTAL_PAD_LINKS)}
        grids={}
        for name,row in contract["pad_bindings"].items():
            with np.load(row["sdf_path"],allow_pickle=False) as packet:
                grids[name]=sdf.SdfGrid(packet["values"].copy(),packet["origin"].copy(),float(packet["spacing"]))
        runtime=FreeRodRuntime(cfg,layout["wire_points"],grids,mesh_poses(snapshot()),
            layout["table_center"],layout["table_size"],episode_uid="construction")
        reset=runtime.reset(physics_step=sdk_setup,episode_uid="pinch-left-"+uuid.uuid4().hex,reset_id=1,pad_poses=mesh_poses(snapshot()))
        np.testing.assert_allclose(reset["node_positions"][0],layout["wire_points"],atol=2e-7,rtol=0)
        if np.max(np.abs(reset["velocities"])) > 1e-7:
            raise RuntimeError("rod reset velocity did not clear")
        provider=DeformableRodStateProvider(runtime,backend="DeformX",
            backend_artifact=args.ropesim_python/"ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so",
            backend_sha256=contract["sources"]["external_sources"][str(args.ropesim_python/"ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so")])

        asset_sim_view = hand._physics_sim_view
        manager_sim_view = SimulationManager.get_physics_sim_view()
        if asset_sim_view is not manager_sim_view:
            raise RuntimeError("hand and manager root simulation aliases differ")
        if provider._reader.__self__ is not runtime:
            raise RuntimeError("provider does not retain the actual runtime reader")
        metadata=serial({
            "source_acquisition_sha256":contract["code_config_sha256"],
            "root_sim_aliases_identical":True,
            "provider_bound_to_runtime":True,
            "actual_pad_bindings":binding, "hand_material":material_report,
            "actuator_readback":sdk_properties,
            "rod_parameter_readback":runtime.parameter_readback,
            "rod_mass_readback":runtime.mass_readback, "reset_readback":reset,
            "rod_tick":runtime.rod_tick, "solver_time_origin_s":runtime.time_origin,
            "solver_time_s":float(runtime.rope.time), "physics_step":runtime.physics_step,
            "authored_physx_table":False, "rod_table_sdf_present":True,
            "measured_solver_step_performed":False, "provider_snapshot_performed":False,
            "native_environments":int(runtime.rope.num_envs),
            "render_products_created":0, "teacher_admission":False, "grasp_qualified":False})
        from scripts.probe_dh116_physx_lifetime import write_json
        write_json(args.output/"acquisition.json",metadata)
        return SimpleNamespace(hand=hand,hand_view=view,composer=composer,runtime=runtime,
            provider=provider,rope=runtime.rope,asset_sim_view=asset_sim_view,
            manager_sim_view=manager_sim_view,metadata=metadata)
    finally:
        # A retained completed builder frame must not retain its native aliases.
        sim = None
        hand = view = composer = runtime = provider = None
        asset_sim_view = manager_sim_view = snapshot = mesh_poses = host = None
        materials = matcfg = full = grids = packet = reset = None
        sdk_properties = binding = geometry = bodies = None
        record_detachment(args.output, sys.exc_info()[0] is not None)
