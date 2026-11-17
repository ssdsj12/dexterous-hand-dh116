"""Versioned Pinch acquisition; explicit retained-frame detachment on all paths."""
import faulthandler
import gc
import json
import os
from pathlib import Path
import sys
import traceback
import uuid
from types import SimpleNamespace
import weakref

import numpy as np

from scripts.dh116_pinch_evidence_v2 import (
    DT,PAIR_CAPACITY,CONTACT_CAPACITY,phase_pair_record,read_table_contacts,require_quiet,serial)
from go2_pvcnn.control.dh116.tabletop_native_lifetime_audit import snapshot_weakrefs
from scripts.probe_dh116_physx_lifetime import (
    OWNED_NAMES as BASE_OWNED_NAMES,LifetimeTrace,write_json,clock)
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call

OWNED_NAMES=(*BASE_OWNED_NAMES,"contact_view")
VERSION="dh116-single-left-pinch-v2"


def record_detachment(output,operation_raised):
    write_json(Path(output)/"build_frame_detached.json",dict(
        version="dh116-pinch-builder-detachment-v2",clearing_completed=True,
        operation_raised=bool(operation_raised),native_release_verified=False))


def author_contact_reports_only(stage,root_path):
    """Add measurement reports without IsaacLab's sleep-threshold mutation."""
    prim = rigid = report = None
    try:
        from pxr import Usd,UsdPhysics,PhysxSchema
        records={}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            rigid=PhysxSchema.PhysxRigidBodyAPI(prim)
            before=rigid.GetSleepThresholdAttr().Get()
            report=PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr().Set(0.)
            after=rigid.GetSleepThresholdAttr().Get()
            if after!=before:
                raise RuntimeError("contact instrumentation changed authored sleep threshold")
            records[str(prim.GetPath())]=dict(authored_sleep_threshold_before=before,
                authored_sleep_threshold_after=after)
        if len(records)!=17:
            raise RuntimeError("exact original seventeen rigid body reporters required")
        return records
    finally:
        stage = prim = rigid = report = None


def refresh_kinematics_without_step(sim,hand):
    try:
        before=clock(sim)
        hand._physics_sim_view.update_articulations_kinematic()
        hand.update(DT)  # Invalidate asset caches only; no SimulationContext step.
        if clock(sim)!=before:
            raise RuntimeError("kinematic refresh unexpectedly changed SDK clock")
    finally:
        sim = hand = None


def build_native(sim, args, contract, cfg, project):
    hand = view = composer = runtime = provider = contact_view = geometry = geometry_capture = None
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
        start=np.asarray(cfg["initial_pose"]["active_q_rad"])
        pose=cfg["initial_pose"]["base_pose_wxyz"]
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
        authored_reports=author_contact_reports_only(sim.stage,path)
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
        tablecfg=sim_utils.CuboidCfg(size=tuple(layout["table_size"]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=cfg["table"]["mu_s"],
                dynamic_friction=cfg["table"]["mu_k"],restitution=0.))
        tablecfg.func("/World/PinchTable",tablecfg,translation=tuple(layout["table_center"]))
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
        refresh_kinematics_without_step(sim,hand)
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
        from scripts.audit_dh116_pinch_initial_pose import load_geometry
        from go2_pvcnn.control.dh116.tabletop_pinch_initial_geometry_v2 import capture_pinch_initial_geometry
        geometry=load_geometry(project,side="left")
        try:
            from omni.physx import get_physx_cooking_interface
            cooking=get_physx_cooking_interface()
        except (ImportError,AttributeError,RuntimeError):
            cooking=None
        geometry_capture=capture_pinch_initial_geometry(sim.stage,hand,geometry,
            cooking_interface=cooking,
            read_clock=lambda:dict(sdk_step=int(sim.current_time_step_index),sdk_time_s=float(sim.current_time)),
            table_center_m=layout["table_center"],table_size_m=layout["table_size"],
            rod_nodes_w=layout["wire_points"],rod_radius_m=cfg["rod"]["radius_m"],
            minimum_table_gap_m=0.,minimum_rod_gap_m=0.)
        cooking=None
        write_json(args.output/"initial_geometry.json",serial(geometry_capture))
        if geometry_capture["initial_geometry_passed"] is not True:
            raise RuntimeError("actual initial source/locked-pad geometry is not clear")
        grids={}
        for name,row in contract["pad_bindings"].items():
            with np.load(row["sdf_path"],allow_pickle=False) as packet:
                grids[name]=sdf.SdfGrid(packet["values"].copy(),packet["origin"].copy(),float(packet["spacing"]))
        runtime=FreeRodRuntime(cfg,layout["wire_points"],grids,mesh_poses(snapshot()),
            layout["table_center"],layout["table_size"],episode_uid="construction")
        runtime.rope.enable_contact_pair_capture(PAIR_CAPACITY)
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
        from pxr import PhysxSchema
        reporting=[]
        actual_paths=list(view.link_paths[0])
        if len(actual_paths)!=17 or len(set(actual_paths))!=17:
            raise RuntimeError("exact original seventeen actual body paths required")
        for bodypath in actual_paths:
            prim=sim.stage.GetPrimAtPath(bodypath)
            enabled=prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            threshold=PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr().Get() if enabled else None
            if not enabled or threshold!=0:
                raise RuntimeError("actual original body contact reporter is absent or filtered")
            reporting.append(dict(body_path=bodypath,contact_report_enabled=True,threshold=threshold,
                **authored_reports[bodypath]))
        contact_view=asset_sim_view.create_rigid_contact_view(
            actual_paths,filter_patterns=[["/World/PinchTable"] for _ in actual_paths],
            max_contact_data_count=CONTACT_CAPACITY)
        if contact_view.sensor_count!=17 or contact_view.filter_count!=1:
            raise RuntimeError("actual contact reporter sensor/filter count differs")
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
            "authored_physx_table":True, "rod_table_sdf_present":True,
            "measured_solver_step_performed":False, "provider_snapshot_performed":False,
            "native_environments":int(runtime.rope.num_envs),
            "contact_reporting":reporting,"contact_table_filter":"/World/PinchTable",
            "contact_capacity":CONTACT_CAPACITY,"pair_capacity":PAIR_CAPACITY,
            "joint_names":list(hand.joint_names),"body_names":list(hand.body_names),
            "active_ids":active,"pad_body_ids":bodies,
            "initial_geometry_passed":True,"initial_geometry_is_contact_measurement":False,
            "render_products_created":0, "teacher_admission":False, "grasp_qualified":False})
        write_json(args.output/"acquisition.json",metadata)
        return SimpleNamespace(hand=hand,hand_view=view,composer=composer,runtime=runtime,
            provider=provider,rope=runtime.rope,asset_sim_view=asset_sim_view,
            manager_sim_view=manager_sim_view,contact_view=contact_view,metadata=metadata)
    finally:
        # A retained completed builder frame must not retain its native aliases.
        sim = None
        hand = view = composer = runtime = provider = contact_view = geometry = geometry_capture = None
        asset_sim_view = manager_sim_view = snapshot = mesh_poses = host = None
        materials = matcfg = full = grids = packet = reset = cooking = prim = tablecfg = None
        sdk_properties = binding = geometry = bodies = None
        record_detachment(args.output, sys.exc_info()[0] is not None)


def measure_owned(sim,args,contract,cfg,owners,trace,project):
    bundle = hand = view = composer = runtime = provider = contact_view = None
    snapshot = mesh_poses = state = rod = before = after = None
    try:
        import torch
        from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
        from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,MIMIC,active_limits
        from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
        from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import (
            candidate_action,integrate_pinch_action,contact_metrics,validate_pinch_ledger)
        from scripts.verify_dh116_deformx_rigid_bridge import rotation_wxyz,wrench_at_com
        from scripts.verify_dh116_deformx_thumb_contact import step_thumb_physics
        from scripts.run_dh116_work_packages import native_physics_budget
        bundle=build_native(sim,args,contract,cfg,project)
        for name in OWNED_NAMES:
            value=owners.keep(name,getattr(bundle,name))
            trace.watch(name,value,owner="Pinch v2 owned scope",release="explicit aliases then STOP",gpu_view=True)
        value=None
        hand,view,composer=bundle.hand,bundle.hand_view,bundle.composer
        runtime,provider,contact_view=bundle.runtime,bundle.provider,bundle.contact_view
        metadata=bundle.metadata
        binding=metadata["actual_pad_bindings"]
        active=metadata["active_ids"];bodies=metadata["pad_body_ids"]
        side="left";start=np.asarray(cfg["initial_pose"]["active_q_rad"])
        sdk_initial=0;sdk_setup=int(metadata["physics_step"])
        if sdk_setup!=2:
            raise RuntimeError("exact two setup steps required")
        write_json(args.output/"plan.json",serial({**contract,**metadata,"native_execution_performed":True,
            "rod_setter_configuration":cfg["rod"],"table_mesh_index":5,
            "state371_policy_binding":False,"palm_transport":"fixed_base_candidate"}))
        trace.checkpoint("resources_acquired")
        def host(value):
            return value.detach().cpu().numpy().astype(float,copy=True)
        def snapshot():
            d=hand.data
            try:
                return {"q":host(d.joint_pos[0]),"qd":host(d.joint_vel[0]),
                    "link_pos":host(d.body_link_pos_w[0,bodies]),
                    "link_quat":host(d.body_link_quat_w[0,bodies]),
                    "com_pos":host(d.body_com_pos_w[0,bodies])}
            finally:
                d=None
        def mesh_poses(state):
            return {name:(state["link_pos"][i].astype(np.float32),
                state["link_quat"][i][[1,2,3,0]].astype(np.float32))
                for i,name in enumerate(DISTAL_PAD_LINKS)}
        previous=np.zeros((5,6)); target=start.copy(); rows=[]; failure=None
        limits=np.stack(active_limits(),axis=-1)
        with (args.output/"physics.jsonl").open("x",encoding="utf-8") as stream, \
                (args.output/"pairs.jsonl").open("x",encoding="utf-8") as pair_stream, \
                (args.output/"table_contacts.jsonl").open("x",encoding="utf-8") as contact_stream:
            for step in range(1,cfg["measured_steps"]+1):
                before=snapshot()
                action=candidate_action(cfg,step)
                target=integrate_pinch_action(target,before["q"][active],action,side=side,dt=DT)
                hand.set_joint_position_target(torch.as_tensor(target[None],dtype=torch.float32,device=args.device),joint_ids=active)
                composer.reset()
                composer.set_forces_and_torques(
                    forces=torch.as_tensor(previous[None,:,:3],dtype=torch.float32,device=args.device),
                    torques=torch.as_tensor(previous[None,:,3:],dtype=torch.float32,device=args.device),
                    body_ids=bodies,is_global=True)
                applied=np.stack([np.r_[rotation_wxyz(before["link_quat"][i])@host(composer.composed_force_as_torch[0,b]),
                                         rotation_wxyz(before["link_quat"][i])@host(composer.composed_torque_as_torch[0,b])]
                                  for i,b in enumerate(bodies)])
                hand.write_data_to_sim()
                actual_target=host(view.get_dof_position_targets()[0])
                sdk=step_thumb_physics(sim,sdk_setup+step-1,budget_end=sdk_initial+240)
                hand.update(DT)
                after=snapshot()
                rod=runtime.step(mesh_poses(after),physics_step=sdk)
                state=provider.snapshot(env_ids=[0],episode_uids=[runtime.episode_uid],reset_ids=[runtime.reset_id],
                                        physics_step=sdk,timestamp_s=sdk*DT)
                named=rod["named_wrenches"]
                next_wrench=np.stack([wrench_at_com(named[n],after["link_pos"][i],after["com_pos"][i])
                                      for i,n in enumerate(DISTAL_PAD_LINKS)])
                mimic=np.asarray([after["q"][hand.joint_names.index(f)]-(ratio*after["q"][hand.joint_names.index(m)]+off)
                                  for f,m,ratio,off in MIMIC])
                failure=joint_safety_failure(after["q"][active],limits,mimic)
                if not np.allclose(actual_target[active],target,atol=1e-7,rtol=0):
                    failure=failure or "sdk_target_mismatch"
                if np.max(np.linalg.norm(next_wrench[:,:3],axis=1))>30:
                    failure=failure or "excess_force"
                metrics=contact_metrics(named)
                row={"phase_step":step,"sdk_step":sdk,"sdk_time_s":float(sim.current_time),"rod_substep":runtime.rod_tick,
                     "env_id":0,"physics_step":sdk,"timestamp_s":sdk*DT,"policy_step":(step-1)//8,
                     "substep":(step-1)%8,"rod_tick":runtime.rod_tick,
                     "solver_time_s":rod["solver_time_s"],"solver_time_origin_s":runtime.time_origin,
                     "dt_s":DT,"side":side,"rod_uid":"rod0","episode_uid":runtime.episode_uid,"reset_id":runtime.reset_id,
                     "actual_pad_bindings":binding,"canonical_action24":action,"sdk_joint_target_rad":actual_target,
                     "pre_state":before,"post_state":after,"rod_nodes_w":rod["node_positions"][0],
                     "rod_velocities_w":rod["velocities"][0],"rod_node_contact_forces_w":rod["distributed_contact"][0],
                     "contact_node_ids":np.flatnonzero(np.linalg.norm(rod["distributed_contact"][0],axis=-1)>1e-5).tolist(),
                     "rod_mesh_wrenches_w":named,"rod_on_hand_com_w":next_wrench,
                     "applied_wrench_com_w":applied,"applied_source_phase_step":step-1,"wrench_write_calls":1,
                     "r0_raw_measurements":metrics,"state_readback_version":state["version"],
                     "state_backend_sha256":state["backend_sha256"],"full_state371_bound":False,
                     "pair_force_residual_n":float(np.linalg.norm(rod["distributed_contact"].sum((0,1))+
                                                                  np.sum([w[:3] for w in named.values()],axis=0))),
                     "external_authored_force_on_rod":False,"attachments":[],"failure":failure}
                pair_record=phase_pair_record(runtime.rope.contact_pair_samples(),row)
                contacts=read_table_contacts(contact_view,row)
                pair_stream.write(json.dumps(serial(pair_record),allow_nan=False)+"\n")
                contact_stream.write(json.dumps(serial(contacts),allow_nan=False)+"\n")
                pair_stream.flush();contact_stream.flush()
                try:
                    require_quiet(pair_record,contacts,row)
                except ValueError as error:
                    failure=failure or str(error)
                    row["failure"]=failure
                stream.write(json.dumps(serial(row),allow_nan=False)+"\n")
                stream.flush()
                rows.append(row)
                if failure: break
                previous=next_wrench
            os.fsync(stream.fileno());os.fsync(pair_stream.fileno());os.fsync(contact_stream.fileno())
        final=sim.current_time_step_index
        result={"scope":cfg["scope"],"side":side,"rod_uid":"rod0","native_execution_performed":True,
                "grasp_qualified":False,"full_state371_bound":False,"failure":failure,
                "physics_rows":len(rows),"wrench_opposition_candidate_observed":any(
                    r["r0_raw_measurements"]["wrench_opposition_candidate"] for r in rows),
                "physics_clock":{"sdk_initial_step":sdk_initial,"sdk_after_setup_step":sdk_setup,
                    "sdk_first_measurement_step":rows[0]["sdk_step"],"sdk_last_measurement_step":rows[-1]["sdk_step"],
                    "sdk_final_step":final,"setup_steps":sdk_setup-sdk_initial,"measured_steps":len(rows),
                    "total_steps":final-sdk_initial},
                "physics_budget":native_physics_budget(initialization_physics_steps=sdk_setup-sdk_initial,
                    warmup_physics_steps=0,measurement_physics_steps=len(rows),other_physics_advancing_steps=0),
                "code_config_sha256":contract["code_config_sha256"],"outer_exit_verified":False}
        if failure is None:
            result.update(validate_pinch_ledger(serial(rows)))
        write_json(args.output/"measurement.json",serial(result))
        return serial(result)
    finally:
        sim = bundle = hand = view = composer = runtime = provider = contact_view = host = None
        snapshot = mesh_poses = state = rod = before = after = value = None
        write_json(args.output/"measurement_frame_detached.json",dict(
            version="dh116-pinch-measurement-detachment-v2",clearing_completed=True,
            operation_raised=sys.exc_info()[0] is not None,native_release_verified=False))

def run_lifecycle(app, *, make_sim, acquire, synchronize, clear_sim, stop,
                  manager_is_none, freeze, output, cleanup):
    if cleanup != "explicit":
        raise ValueError("unsupported cleanup case")
    output = Path(output)
    trace = LifetimeTrace(output / "ownership.jsonl")
    sim = None
    sim_ref = None
    failure = []
    clocks = []
    acquired = {}
    released = False

    def observe(phase):
        target = sim_ref() if sim_ref is not None else None
        clocks.append({"phase": phase, "clock_available": target is not None,
                       **(clock(target) if target is not None else {})})
        del target
        trace.checkpoint(phase)

    try:
        sim = make_sim()
        sim_ref = weakref.ref(sim)
        trace.watch("simulation", sim, owner="lifecycle", release="clear then STOP", callbacks=True)
        acquired = run_owned_call(lambda owners: acquire(sim, owners, trace, cleanup),
                                  synchronize=synchronize, ownership=trace)
    except BaseException as error:
        failure.append(traceback.format_exc())
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    # The active exception must end before STOP and close; only text survives.
    try:
        if freeze() is not False:
            raise RuntimeError("Kit physics freeze did not read back false")
        if sim is not None:
            observe("before_stop")
            clear_sim(sim)
            sim = None
            gc.collect()
            stop()
            if freeze() is not False:
                raise RuntimeError("Kit physics resumed during STOP")
            app.update()
            observe("after_stop")
            trace.event("SDK STOP", manager_view_released=manager_is_none())
        observe("before_app_close")
        state = snapshot_weakrefs(trace.references)["resources"]
        released = all(state.get(name, {}).get("alive") is False for name in OWNED_NAMES)
    except BaseException as error:
        failure.append(traceback.format_exc())
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    try:
        if freeze() is not False:
            raise RuntimeError("Kit physics must remain frozen before close")
        write_json(output / "before_app_close.json",
                   {"case": cleanup, "clocks": clocks, "lifetime_released": released,
                    "play_simulations": False, "process_exit_verified": False})
        app.close()
        write_json(output / "app_close_returned.json",
                   {"case": cleanup, "app_close_returned": True, "process_exit_verified": False})
        observe("after_app_close")
    except BaseException as error:
        failure.append(traceback.format_exc())
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    if acquired.get("failure"):
        failure.append(str(acquired["failure"]))
    steps = [row["sdk_step"] for row in clocks if row.get("clock_available")]
    result = {"version": VERSION, "case": cleanup, "acquisition": acquired, "clocks": clocks,
              "physics_steps": max(steps) if steps else None,
              "measurement_steps": acquired.get("physics_rows"),
              "rod_steps": acquired.get("physics_rows"),
              "code_config_sha256": acquired.get("code_config_sha256"),
              "physics_clock": acquired.get("physics_clock"),"physics_budget": acquired.get("physics_budget"), "ppo_updates": 0, "play_simulations": False,
              "lifetime_released_before_close": released, "failure": "\n".join(failure) or None,
              "process_exit_verified": False, "grasp_qualified": False, "teacher_admission": False}
    write_json(output / "result.json", result)
    if failure:
        raise RuntimeError("\n".join(failure)) from None
    return result



def execute(args, contract, cfg, project):
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
    from isaaclab.app import AppLauncher
    with native_instance_lock():
        from ropesim import _ropesim
        from scripts.verify_dh116_pinch_candidate_v2 import sha
        loaded=Path(_ropesim.__file__).resolve(strict=True)
        expected=args.ropesim_python/"ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so"
        if loaded!=expected or sha(loaded)!=contract["sources"]["external_sources"][str(expected)]:
            raise RuntimeError("loaded pair solver differs from qualified frozen backend")
        app=AppLauncher(headless=True,device=args.device,enable_cameras=False,
            fast_shutdown=False,
            experience="/mnt/mydisk/xk/projects/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.kit").app
        faulthandler.enable(file=sys.__stderr__,all_threads=True)
        import carb.settings
        import torch
        import isaaclab.sim as sim_utils
        from isaacsim.core.simulation_manager import SimulationManager
        settings=carb.settings.get_settings()
        def freeze():
            settings.set_bool("/app/player/playSimulations",False)
            return settings.get("/app/player/playSimulations")
        def stop():
            import omni.timeline
            timeline=omni.timeline.get_timeline_interface()
            timeline.stop()
            del timeline
        run_lifecycle(
            SimpleNamespace(update=app.update,close=lambda:app.close(wait_for_replicator=False)),
            make_sim=lambda:sim_utils.SimulationContext(sim_utils.SimulationCfg(
                dt=DT,device=args.device,render_interval=1,
                gravity=tuple(cfg["rod"]["gravity_world_m_s2"]),
                physx=sim_utils.PhysxCfg(solver_type=0))),
            acquire=lambda sim,owners,trace,cleanup:measure_owned(sim,args,contract,cfg,owners,trace,project),
            synchronize=lambda:torch.cuda.synchronize(args.device),
            clear_sim=lambda sim:type(sim).clear_instance(),stop=stop,
            manager_is_none=lambda:SimulationManager.get_physics_sim_view() is None,
            freeze=freeze,output=args.output,cleanup=args.cleanup)
