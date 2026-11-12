#!/usr/bin/env python3
"""Bounded original-DH116 left Pinch candidate: free rod, gravity and finite table.

Default execution is a CPU-only byte/source preflight. Native execution requires
the exact current preflight code_config_sha256 as an explicit approval argument.
"""
import argparse
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid
from types import SimpleNamespace

import numpy as np

from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import (
    DT, FreeRodRuntime, candidate_action, contact_metrics, digest, integrate_pinch_action,
    load_candidate, material_mapping, validate_pinch_ledger, run_owned_call,
)
from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits, expand
from scripts.bake_dh116_deformx_pad_sdfs import checked_json, checked_bytes
from scripts.run_dh116_work_packages import native_physics_budget
from scripts.verify_dh116_cooked_convex_hulls_native import _sources
from scripts.verify_dh116_deformx_rigid_bridge import rotation_wxyz, wrench_at_com, write_json
from scripts.verify_dh116_deformx_thumb_contact import reset_thumb_physics, step_thumb_physics, _freeze_kit_physics

EVIDENCE = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence")
SDF_SHA = "db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab"


def default_arguments(project):
    return SimpleNamespace(
        candidate=project/"configs/dh116_tasks/flexible_pinch_candidate_v1.json",
        ropesim_python=EVIDENCE/"flexible-f1-20261112-883RS4/build-attempt3/python",
        backend_source=Path("/mnt/mydisk/xk/projects/deformx-f0-04f6a11/backend"),
        sdf_dir=EVIDENCE/"flex-f4-sdf-register-rK7gw0/bake",
        capture_dir=EVIDENCE/"flex-f4-parity-prep-Vu9H0f")


def preflight(args, project):
    cfg=load_candidate(args.candidate,project)
    lock=json.loads((project/"docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json").read_text())
    backend=json.loads((project/"docs/contracts/dh116_deformx_backend_lock.json").read_text())
    native=next(row for row in backend["repositories"] if row["path"]=="Cosserat-Rod-Solver-CUDA")
    source_files={}
    for name,sha in native["file_sha256"].items():
        checked_bytes(args.backend_source/name,sha)
        source_files[str(args.backend_source/name)]=sha
    for name,sha in lock["backend_files"].items():
        checked_bytes(args.ropesim_python/"ropesim"/name,sha)
        source_files[str(args.ropesim_python/"ropesim"/name)]=sha
    summary=checked_json(args.sdf_dir/"result.json",SDF_SHA)
    source_files[str(args.sdf_dir/"result.json")]=SDF_SHA
    sources=_sources(project)

    bindings={}
    captures={}
    for name in DISTAL_PAD_LINKS:
        key=f'{cfg["side"]}.{name}'
        row=summary["pads"][key]
        if row["rod_radius_m"] != cfg["rod"]["radius_m"] or not row["closed_welded_hull"]:
            raise ValueError("SDF radius/hull identity differs")
        sdfpath=args.sdf_dir/f"{key}.npz"
        cap=args.capture_dir/"gpu"/f'{cfg["side"]}_{name}.json'
        checked_bytes(sdfpath,row["sdf_file_sha256"])
        captures[name]=checked_json(cap,lock["gpu_rows"][key])
        if row["gpu_row_sha256"] != lock["gpu_rows"][key]:
            raise ValueError("SDF is not bound to the locked native pad")
        bindings[name]={"sdf_path":str(sdfpath),"sdf_sha256":row["sdf_file_sha256"],
                        "capture_sha256":lock["gpu_rows"][key],
                        "geometry_sha256":captures[name]["source"]["geometry_sha256"]}
        source_files[str(sdfpath)]=row["sdf_file_sha256"]
        source_files[str(cap)]=lock["gpu_rows"][key]
    local=dict(cfg["sources"])
    asset_relative=str(Path(sources[cfg["side"]]["path"]).relative_to(project))
    local[asset_relative]=sources[cfg["side"]]["sha256"]
    local["assets/fixed_dual_panda_dh116_v1/manifest.json"]=digest(project/"assets/fixed_dual_panda_dh116_v1/manifest.json")
    # Freeze the local transitive control/runtime sources, including this entry.
    prior=json.loads((project/"configs/dh116_execution/deformx_thumb_left_lifecycle_v2.json").read_text())
    extras = [project/name for name in prior["source_files"]]
    extras += [project/"src/go2_pvcnn/control/dh116"/name for name in (
        "tabletop_deformable_grasp_runtime.py", "tabletop_material.py",
        "tabletop_stereo_rod_state.py", "tabletop_stereo_contract.py")]
    extras += [project/"scripts"/name for name in (
        "verify_dh116_deformx_pinch_contact.py", "run_dh116_pinch_work_packages.py",
        "verify_dh116_deformx_thumb_contact.py",
        "verify_dh116_deformx_rigid_bridge.py", "bake_dh116_deformx_pad_sdfs.py",
        "verify_dh116_cooked_convex_hulls_native.py", "run_dh116_work_packages.py")]
    for path in extras:
        local[str(path.relative_to(project))]=digest(path)
    local[str(args.candidate.resolve().relative_to(project.resolve()))]=digest(args.candidate)
    payload={"local_sources":local,"external_sources":source_files}
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    report={"version":cfg["version"],"scope":cfg["scope"],"code_config_sha256":fingerprint,
            "sources":payload,"candidate_sha256":digest(args.candidate),"side":cfg["side"],
            "source_usd":sources[cfg["side"]],"pad_bindings":bindings,
            "material_mapping":material_mapping(cfg),"assumptions":cfg["assumptions"],
            "physics_budget":native_physics_budget(initialization_physics_steps=2,
                warmup_physics_steps=0,measurement_physics_steps=238,other_physics_advancing_steps=0),
            "native_execution_performed":False,"grasp_qualified":False}
    return report,cfg


def serial(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(v) for v in value]
    return value


def _run_impl(sim,args,contract,cfg,ownership,owners):
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
    owners.keep("hand",hand)
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
    sdk_initial,sdk_setup=reset_thumb_physics(sim)
    hand.reset()
    view=owners.keep("view",hand.root_physx_view)
    composer=owners.keep("composer",hand.permanent_wrench_composer)
    ownership.watch("hand",hand,owner="candidate",release="asset STOP",callbacks=True)
    ownership.watch("hand_view",view,owner="hand",release="asset STOP",gpu_view=True)
    ownership.watch("composer",composer,owner="hand",release="exclusive reset",composer=True)
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
    owners.keep("runtime",runtime)
    ownership.watch("rope",runtime.rope,owner="runtime",release="drop owned solver",gpu_view=True)
    reset=runtime.reset(physics_step=sdk_setup,episode_uid="pinch-left-"+uuid.uuid4().hex,reset_id=1,pad_poses=mesh_poses(snapshot()))
    np.testing.assert_allclose(reset["node_positions"][0],layout["wire_points"],atol=2e-7,rtol=0)
    if np.max(np.abs(reset["velocities"])) > 1e-7:
        raise RuntimeError("rod reset velocity did not clear")
    provider=DeformableRodStateProvider(runtime,backend="DeformX",
        backend_artifact=args.ropesim_python/"ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so",
        backend_sha256=contract["sources"]["external_sources"][str(args.ropesim_python/"ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so")])
    owners.keep("provider",provider)
    write_json(args.output/"plan.json",serial({**contract,"native_execution_performed":True,
        "actual_pad_bindings":binding,"actuator_readback":sdk_properties,"hand_material":material_report,
        "rod_parameter_readback":runtime.parameter_readback,"rod_mass_readback":runtime.mass_readback,
        "rod_setter_configuration":cfg["rod"],"reset_readback":reset,
        "table_mesh_index":5,"state371_policy_binding":False,"palm_transport":"fixed_base_candidate"}))
    previous=np.zeros((5,6)); target=start.copy(); rows=[]; failure=None
    limits=np.stack(active_limits(),axis=-1)
    with (args.output/"physics.jsonl").open("x",encoding="utf-8") as stream:
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
            stream.write(json.dumps(serial(row),allow_nan=False)+"\n")
            stream.flush()
            rows.append(row)
            if failure: break
            previous=next_wrench
        os.fsync(stream.fileno())
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
    return result


def _run(sim,args,contract,cfg,ownership):
    import torch
    return run_owned_call(lambda owners:_run_impl(sim,args,contract,cfg,ownership,owners),
                          synchronize=lambda:torch.cuda.synchronize(args.device),ownership=ownership)


def run_app(app,args,contract,cfg,ownership):
    import carb.settings
    import isaaclab.sim as sim_utils
    from isaacsim.core.simulation_manager import SimulationManager
    from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import close_calibration_app
    settings=carb.settings.get_settings()
    sim=result=None
    failure_text=None
    try:
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT,device=args.device,
              render_interval=1,gravity=tuple(cfg["rod"]["gravity_world_m_s2"]),
              physx=sim_utils.PhysxCfg(solver_type=0)))
        ownership.watch("simulation",sim,owner="candidate",release="clear then STOP",callbacks=True)
        result=_run(sim,args,contract,cfg,ownership)
        write_json(args.output/"result.json",result)
        if result["failure"]: raise RuntimeError(result["failure"])
    except BaseException as error:
        failure_text=traceback.format_exc()
        traceback.clear_frames(error.__traceback__)
        error.__traceback__=None
        write_json(args.output/"failure.json",{"traceback":failure_text})
        # Leave this except block without rethrowing: active exception frames
        # must not retain SimulationContext while STOP/App close runs.
    finally:
        try:
            _freeze_kit_physics(settings)
            if sim is not None:
                import omni.timeline
                final=sim.current_time_step_index
                clock=result["physics_clock"] if result else None
                matches=clock is None or final==clock["sdk_final_step"]==clock["sdk_last_measurement_step"]
                ownership.event("SDK shutdown clock frozen",sdk_final_step=final,
                     last_measured_sdk_step=clock["sdk_last_measurement_step"] if clock else None,
                     matches_measurement=matches if clock else None,play_simulations=False)
                type(sim).clear_instance()
                sim=None
                gc.collect()
                timeline=omni.timeline.get_timeline_interface()
                timeline.stop()
                _freeze_kit_physics(settings)
                app.update()
                del timeline
                released=SimulationManager.get_physics_sim_view() is None
                ownership.event("SDK STOP",manager_view_released=released)
                if not released: raise RuntimeError("SimulationManager view survived STOP")
                if not matches: raise RuntimeError("SDK clock advanced after last measurement")
        finally:
            _freeze_kit_physics(settings)
            close_calibration_app(app,args.output)
            ownership.checkpoint("app.close returned, outer exit pending")
    if failure_text is not None:
        raise RuntimeError(failure_text) from None


def validate_pinch_artifacts(output, outer_exit_path, expected_code_config_sha256):
    """CPU-only independent re-open after the native process has exited."""
    from scripts.run_dh116_work_packages import (
        _validate_native_physics_shutdown, validate_native_physics_clock, debugger_status)
    output,outer_exit_path=Path(output).resolve(),Path(outer_exit_path).resolve()
    outer=json.loads(outer_exit_path.read_text())
    if (type(outer.get("outer_exit_code")) is not int or outer["outer_exit_code"]!=0
            or outer.get("native_lock_observed") is not True):
        raise ValueError("independent outer exit 0 and observed native lock required")
    log=outer_exit_path.parent/"native.log"
    if digest(log)!=outer.get("log_sha256"):
        raise ValueError("outer process log SHA differs")
    debug=debugger_status(log)
    if (debug.get("debugger_status_verified") is not True or debug.get("debugger_inferior_exit_code")!=0
            or debug.get("debugger_inferior_signal") is not None):
        raise ValueError("natural debugger inferior exit required")
    files=("plan.json","physics.jsonl","result.json","ownership.jsonl","shutdown_complete.json")
    artifact_sha={name:digest(output/name) for name in files}
    plan=json.loads((output/"plan.json").read_text())
    result=json.loads((output/"result.json").read_text())
    if any(record.get("code_config_sha256")!=expected_code_config_sha256 for record in (plan,result)):
        raise ValueError("candidate code/config identity differs")
    with (output/"physics.jsonl").open() as stream:
        rows=[]
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("candidate ledger has an incomplete row")
            rows.append(json.loads(line))
    evidence=validate_pinch_ledger(rows)
    clock=validate_native_physics_clock(result,[(r["phase_step"],r["sdk_step"]) for r in rows],
                                        setup_steps=2,total_steps=240)
    _validate_native_physics_shutdown(output,{"close_marker":"shutdown_complete.json",
                                             "ownership_ledger":"ownership.jsonl"},clock)
    budget=native_physics_budget(initialization_physics_steps=2,warmup_physics_steps=0,
                                 measurement_physics_steps=238,other_physics_advancing_steps=0)
    if result.get("physics_budget")!=budget or plan.get("physics_budget")!=budget:
        raise ValueError("candidate exact measured phase budget differs")
    reset=plan["reset_readback"]
    if (reset["physics_step"],reset["rod_tick"],reset["reset_id"],reset["episode_uid"]) != (
            2,0,1,rows[0]["episode_uid"]) or reset["contact_valid"] is not False:
        raise ValueError("candidate reset is not bound to the same fresh episode")
    mass=plan["rod_mass_readback"]
    if mass["pinned_node_ids"]!=[] or len(mass["node_mass_kg"])!=41 or any(v<=0 for v in mass["node_mass_kg"]):
        raise ValueError("candidate endpoints are not free")
    if (result.get("grasp_qualified") is not False or result.get("full_state371_bound") is not False
            or result.get("failure") is not None):
        raise ValueError("candidate cannot claim grasp or policy qualification")
    return {**evidence,"artifact_sha256":artifact_sha,"physics_clock":clock,"physics_budget":budget,
            "outer_exit_verified":True,"native_scope":"free_pinch_contact_candidate_only",
            "code_config_sha256":expected_code_config_sha256,"teacher_admission":False}


def require_pinch_worker(args,contract,project):
    """Reject generic-scheduler/direct native launches before importing the SDK."""
    expected={"version":"dh116-pinch-worker-v1",
              "wrapper_sha256":digest(project/"scripts/run_dh116_pinch_work_packages.py"),
              "candidate_code_config_sha256":contract["code_config_sha256"],
              "output":str(args.output.resolve())}
    try:
        actual=json.loads(os.environ.get("DH116_PINCH_WORKER_V1",""))
    except (TypeError,ValueError):
        actual=None
    if actual!=expected:
        raise ValueError("native execution requires the dedicated Pinch worker with current wrapper/candidate/output identity")


def main():
    project=Path(__file__).resolve().parents[1]
    defaults=default_arguments(project)
    parser=argparse.ArgumentParser(description=__doc__)
    for name,value in vars(defaults).items():
        parser.add_argument("--"+name.replace("_","-"),type=Path,default=value)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--validate-output",type=Path)
    parser.add_argument("--outer-exit",type=Path)
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--approved-code-config-sha256")
    parser.add_argument("--steps",type=int,default=240)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--headless",action="store_true")
    args=parser.parse_args()
    contract,cfg=preflight(args,project)
    if args.steps != 240:
        parser.error("candidate requires exactly 240 total SDK ticks")
    if args.validate_output is not None:
        if args.execute_native or args.outer_exit is None:
            parser.error("disk validation requires --outer-exit and cannot launch native")
        print(json.dumps(validate_pinch_artifacts(args.validate_output,args.outer_exit,
                                                 contract["code_config_sha256"]),indent=2))
        return
    if not args.execute_native:
        print(json.dumps(contract,indent=2))
        return
    if args.approved_code_config_sha256 != contract["code_config_sha256"]:
        parser.error("native candidate needs explicit approval of current code_config_sha256")
    if args.output is None or not args.headless or args.device != "cuda:0":
        parser.error("native candidate requires fresh output, --headless and cuda:0")
    require_pinch_worker(args,contract,project)
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.ropesim_python))
    from isaaclab.app import AppLauncher
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
    from go2_pvcnn.control.dh116.tabletop_deformable_rod_bridge import NativeOwnershipTrace
    ownership=NativeOwnershipTrace(args.output/"ownership.jsonl")
    with native_instance_lock():
        app=AppLauncher(headless=True,device=args.device,enable_cameras=False,fast_shutdown=False).app
        faulthandler.enable(file=sys.__stderr__,all_threads=True)
        run_app(app,args,contract,cfg,ownership)


if __name__=="__main__": main()
