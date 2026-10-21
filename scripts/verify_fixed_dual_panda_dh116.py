"""Mounted diagnostics or explicitly gated continuous grasp, synthetic gains only."""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import asdict
from time import perf_counter
from go2_pvcnn.control.dh116.render_schedule import should_publish_wire
from pathlib import Path
from go2_pvcnn.control.dh116.evidence_json import dumps as evidence_json
from go2_pvcnn.control.dh116.evidence_journal import EvidenceJournal
from go2_pvcnn.control.dh116.run_evidence import write_start_manifest,check_stop_request
from go2_pvcnn.control.dh116.grasp_sequence import TabletopGraspSequence,validate_continuous_options
from go2_pvcnn.control.dh116.stage_state import load_state,save_state,PHASES
from go2_pvcnn.control.dh116.stage_runtime import StageResetSequence,robot_restore_arrays
from go2_pvcnn.control.dh116.touch_first import bounded_handoff
from go2_pvcnn.control.dh116.speed_profile import GraspSpeedProfile
from go2_pvcnn.control.dh116.perf_meter import PerfMeter
from go2_pvcnn.control.dh116.runtime_snapshot import SnapshotStore,ReactionBuffer
from go2_pvcnn.control.dh116.approach_wire_mode import wire_dynamics_enabled
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--asset-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--point-references',type=Path,help='Optional calibrated source points for COM Jacobian comparison')
parser.add_argument('--active-pose',type=Path,help='Pinned static initial pose diagnostic; no trajectory or grasp claim')
parser.add_argument('--active-pose-sha256')
parser.add_argument('--approach-certificate',type=Path,help='Pinned source-swept prepose-to-open-grasp approach; no closure')
parser.add_argument('--approach-certificate-sha256')
parser.add_argument('--closure-pilot',action='store_true',help='Independent initial-touch force-control diagnostic, NOT continuous approach/grasp')
parser.add_argument('--grasp-sequence',action='store_true',help='Measured settle/approach/close/hold/lift sequence; all physical guards mandatory')
parser.add_argument('--approach-wire-mode',choices=('dynamic','frozen'),default='dynamic',
                    help='Wire dynamics during approach; frozen keeps settled shape until close, then resumes DeformX')
parser.add_argument('--symmetric-lift',action='store_true',help='Stage-local measured height synchronization; not a global mirror constraint')
parser.add_argument('--grasp-speed-profile',choices=('baseline','staged'),default='baseline',
                    help='Opt-in experimental measured-gap speed profile, not hardware rated or grasp qualified')
parser.add_argument('--restore-state',type=Path,help='Pinned qualified stage initialization; NOT continuous-run certification')
parser.add_argument('--restore-state-sha256')
parser.add_argument('--save-stage-states',action='store_true',help='Publish exclusive states at measured qualified transitions')
parser.add_argument('--coordinated-closure',action='store_true',help='Compensate pad tangent velocity with arm motion while holding palm orientation')
parser.add_argument('--closure-loads',type=float,nargs=6,help='Explicit simulation normal-force targets: left thumb/index/middle, right thumb/index/middle')
parser.add_argument('--gravity-hold',action='store_true',help='Hold initial pose under gravity with current-model feedforward')
parser.add_argument('--arm-effort-limit',type=float,default=5.,help='Explicit synthetic simulation limit, not hardware specification')
parser.add_argument('--hand-effort-limit',type=float,default=.1,help='Explicit synthetic simulation limit, not DH116 hardware capability')
parser.add_argument('--hand-stiffness',type=float,default=1.,help='Explicit simulation-only position gain')
parser.add_argument('--hand-damping',type=float,default=.05,help='Explicit simulation-only damping gain')
parser.add_argument('--external-force-iterations',action='store_true',help='Integrate external forces in each TGS position iteration')
parser.add_argument('--solver-position-iterations',type=int,choices=range(1,256),default=16,
                    help='Explicit diagnostic position iteration count; default unchanged, not a convergence guarantee')
parser.add_argument('--dynamics-diagnostics',action='store_true',help='Record measured Jacobians, mass matrices and actuation readback')
parser.add_argument('--motion-prediction-order',type=int,choices=[1,2],default=2)
parser.add_argument('--steps',type=int,default=1920)
parser.add_argument('--render-every',type=int,default=10,
                    help='Render every N physics steps; 1 restores baseline, physics and guards remain per step')
parser.add_argument('--physics-dt',type=float,default=1/240,help='Explicit co-simulation coupling interval in seconds')
parser.add_argument('--table-wire',action='store_true',help='Free DeformX rod settling only; no hand-wire contacts yet')
parser.add_argument('--backend-root',type=Path,default=Path(__file__).resolve().parents[1]/'vendor/DeformX')
parser.add_argument('--contact-manifest',type=Path)
parser.add_argument('--contact-manifest-sha256')
parser.add_argument('--contact-subsamples',type=int,choices=range(1,129),default=1,
                    help='Weighted contact quadrature per rod element; not extra rod DOFs')
parser.add_argument('--clearance-certificate-cache',action='store_true',
                    help='Conservative source-clearance reuse with actual displacement debit; pad checks stay per step')
parser.add_argument('--contact-broad-phase',action='store_true',
                    help='Conservative contact AABB culling; exact contact observations and safety guards unchanged')
parser.add_argument('--contact-tangential-damping',type=float,default=10.,
                    help='Explicit simulation-only viscous Coulomb regularization; not calibrated static friction')
parser.add_argument('--probe-channel',choices=['left_thumb','left_index','right_thumb','right_index'],
                    help='Diagnostic initial 0.1mm pad overlap; not an approach or grasp')
parser.add_argument('--probe-overlap-m',type=float,default=.0001,help='Explicit diagnostic overlap, not grip success threshold')
parser.add_argument('--disable-contact-reaction',action='store_true',help='Diagnostic uncoupled baseline only')
parser.add_argument('--profile',action='store_true',help='Write phase-aware wall-clock perf.jsonl and perf_summary.json')
parser.add_argument('--perf-flush-steps',type=int,default=1000,
                    help='Flush performance rows after this many steps when profiling')
parser.add_argument('--ui-hz',type=float,default=15.,
                    help='Wall-clock wire GUI publication rate; physics still steps every iteration')
parser.add_argument('--evidence-flush-policy',choices=('audit','buffered'),default='audit',
                    help='Evidence durability policy; audit flushes each row')
parser.add_argument('--evidence-flush-steps',type=int,default=512,
                    help='Buffered evidence flush count')
parser.add_argument('--evidence-flush-seconds',type=float,default=.5,
                    help='Buffered evidence flush wall-clock interval')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if not math.isfinite(args.ui_hz) or args.ui_hz<=0:
    raise ValueError('ui hz must be finite and positive')
if args.evidence_flush_steps<1 or not math.isfinite(args.evidence_flush_seconds) or args.evidence_flush_seconds<=0:
    raise ValueError('evidence flush settings must be positive')
validate_continuous_options(args)
if args.grasp_speed_profile!='baseline' and not args.grasp_sequence:
    raise ValueError('staged speed requires the guarded grasp sequence')
if args.approach_wire_mode=='frozen' and not args.grasp_sequence:
    raise ValueError('frozen approach wire mode requires the guarded grasp sequence')
speed_profile=None if args.grasp_speed_profile=='baseline' else GraspSpeedProfile()
if args.symmetric_lift and not args.grasp_sequence:
    raise ValueError('symmetric lift requires the measured grasp sequence')
if bool(args.restore_state)!=bool(args.restore_state_sha256):
    raise ValueError('restore state requires an external SHA')
if (args.restore_state or args.save_stage_states) and not args.grasp_sequence:
    raise ValueError('stage state options require guarded grasp sequence')
restore_payload=None
grasp_mode=args.closure_pilot or args.grasp_sequence
closure_active=args.closure_pilot
grasp_sequence=None
if args.steps<1:raise ValueError('steps must be positive')
if not math.isfinite(args.contact_tangential_damping) or args.contact_tangential_damping<0:
    raise ValueError('contact tangential damping must be finite and nonnegative')
if not 0<args.physics_dt<=.01:raise ValueError('physics dt must be in (0,.01] seconds')
if any(not math.isfinite(x) or x<=0 for x in (args.hand_stiffness,args.hand_damping,args.hand_effort_limit)):
    raise ValueError('hand gains and effort limit must be positive finite')
if args.table_wire and not args.gravity_hold:raise ValueError('table wire requires gravity hold')
if bool(args.active_pose)!=bool(args.active_pose_sha256):raise ValueError('active pose requires external SHA')
if args.active_pose and (not args.gravity_hold or not args.table_wire or args.probe_channel):
    raise ValueError('active pose is a stationary tabletop gravity diagnostic, not a contact-placement probe')
if bool(args.approach_certificate)!=bool(args.approach_certificate_sha256):
    raise ValueError('approach certificate requires external SHA')
if args.approach_certificate and (not args.active_pose or not args.contact_manifest or not args.point_references
        or args.disable_contact_reaction or args.steps*args.physics_dt<4.):
    raise ValueError('approach requires pinned initial pose, contacts, point references, reactions and at least 4s')
if args.closure_pilot and (not args.active_pose or not args.contact_manifest or not args.point_references
        or args.approach_certificate or args.disable_contact_reaction or args.physics_dt>.0001
        or args.closure_loads is None or any(not math.isfinite(x) or not 0<x<=3 for x in args.closure_loads)):
    raise ValueError('closure pilot requires pinned pose, references, contacts, reactions, dt<=100us and explicit loads in (0,3] N')
if args.closure_loads is not None and not grasp_mode:raise ValueError('closure loads require grasp mode')
if args.coordinated_closure and not grasp_mode:raise ValueError('coordinated closure requires grasp mode')
if args.contact_manifest and (not args.table_wire or not args.contact_manifest_sha256):
    raise ValueError('contact requires tabletop mode and external manifest SHA')
if (args.probe_channel or args.disable_contact_reaction) and not args.contact_manifest:
    raise ValueError('contact probe/baseline requires contact manifest')
if args.output.exists(): raise ValueError('refusing to overwrite evidence')
args.output.mkdir(parents=True)
app=AppLauncher(args).app
result=dict(passed=False,physics_stepped=False,hardware_calibrated=False)
perf_meter=PerfMeter(args.output/'perf.jsonl',args.physics_dt,args.perf_flush_steps) if args.profile else None
try:
    import numpy as np
    import torch
    import isaaclab.sim as su
    from isaaclab.assets import Articulation,ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
    from go2_pvcnn.control.dh116.no_load import reference,metrics,contact_metrics
    from go2_pvcnn.control.dh116.usd_kinematics import SideKinematics
    from go2_pvcnn.control.dh116.point_probe import compare_point
    from go2_pvcnn.control.dh116.gravity import compensated_efforts,drive_headroom
    from pxr import Usd
    torch.set_num_threads(1)
    asset=args.asset_dir/'fixed_dual_panda_dh116.usd'
    manifest=json.loads((args.asset_dir/'manifest.json').read_text())
    if hashlib.sha256(asset.read_bytes()).hexdigest()!=manifest['asset_sha256']:
        raise ValueError('asset identity mismatch')
    for src in manifest['sources']:
        if hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()!=src['sha256']:
            raise ValueError('source identity mismatch')
    point_refs=None; point_rows=[]
    if args.point_references:
        point_refs=json.loads(args.point_references.read_text())
        if point_refs['asset_sha256']!=manifest['asset_sha256']:raise ValueError('point asset mismatch')
        for path,sha in point_refs['layers'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('point layer mismatch')
        model_stage=Usd.Stage.Open(str(asset))
        models={side:SideKinematics(model_stage,side) for side in ('left','right')}
    dt=args.physics_dt
    initial_actives=None
    approach=None
    if args.active_pose:
        from go2_pvcnn.control.dh116.collision_scene import SourceCollisionScene
        from go2_pvcnn.control.dh116.pose_artifact import load_active_pose
        source_scene=SourceCollisionScene(Usd.Stage.Open(str(asset)),cache_certificates=args.clearance_certificate_cache)
        initial_actives=load_active_pose(args.active_pose,args.active_pose_sha256,
            manifest['asset_sha256'],source_scene.layer_pins)
        initial_wire=np.c_[np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.308 if args.closure_pilot else .318)]
        result['initial_source_screen']=source_scene.check(initial_actives,initial_wire,permit_pad_contact=args.closure_pilot)
        if not result['initial_source_screen']['source_clearance_passed']:
            raise ValueError('initial pose rejected by full source geometry screen')
        if args.approach_certificate:
            from go2_pvcnn.control.dh116.approach import load_approach,approach_target
            approach=load_approach(args.approach_certificate,args.approach_certificate_sha256,source_scene.layer_pins)
            if any(not np.allclose(approach[0][s],initial_actives[s],atol=1e-12,rtol=0) for s in initial_actives):
                raise ValueError('approach start does not match pinned initial pose')
    result['run_config']=dict(dt=dt,steps=args.steps,gravity=args.gravity_hold,
        arm_effort_limit_nm=args.arm_effort_limit,hand_effort_limit_nm=args.hand_effort_limit,
        probe_channel=args.probe_channel,probe_overlap_m=args.probe_overlap_m,
        hand_stiffness=args.hand_stiffness,hand_damping=args.hand_damping,
        external_force_iterations=args.external_force_iterations,
        solver_position_iterations=args.solver_position_iterations,solver_velocity_iterations=4,
        motion_prediction_order=args.motion_prediction_order,
        contact_manifest_sha256=args.contact_manifest_sha256,
        active_pose_sha256=args.active_pose_sha256,
        approach_certificate_sha256=args.approach_certificate_sha256,
        closure_pilot=args.closure_pilot,grasp_sequence=args.grasp_sequence,closure_loads_n=args.closure_loads,
        approach_wire_mode=args.approach_wire_mode,
        coordinated_closure=args.coordinated_closure,
        speed_profile=None if speed_profile is None else asdict(speed_profile),
        grasp_policy=('touch-first-symmetric-lift-v1' if args.symmetric_lift else
                      'touch-first-v1' if args.grasp_sequence else 'diagnostic'),
        contact_subsamples=args.contact_subsamples,
        contact_broad_phase=args.contact_broad_phase,
        clearance_certificate_cache=args.clearance_certificate_cache,
        contact_tangential_damping=args.contact_tangential_damping,
        diagnostic_initial_touch_not_continuous_grasp=args.closure_pilot,
        profile=args.profile,perf_flush_steps=args.perf_flush_steps,
        evidence_flush_policy=args.evidence_flush_policy,
        evidence_flush_steps=args.evidence_flush_steps,
        evidence_flush_seconds=args.evidence_flush_seconds,
        source_commit=subprocess.run(
            ['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],
            check=False,capture_output=True,text=True).stdout.strip() or 'unknown',
        device=args.device,physics_dt=dt,wire_py_dt=1e-5,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        numeric_threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')},
        platform=platform.platform(),python_version=platform.python_version(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_start_manifest(args.output,result['run_config'],manifest)
    sim=su.SimulationContext(su.SimulationCfg(dt=dt,device=args.device,
        physx=su.PhysxCfg(enable_external_forces_every_iteration=args.external_force_iterations)))
    if not args.headless:
        sim.set_camera_view(eye=(1.35, -1.35, 1.05), target=(.36, 0., .32))
        light=su.DomeLightCfg(intensity=2000.)
        light.func('/World/DemoLight',light)
    engine=None
    if args.table_wire:
        import sys
        sys.path.insert(0,str(args.backend_root.resolve(strict=True)))
        from co_sim.models import CoSimConfig
        from co_sim.table_rod import TableRodEngine
        from go2_pvcnn.control.dh116.wire_table import TABLE_CENTER,TABLE_SIZE,settled_metrics
        from go2_pvcnn.control.fixed_dual_panda_o6.wire_visual import rod_surface
        from pxr import UsdGeom,Gf,Vt
        import omni.usd
        wire_config=CoSimConfig(n_elem=40,base_length=.5,base_radius=.008,
            density=1000.,youngs_modulus=1e6,shear_modulus_ratio=1.5,py_dt=1e-5,
            isaac_dt=dt,damping_constant=.01,rod_start=np.array([.43,-.25,.308 if args.closure_pilot else .318]),
            rod_direction=np.array([0.,1.,0.]),rod_normal=np.array([0.,0.,1.]))
        if args.contact_manifest:
            from co_sim.moving_contact import MovingContactRodEngine,MeshAsset
            from go2_pvcnn.control.dh116.contact_runtime import BODY_NAMES,load_mesh_manifest,mesh_motions,probe_rod_start
            from go2_pvcnn.control.fixed_dual_panda_o6.contact_bridge import (
                BodyState,ContactReactionBridge,normalized_wxyz_to_rotation)
            from go2_pvcnn.control.dh116.prepublish import prepublish_contact_gates
            contact_records=load_mesh_manifest(args.contact_manifest,args.contact_manifest_sha256)
            contact_assets={c:MeshAsset(r['resolved_path'],r['sha256']) for c,r in contact_records.items()}
            engine=MovingContactRodEngine(wire_config,table_center=TABLE_CENTER,table_size=TABLE_SIZE,meshes=contact_assets,
                contact_subsamples=args.contact_subsamples,contact_broad_phase=args.contact_broad_phase,
                velocity_damping_coefficient=args.contact_tangential_damping)
            result['backend_pins']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.backend_root/'co_sim/moving_contact.py',
                          args.backend_root/'co_sim/moving_contact_models.py',args.backend_root/'co_sim/free_rod.py',
                          args.backend_root/'co_sim/table_rod.py',args.backend_root/'elastica/contact_forces.py',
                          args.backend_root/'elastica/_contact_functions.py',
                          args.backend_root/'elastica/rigidbody/mesh_rigid_body.py')}
            result['wire_physics']=dict(length_m=.5,radius_m=.008,elements=40,density_kg_m3=1000.,
                youngs_modulus_pa=1e6,shear_modulus_ratio=1.5,py_dt_s=1e-5,damping_constant=.01,
                contact_stiffness_n_m=engine.contact_k,contact_damping=engine.contact_nu,
                friction_coefficient=engine.friction_coefficient,
                velocity_damping_coefficient=engine.velocity_damping_coefficient,hardware_calibrated=False)
            result['wire_physics']['contact_subsamples']=engine.contact_subsamples
            result['wire_physics']['mesh_signed_distance_methods']={
                c:body.signed_distance_method for c,body in engine.mesh_bodies.items()}
            result['dh116_module_pins']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__).resolve().parents[1]/'src/go2_pvcnn/control/dh116').glob('*.py')}
            if grasp_mode:
                import open3d as o3d
                from go2_pvcnn.control.dh116.pad_region_guard import PadRegionGuard
                from go2_pvcnn.control.dh116.closure import closure_increment
                from go2_pvcnn.control.dh116.coordinated_closure import (
                    coordinated_closure_increment, coordinated_bimanual_increment,
                    orient_normals_toward_wire, wire_approach_normals)
                from go2_pvcnn.control.dh116.contact_frames import point_jacobian
                from go2_pvcnn.control.dh116.loaded_contact import normal_contact_sample
                if point_refs.get('schema') not in ('dh116-grasp-references-v1','dh116-grasp-references-v2-middle-hook'):
                    raise ValueError('pilot requires selected grasp references')
                selected_refs={r['side']+'_'+r['digit']:r for r in point_refs['references']}
                if tuple(selected_refs)!=tuple(BODY_NAMES):raise ValueError('pilot requires exactly six ordered grasp references')
                pad_guards={}
                for c,r in contact_records.items():
                    mesh_data=o3d.io.read_triangle_mesh(str(r['resolved_path']))
                    ref=selected_refs[c]
                    if ref['body_name']!=BODY_NAMES[c]:raise ValueError('selected grasp body mismatch')
                    pad_guards[c]=PadRegionGuard(np.asarray(mesh_data.vertices),np.asarray(mesh_data.triangles),
                        np.array(ref['point'])-np.array(r['body_local_mesh_com_m']),ref['normal'])
                result['pad_regions']={c:dict(front_face_ids=g.front_face_ids,total_faces=g.total_faces,
                    front_cap_method=g.front_cap_method,partially_clipped_face_ids=g.partially_clipped_face_ids,
                    partition_sha256=g.partition_sha256,
                    geometry_sha256=g.geometry_sha256) for c,g in pad_guards.items()}
        else:engine=TableRodEngine(wire_config,table_center=TABLE_CENTER,table_size=TABLE_SIZE,
            contact_subsamples=args.contact_subsamples,contact_broad_phase=args.contact_broad_phase,
            velocity_damping_coefficient=args.contact_tangential_damping)
        table=su.CuboidCfg(size=TABLE_SIZE,collision_props=su.CollisionPropertiesCfg())
        table.func('/World/Table',table,translation=TABLE_CENTER)
        mesh=UsdGeom.Mesh.Define(omni.usd.get_context().get_stage(),'/World/CosseratWire')
        mesh.CreateSubdivisionSchemeAttr().Set('none')
        mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(1.,.2,.035)])
        # Visual only: DeformX owns rod/table contacts. No duplicate rigid wire.
        from go2_pvcnn.control.dh116.wire_mesh_publisher import WireMeshPublisher
        publish_wire=WireMeshPublisher(mesh,.008)
        if args.grasp_sequence:
            def digest_json(value):
                return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
            state_pins=dict(asset=manifest['asset_sha256'],
                config=digest_json({k:v for k,v in result['run_config'].items() if k!='steps'}),
                references=hashlib.sha256(args.point_references.read_bytes()).hexdigest(),
                backend=digest_json(result['backend_pins']),modules=digest_json(result['dh116_module_pins']))
            if args.restore_state:
                restore_payload=load_state(args.restore_state,args.restore_state_sha256,state_pins)
                if restore_payload['target_loads_n']!=args.closure_loads:
                    raise ValueError('restored force targets mismatch')
                engine.restore_physical_state(restore_payload['rod'])
                initial_actives={s:np.array([r['q'][r['joint_names'].index(n)] for n in
                    [f'{s}_panda_joint{i}' for i in range(1,8)]+list(ACTIVE)]) for s,r in restore_payload['robot'].items()}
                closure_active=restore_payload['entry_stage'] in ('hold','lift')
                result['stage_reset']=dict(source=str(args.restore_state),sha256=args.restore_state_sha256,
                    entry_stage=restore_payload['entry_stage'],imported_evidence=restore_payload['source'],
                    continuous_grasp_qualified=False,local_stages_passed=False)
        publish_wire(engine.snapshot())
    spawn=su.UsdFileCfg(usd_path=str(asset),activate_contact_sensors=True,
        rigid_props=su.RigidBodyPropertiesCfg(disable_gravity=not args.gravity_hold),
        articulation_props=su.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
            solver_position_iteration_count=args.solver_position_iterations,solver_velocity_iteration_count=4))
    spawn.func('/World/Rig',spawn)
    arms={}; sensors={}
    home=[0.,-.6,0.,-2.2,0.,1.8,.7]
    for side in ('left','right'):
        arm_names=[f'{side}_panda_joint{i}' for i in range(1,8)]
        initial_arm=home if initial_actives is None else initial_actives[side][:7]
        initial_hand=reference(0) if initial_actives is None else initial_actives[side][7:]
        init=dict(zip(arm_names,map(float,initial_arm))); init.update(zip(JOINTS,map(float,expand(initial_hand))))
        cfg=ArticulationCfg(prim_path=f'/World/Rig/{side}_arm',spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=init),
            actuators={
                'arm':ImplicitActuatorCfg(joint_names_expr=arm_names,stiffness=80.,damping=4.,effort_limit_sim=args.arm_effort_limit,velocity_limit_sim=.15),
                'hand':ImplicitActuatorCfg(joint_names_expr=list(ACTIVE),stiffness=args.hand_stiffness,damping=args.hand_damping,effort_limit_sim=args.hand_effort_limit,velocity_limit_sim=.3),
                'passive':ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE],stiffness=0.,damping=0.,effort_limit_sim=0.,velocity_limit_sim=.3)})
        arms[side]=Articulation(cfg)
        sensors[side]=[ContactSensor(ContactSensorCfg(prim_path=f'/World/Rig/{side}_arm/.*',update_period=0.)),
                       ContactSensor(ContactSensorCfg(prim_path=f'/World/Rig/{side}_arm/dh116/.*',update_period=0.))]
    sim.reset()
    for side, arm in arms.items():
        if len(arm.joint_names)!=18 or set(arm.joint_names)!=set(JOINTS)|{f'{side}_panda_joint{i}' for i in range(1,8)}:
            raise ValueError('mounted joint set mismatch')
        arm.write_joint_state_to_sim(arm.data.default_joint_pos.clone(),torch.zeros_like(arm.data.default_joint_vel))
        if restore_payload is not None:
            restored=robot_restore_arrays(restore_payload['robot'][side],arm.joint_names,
                arm.data.joint_pos_limits[0].cpu().numpy(),arm.data.root_state_w[0].cpu().numpy())
            arm.write_joint_state_to_sim(
                torch.tensor(restored['q'][None],dtype=torch.float32,device=args.device),
                torch.tensor(restored['qd'][None],dtype=torch.float32,device=args.device))
        arm.reset()
        coverage=[n for sensor in sensors[side] for n in sensor.body_names]
        if len(coverage)!=len(arm.body_names) or set(coverage)!=set(arm.body_names):
            raise ValueError('mounted sensor body coverage mismatch')
    # Joint teleports invalidate link transforms. Refresh both articulations
    # before gravity queries/physics, independent of lazy diagnostic pose reads.
    sim.physics_sim_view.update_articulations_kinematic()
    previous_states=None;contact_peak=0.
    approach_frozen_snapshot=None
    def measured_source_guard(wire_snapshot,wire_path_bound=0.,wire_radius=.008):
        from go2_pvcnn.control.fixed_dual_panda_o6.contact_bridge import normalized_wxyz_to_rotation
        active={};measured={}
        for side,arm in arms.items():
            names=[f'{side}_panda_joint{i}' for i in range(1,8)]+list(ACTIVE)
            active[side]=arm.data.joint_pos[0,[arm.joint_names.index(n) for n in names]].cpu().numpy()
            body_poses=arm.data.body_link_pose_w[0].cpu().numpy()
            for body,path in source_scene.models[side].bodies.items():
                if path not in source_scene.geometry:continue
                p=body_poses[arm.body_names.index(body)];t=np.eye(4)
                t[:3,:3]=normalized_wxyz_to_rotation(p[3:7]);t[:3,3]=p[:3];measured[path]=t
        delta={s:np.r_[np.full(7,.151*dt),np.full(11,.31*dt)] for s in arms} if grasp_mode else None
        check=source_scene.check(active,np.asarray(wire_snapshot['rod_position']).T,
                                measured_body_poses=measured,permit_pad_contact=closure_active,
                                physical_motion_delta=delta,wire_motion_bound=wire_path_bound,wire_radius=wire_radius)
        if closure_active:
            region_checks={}
            for c,g in pad_guards.items():
                side=c.split('_')[0];path=source_scene.models[side].bodies[BODY_NAMES[c]];pose=measured[path]
                motion=float(source_scene.geometry[path]['motion_radii']@delta[side])
                region_checks[c]=g.check(wire_snapshot['rod_position'].T,pose[:3,:3],
                    pose[:3,3]+pose[:3,:3]@np.array(contact_records[c]['body_local_mesh_com_m']),
                    wire_radius+wire_path_bound+motion)
            check['pad_regions']=region_checks
            check['source_clearance_passed']=bool(check['source_clearance_passed'] and all(r['passed'] for r in region_checks.values()))
        if not check['source_clearance_passed']:
            result['measured_source_guard_failure']=check
            result['measured_source_guard_active_q']={s:q.tolist() for s,q in active.items()}
            raise ValueError('actual body source-clearance guard rejected motion')
        return check
    if grasp_mode:
        result['initial_measured_source_guard']=measured_source_guard(engine.snapshot())
        closure_start=approach[1] if args.grasp_sequence else initial_actives
        closure_targets={s:closure_start[s][7:].copy() for s in arms}
        closure_arm_targets={s:closure_start[s][:7].copy() for s in arms}
        if restore_payload is not None:
            for s,r in restore_payload['robot'].items():
                closure_targets[s]=np.array([r['target'][r['joint_names'].index(n)] for n in ACTIVE])
                closure_arm_targets[s]=np.array([r['target'][r['joint_names'].index(f'{s}_panda_joint{i}')] for i in range(1,8)])
        previous_loads=np.zeros(6)
        from go2_pvcnn.control.dh116.material_tracking import material_in_palm
        fixed_material_ids=[]
        initial_nodes=engine.snapshot()['rod_position'].T
        for c,ref in selected_refs.items():
            arm=arms[ref['side']];bi=arm.body_names.index(ref['body_name'])
            pose=arm.data.body_link_pose_w[0,bi].cpu().numpy()
            point=pose[:3]+normalized_wxyz_to_rotation(pose[3:7])@np.array(ref['point'])
            if args.grasp_sequence:
                # Freeze IDs from the nominated grasp regions, never reassign
                # using a moving nearest contact during closure or transport.
                point=models[ref['side']].point(approach[1][ref['side']],ref['body_name'],ref['point'])
            fixed_material_ids.append(int(np.argmin(np.linalg.norm(initial_nodes-point,axis=1))))
        if restore_payload is not None:fixed_material_ids=restore_payload['material_ids'][:]
        result['fixed_material_ids']=dict(zip(BODY_NAMES,fixed_material_ids))
        material_baseline=None;contact_since=None;longest_contact_s=0.;max_material_slip_m=0.
        if args.grasp_sequence:
            grasp_sequence=TabletopGraspSequence(material_ids=fixed_material_ids,target_loads_n=args.closure_loads,
                                                state_triggered=True,symmetric_lift=args.symmetric_lift,speed_profile=speed_profile)
            if restore_payload is not None:grasp_sequence=StageResetSequence(restore_payload,state_pins,
                state_triggered=True,symmetric_lift=args.symmetric_lift,speed_profile=speed_profile)
            approach_measured=None
            approach_goals={}
            for c,ref in selected_refs.items():
                m=models[ref['side']];pose=m.fk(approach[1][ref['side']])[0][m.bodies[ref['body_name']]]
                approach_goals[c]=(pose[:3,3]+pose[:3,:3]@ref['point'],pose[:3,:3]@ref['normal'])
    if args.contact_manifest:
        body_indices={c:arms[c.split('_')[0]].body_names.index(n) for c,n in BODY_NAMES.items()}
        offsets={c:r['body_local_mesh_com_m'] for c,r in contact_records.items()}
        class IsaacReactionSink:
            def __init__(self):
                self.buffers={side:ReactionBuffer.allocate(args.device,len(arms[side].body_names))
                              for side in ('left','right')}
                self.environment_indices=torch.zeros((1,),device=args.device,dtype=torch.int32)

            def submit(self,side,force,torque,position):
                if args.disable_contact_reaction:return
                arm=arms[side]
                buffer=self.buffers[side].clear_()
                for i,digit in enumerate(('thumb','index','middle')):
                    bi=body_indices[f'{side}_{digit}']
                    buffer.force[0,bi].copy_(torch.as_tensor(force[i],device=args.device,dtype=torch.float32))
                    buffer.torque[0,bi].copy_(torch.as_tensor(torque[i],device=args.device,dtype=torch.float32))
                    buffer.position[0,bi].copy_(torch.as_tensor(position[i],device=args.device,dtype=torch.float32))
                arm.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=buffer.force,torque_data=buffer.torque,position_data=buffer.position,
                    indices=self.environment_indices,is_global=True)
        reaction_sink=IsaacReactionSink()
        bridge=ContactReactionBridge(reaction_sink,max_force_n=10.)
        if args.probe_channel:
            c=args.probe_channel;side=c.split('_')[0];arm=arms[side];bi=body_indices[c]
            pose=arm.data.body_link_pose_w[0,bi].cpu().numpy()
            R=normalized_wxyz_to_rotation(pose[3:7]);r=contact_records[c]
            point=pose[:3]+R@np.array(r['reference_point_body'])
            normal=R@np.array(r['reference_normal_body'])
            wire_config.rod_start,direction=probe_rod_start(point,normal,R[:,0],args.probe_overlap_m)
            wire_config.rod_direction=direction;wire_config.rod_normal=normal
            engine=MovingContactRodEngine(wire_config,table_center=TABLE_CENTER,table_size=TABLE_SIZE,meshes=contact_assets,
                contact_subsamples=args.contact_subsamples,contact_broad_phase=args.contact_broad_phase,
                velocity_damping_coefficient=args.contact_tangential_damping)
            publish_wire(engine.snapshot())
    rows=EvidenceJournal(args.output/'trajectory.jsonl');rows.configure(
        flush_policy=args.evidence_flush_policy,flush_every=args.evidence_flush_steps,
        flush_seconds=args.evidence_flush_seconds)
    max_gravity_effort=0.
    result['initial_state']={side:dict(joint_names=arm.joint_names,
        joints=arm.data.joint_pos[0].cpu().tolist()) for side,arm in arms.items()}
    if args.dynamics_diagnostics:
        result['drive_readback']={side:dict(
            stiffness=arm.root_physx_view.get_dof_stiffnesses()[0].cpu().tolist(),
            damping=arm.root_physx_view.get_dof_dampings()[0].cpu().tolist(),
            max_force=arm.root_physx_view.get_dof_max_forces()[0].cpu().tolist(),
            max_velocity=arm.root_physx_view.get_dof_max_velocities()[0].cpu().tolist()) for side,arm in arms.items()}
    handoff_seed=None
    closure_normals=None
    timing=dict(control_s=0.,wire_s=0.,rod_solver_s=0.,wire_visual_s=0.,
                isaac_step_s=0.,validation_s=0.,steps=0,rendered_steps=0)
    result['performance']=timing
    result['render_every']=args.render_every
    result['ui_hz']=args.ui_hz
    next_wire_publish=0.
    snapshot_store=SnapshotStore()
    previous_evidence_stage=None
    for step in range(args.steps):
        check_stop_request(args.output)
        tick_start=perf_counter()
        stage_name=getattr(grasp_sequence,'stage','diagnostic')
        wire_step_enabled=wire_dynamics_enabled(stage_name,args.approach_wire_mode)
        if (engine is not None and not wire_step_enabled
                and approach_frozen_snapshot is None):
            # The settle step immediately before approach has already advanced
            # DeformX. Keep that measured state while the robot approaches.
            approach_frozen_snapshot=engine.snapshot()
            previous_states=None
        pre_snapshot=snapshot_store.capture_pre(
            step=step,stage=stage_name,sample_time=step*dt,
            values={'stage':stage_name,'step':step})
        if perf_meter is not None:
            perf_meter.begin_step(step,stage_name,step*dt)
        rod_total_before=timing['rod_solver_s']
        allocation_report=None
        t=(step+1)*dt
        if args.grasp_sequence:
            closure_active=grasp_sequence.stage in ('close','hold','lift')
            if restore_payload is not None:closure_active=grasp_sequence.control_contact_active
            handoff_in_progress=handoff_seed is not None
            if handoff_in_progress:
                commands=np.array([np.r_[closure_arm_targets[s],closure_targets[s]] for s in arms])
                commands,handoff_done=bounded_handoff(commands,handoff_seed,dt)
                for i,s in enumerate(arms):
                    closure_arm_targets[s]=commands[i,:7].copy();closure_targets[s]=commands[i,7:].copy()
                if handoff_done:handoff_seed=None
            if closure_active and not handoff_in_progress:
                if closure_normals is None and grasp_sequence.stage == 'close':
                    pad_points=[];pad_normals=[]
                    rod_position=np.asarray(engine.snapshot()['rod_position'])
                    for c in BODY_NAMES:
                        ref=selected_refs[c];arm=arms[ref['side']]
                        bi=arm.body_names.index(ref['body_name'])
                        pose=arm.data.body_link_pose_w[0,bi].cpu().numpy()
                        R=normalized_wxyz_to_rotation(pose[3:7])
                        pad_points.append(pose[:3]+R@np.asarray(ref['point']))
                        pad_normals.append(R@np.asarray(ref['normal']))
                    oriented, signs=orient_normals_toward_wire(
                        np.asarray(pad_points),np.asarray(pad_normals),rod_position)
                    approach_normals=wire_approach_normals(
                        np.asarray(pad_points),rod_position)
                    closure_normals={c:approach_normals[i] for i,c in enumerate(BODY_NAMES)}
                    result['closure_normal_orientation']=dict(
                        channels=list(BODY_NAMES),signs=signs.tolist(),
                        approach_normals=approach_normals.tolist(),
                        source='measured wire centerline at close entry',
                        scope='simulation closure and normal-load sign only')
                control_J=[];control_n=[];control_W=[]
                for side,arm in arms.items():
                    raw=arm.root_physx_view.get_jacobians()[0].cpu().numpy()
                    Js=[];ns=[]
                    for digit in ('thumb','index','middle'):
                        ref=selected_refs[side+'_'+digit];bi=arm.body_names.index(ref['body_name'])
                        pose=arm.data.body_link_pose_w[0,bi].cpu().numpy();R=normalized_wxyz_to_rotation(pose[3:7])
                        point=pose[:3]+R@np.array(ref['point'])
                        Js.append(point_jacobian(raw[bi-1],point-arm.data.body_com_pos_w[0,bi].cpu().numpy(),arm.joint_names,side)[:,:10])
                        # Isaac's fixed-base Jacobian rows use the opposite
                        # command sign from the measured point displacement
                        # in this asset.  Keep contact-load normals unchanged;
                        # only the velocity-allocation command is inverted.
                        ns.append(-closure_normals[side+'_'+digit])
                    pi=arm.body_names.index('base_link');W=np.zeros((3,10))
                    ids=[arm.joint_names.index(f'{side}_panda_joint{j}') for j in range(1,8)]
                    W[:,:7]=raw[pi-1,3:][:,ids]
                    control_J.append(Js);control_n.append(ns);control_W.append(W)
                # One common scaling before either hand's target is published.
                touch_command=grasp_sequence.touch_output if grasp_sequence.stage=='close' else None
                increments,allocation_report=coordinated_bimanual_increment(control_J,control_n,control_W,
                    previous_loads.reshape(2,3),dt,
                    target_loads_n=(np.asarray(args.closure_loads).reshape(2,3) if touch_command is None
                                    else touch_command['target_loads_n']),
                    normal_speeds_m_s=None if touch_command is None else touch_command['normal_speeds_m_s'],
                    normal_speed_limit_m_s=.002 if speed_profile is None else speed_profile.far_m_s,
                    force_speed_limit_m_s=.002 if speed_profile is None else speed_profile.loaded_m_s,
                    active_pad_mask=None if touch_command is None else
                        np.abs(np.asarray(touch_command['normal_speeds_m_s']))>1e-12,
                    return_diagnostics=True,
                    transport_velocity_w=grasp_sequence.transport_velocity(t))
                for i,side in enumerate(arms):
                    closure_arm_targets[side]+=increments[i,:7]
                    closure_targets[side][:3]+=increments[i,7:]
        h=reference(0. if args.gravity_hold else t)
        arm_target=np.array(home); arm_target[6]+=(h[2]-.2)*.2
        target_hands={};target_arms={}
        approach_q=None if approach is None else approach_target(*approach,t)
        if restore_payload is not None:
            approach_q=approach_target(*approach,grasp_sequence.motion_time(t))
        for side,arm in arms.items():
            target_hands[side]=h if initial_actives is None else initial_actives[side][7:]
            target_arms[side]=arm_target if initial_actives is None else initial_actives[side][:7]
            if approach_q is not None:
                target_hands[side]=approach_q[side][7:];target_arms[side]=approach_q[side][:7]
            if args.closure_pilot:
                normal_rows=[];point_rows_control=[];normals=[]
                raw=arm.root_physx_view.get_jacobians()[0].cpu().numpy()
                for digit in ('thumb','index','middle'):
                    c=side+'_'+digit;ref=selected_refs[c];bi=arm.body_names.index(ref['body_name'])
                    pose=arm.data.body_link_pose_w[0,bi].cpu().numpy();R=normalized_wxyz_to_rotation(pose[3:7])
                    point=pose[:3]+R@np.array(ref['point']);normal=R@np.array(ref['normal'])
                    J=point_jacobian(raw[bi-1],point-arm.data.body_com_pos_w[0,bi].cpu().numpy(),arm.joint_names,side)
                    normal_rows.append(normal@J[:,7:10])
                    point_rows_control.append(J[:,:10]);normals.append(normal)
                i=0 if side=='left' else 3
                if t>.1:
                    if args.coordinated_closure:
                        pi=arm.body_names.index('base_link')
                        arm_ids=[arm.joint_names.index(f'{side}_panda_joint{j}') for j in range(1,8)]
                        palm_angular=np.zeros((3,10));palm_angular[:,:7]=raw[pi-1,3:][:,arm_ids]
                        increment=coordinated_closure_increment(point_rows_control,normals,palm_angular,
                            previous_loads[i:i+2],dt,target_loads_n=args.closure_loads[i:i+2])
                        closure_arm_targets[side]+=increment[:7]
                        closure_targets[side][:3]+=increment[7:]
                    else:
                        closure_targets[side][:3]+=closure_increment(normal_rows,previous_loads[i:i+2],dt,
                            target_loads_n=args.closure_loads[i:i+2],force_to_speed=.01,max_normal_speed=.002)
                target_hands[side]=closure_targets[side]
                target_arms[side]=closure_arm_targets[side]
            if args.grasp_sequence and closure_active:
                target_hands[side]=closure_targets[side]
                target_arms[side]=closure_arm_targets[side]
            if restore_payload is not None and grasp_sequence.stage=='restore_validate':
                # A close-stage restore is already at the approach endpoint.  During
                # restore validation hold the measured live pose rather than replaying
                # the approach/closure target.  Replaying that target while the
                # checkpoint still has residual qd creates an artificial transient,
                # trips the source-clearance guard, and makes a direct-close restore
                # indistinguishable from restarting approach.
                live_q=arm.data.joint_pos[0].cpu().numpy()
                target_arms[side]=np.asarray(
                    [live_q[arm.joint_names.index(f'{side}_panda_joint{i}')] for i in range(1,8)],
                    dtype=float)
                target_hands[side]=np.asarray(
                    [live_q[arm.joint_names.index(n)] for n in ACTIVE], dtype=float)
            target=dict(zip(JOINTS,expand(target_hands[side])));target.update(zip([f'{side}_panda_joint{i}' for i in range(1,8)],target_arms[side]))
            arm.set_joint_position_target(torch.tensor([[target[n] for n in arm.joint_names]],dtype=torch.float32,device=args.device))
            if args.gravity_hold:
                gravity=arm.root_physx_view.get_gravity_compensation_forces()
                try:
                    command=compensated_efforts(gravity,arm.joint_names,side,args.arm_effort_limit,args.hand_effort_limit)
                except ValueError:
                    result['gravity_failure_context']=dict(step=step,side=side,joint_names=arm.joint_names,
                        gravity_nm=gravity[0].cpu().tolist(),actual_q=arm.data.joint_pos[0].cpu().tolist())
                    raise
                max_gravity_effort=max(max_gravity_effort,float(command.abs().max().item()))
                arm.set_joint_effort_target(command)
                if grasp_mode:
                    arm.write_joint_effort_limit_to_sim(drive_headroom(command,arm.joint_names,side,
                        args.arm_effort_limit,args.hand_effort_limit))
            arm.write_data_to_sim()
        control_end=perf_counter()
        if perf_meter is not None:
            perf_meter.record_section('control',control_end-tick_start)
        do_render=should_publish_wire(headless=args.headless,now=perf_counter(),
            next_publish=next_wire_publish,ui_hz=args.ui_hz)
        record=dict(step=step,passed=True,sides={})
        record['pre_snapshot_token']=pre_snapshot.token
        if allocation_report is not None:record['control_allocation']=allocation_report
        source_pre_s=None
        source_post_s=None
        reaction_submit_s=None
        if grasp_mode:
            record['targets']={s:dict(arm=target_arms[s].tolist(),hand=target_hands[s].tolist()) for s in arms}
        if args.dynamics_diagnostics:
            record['dynamics_before']={side:dict(q=arm.data.joint_pos[0].cpu().tolist(),
                qd=arm.data.joint_vel[0].cpu().tolist(),
                mass=arm.root_physx_view.get_mass_matrices()[0].cpu().tolist(),
                gravity=arm.root_physx_view.get_gravity_compensation_forces()[0].cpu().tolist(),
                actuation=arm.root_physx_view.get_dof_actuation_forces()[0].cpu().tolist(),
                estimated_drive_torque=arm.data.applied_torque[0].cpu().tolist()) for side,arm in arms.items()}
        if engine is not None:
            frozen_approach=not wire_step_enabled
            if frozen_approach:
                snapshot=approach_frozen_snapshot
                if snapshot is None:
                    raise RuntimeError('frozen approach state was not initialized')
                # Keep contact evidence explicit: no contact/reaction is
                # allowed before the close stage. Source and collision guards
                # still run below against the measured robot pose.
                samples={c:dict(valid=False,normal_load_n=0.,closest_element=0,
                    closest_sample=0,gap_m=1.,separation_speed_m_s=0.,
                    contact_duration_s=0.,noise_floor_n=.005,
                    separation_tolerance_m_s=.001) for c in BODY_NAMES}
                previous_loads=np.zeros(6)
                record['contact']=dict(normal_samples=samples,
                    max_wire_speed_m_s=float(np.linalg.norm(snapshot['rod_velocity'],axis=0).max()),
                    max_pad_penetration_m=0.,reaction_published=False,
                    reaction_applied=False,approach_wire_frozen=True,
                    peak_force_n={c:0. for c in BODY_NAMES},
                    force_w={c:[0.,0.,0.] for c in BODY_NAMES},
                    torque_com_w={c:[0.,0.,0.] for c in BODY_NAMES})
                record.update(middle_hook_load_n=0., middle_hook_valid=False,
                    middle_hook_direction_w=[0.,0.,1.], middle_hook_contact_duration_s=0.)
                record['rod_state']={k:np.asarray(snapshot[k]).tolist() for k in
                    ('rod_position','rod_velocity','rod_director','rod_omega')}
                if initial_actives is not None:
                    source_guard_start=perf_counter()
                    record['source_before_physics']=measured_source_guard(
                        snapshot,0.,float(engine.rod.radius.max()))
                    source_pre_s=perf_counter()-source_guard_start
            elif args.contact_manifest:
                states={};positions={}
                for c,body in BODY_NAMES.items():
                    arm=arms[c.split('_')[0]];bi=body_indices[c]
                    pose=arm.data.body_link_pose_w[0,bi].cpu().numpy()
                    com=arm.data.body_com_pos_w[0,bi].cpu().numpy();velocity=arm.data.body_com_vel_w[0,bi].cpu().numpy()
                    states[c]=BodyState(body,pose[:3],normalized_wxyz_to_rotation(pose[3:7]),
                        com,velocity[:3],velocity[3:],step*dt,f'step-{step}')
                    positions[c]=com
                motions=mesh_motions(states,offsets,previous_states,dt,args.motion_prediction_order)
                rod_start=perf_counter()
                snapshot,impulses=engine.advance_with_meshes(motions,dt)
                timing['rod_solver_s']+=perf_counter()-rod_start
                reaction_snapshot=snapshot_store.capture_post(
                    step=step,stage=stage_name,sample_time=step*dt+dt,
                    values={'rod_position':snapshot['rod_position']})
                if grasp_mode:
                    # Preserve candidate physics even when a guard rejects the
                    # wrench before sink submission. This is not published-force evidence.
                    record.update(rod_state={k:np.asarray(snapshot[k]).tolist() for k in
                        ('rod_position','rod_velocity','rod_director','rod_omega')})
                    record['contact_candidate']=dict(reaction_published=False,
                        impulses={c:dict(linear_w=i.linear_w.tolist(),angular_com_w=i.angular_com_w.tolist(),
                            peak_force_n=i.peak_force_n,duration_s=i.contact_duration_s) for c,i in impulses.items()},
                        mesh_end_poses={c:dict(position_w=b.position_collection[:,0].tolist(),
                            rotation_wb=b.director_collection[:,:,0].T.tolist()) for c,b in engine.mesh_bodies.items()})
                if initial_actives is not None:
                    source_guard_start=perf_counter()
                    record['source_before_physics']=measured_source_guard(snapshot,
                        engine.last_contact_diagnostics['rod_path_bound_m'] if grasp_mode else 0.,
                        engine.last_contact_diagnostics['max_rod_radius_m'] if grasp_mode else .008)
                    source_pre_s=perf_counter()-source_guard_start
                if grasp_mode:
                    failures=[];samples={}
                    for c,body in engine.mesh_bodies.items():
                        geometry=engine.contact_geometry(c)
                        side=c.split('_')[0]
                        normal=(closure_normals[c] if closure_normals is not None
                                else states[c].rotation_wb@np.array(selected_refs[c]['normal']))
                        samples[c]=normal_contact_sample(force_on_mesh=impulses[c].linear_w/dt,normal=normal,
                            gaps=geometry['gaps'],relative_normal_speeds=geometry['relative_normal_speeds'],
                            element_ids=geometry['element_ids'],
                            contact_duration_s=impulses[c].contact_duration_s,dt=dt)
                    previous_loads=np.array([samples[c]['normal_load_n'] for c in BODY_NAMES])
                    max_speed=float(np.linalg.norm(snapshot['rod_velocity'],axis=0).max())
                    if max_speed>2:failures.append('wire speed gate')
                    contact_metrics_before=dict(normal_samples=samples,max_wire_speed_m_s=max_speed)
                    middle = [samples[f'{side}_middle'] for side in ('left','right')]
                    record.update(middle_hook_load_n=float(max(x['normal_load_n'] for x in middle)),
                        middle_hook_valid=bool(all(x['valid'] for x in middle)),
                        middle_hook_direction_w=[0.,0.,1.],
                        middle_hook_contact_duration_s=float(min(x['contact_duration_s'] for x in middle)))
                    if not closure_active:
                        no_contact_failures,no_contact_metrics=prepublish_contact_gates(snapshot,engine.mesh_bodies,motions,
                            impulses=impulses,mode='no-contact',selected_channel=None,max_speed_m_s=2.,
                            max_pad_penetration_m=.001,radius_m=engine.rod.radius)
                        failures.extend(no_contact_failures)
                        contact_metrics_before['preclosure_no_contact']=no_contact_metrics
                else:
                    failures,contact_metrics_before=prepublish_contact_gates(snapshot,engine.mesh_bodies,motions,
                        impulses=impulses,mode='single-pad' if args.probe_channel else 'no-contact',
                        selected_channel=args.probe_channel,max_speed_m_s=2.,
                        max_pad_penetration_m=.001,radius_m=engine.rod.radius)
                if failures:
                    result['contact_gate_failure']=dict(failures=failures,**contact_metrics_before)
                    raise ValueError('contact gates failed before reaction publish')
                result['pending_contact']=dict(step=step,dt=dt,
                    durations_s={c:i.contact_duration_s for c,i in impulses.items()},
                    peak_force_n={c:i.peak_force_n for c,i in impulses.items()})
                reaction_start=perf_counter()
                batch=bridge.publish(f'step-{step}-post-{reaction_snapshot.token}',step*dt,
                    impulses,engine.last_contact_diagnostics,positions,dt)
                reaction_submit_s=perf_counter()-reaction_start
                if grasp_mode:record['contact_candidate']['reaction_published']=True
                previous_states=states
                contact_peak=max(contact_peak,max(i.peak_force_n for i in impulses.values()))
                record['contact']=dict(peak_force_n={c:i.peak_force_n for c,i in impulses.items()},
                    reaction_snapshot_token=reaction_snapshot.token,
                    reaction_applied=not args.disable_contact_reaction,
                    **contact_metrics_before,
                    force_w={c:w.force_w.tolist() for c,w in batch.wrenches.items()},
                    torque_com_w={c:w.torque_com_w.tolist() for c,w in batch.wrenches.items()})
                if grasp_mode:
                    record['contact']['impulses']={c:dict(linear_w=i.linear_w.tolist(),angular_com_w=i.angular_com_w.tolist(),
                        duration_s=i.contact_duration_s) for c,i in impulses.items()}
                    record['rod_path_bound_m']=engine.last_contact_diagnostics['rod_path_bound_m']
                if args.dynamics_diagnostics:
                    record['contact']['joint_wrench_nm']={}
                    for c,w in batch.wrenches.items():
                        side=c.split('_')[0];arm=arms[side];bi=body_indices[c]
                        J=arm.root_physx_view.get_jacobians()[0,bi-1].cpu().numpy()
                        record['contact']['joint_wrench_nm'][c]=(J.T@np.r_[w.force_w,w.torque_com_w]).tolist()
            else:
                rod_start=perf_counter()
                snapshot=engine.advance(dt)
                timing['rod_solver_s']+=perf_counter()-rod_start
            wm=settled_metrics(snapshot['rod_position'],snapshot['rod_velocity'],snapshot['active_constraints'])
            record['wire']=wm
            record['passed']=bool(np.isfinite(snapshot['rod_position']).all() and
                wm.get('min_surface_z_m',-np.inf)>=.299 and wm.get('max_speed_m_s',np.inf)<=2.)
        wire_end=perf_counter()
        if perf_meter is not None:
            perf_meter.record_section('rod_and_contact',timing['rod_solver_s']-rod_total_before)
            if source_pre_s is not None:
                perf_meter.record_section('source_guard_pre',source_pre_s)
            if reaction_submit_s is not None:
                perf_meter.record_section('reaction_submit',reaction_submit_s)
        sim.step(render=False)
        isaac_end=perf_counter()
        visual_start=perf_counter()
        if do_render:
            publish_wire(snapshot)
            sim.render()
            next_wire_publish=perf_counter()+1./args.ui_hz
        timing['wire_visual_s']+=perf_counter()-visual_start
        if perf_meter is not None:
            perf_meter.record_section('wire_visual',perf_counter()-visual_start)
        if perf_meter is not None:
            perf_meter.record_section('physx_step',isaac_end-wire_end)
        post_snapshot=snapshot_store.capture_post(
            step=step,stage=stage_name,sample_time=(step+1)*dt,
            values={'wire':snapshot if engine is not None else None})
        state_start=perf_counter()
        result['physics_stepped']=True
        for side,arm in arms.items():
            arm.update(dt)
            q=arm.data.joint_pos[0].cpu().numpy();v=arm.data.joint_vel[0].cpu().numpy()
            ids=[arm.joint_names.index(n) for n in JOINTS]
            hm=metrics(q[ids],v[ids],target_hands[side])
            arm_ids=[arm.joint_names.index(f'{side}_panda_joint{i}') for i in range(1,8)]
            arm_error=float(np.max(np.abs(q[arm_ids]-target_arms[side])))
            wi=arm.body_names.index(f'{side}_panda_link8');pi=arm.body_names.index('base_link')
            poses=arm.data.body_link_state_w[0]
            mount_error=float(torch.linalg.vector_norm(poses[wi,:3]-poses[pi,:3]).item())
            orientation_error=float(torch.minimum(torch.linalg.vector_norm(poses[wi,3:7]-poses[pi,3:7]),
                                                  torch.linalg.vector_norm(poses[wi,3:7]+poses[pi,3:7])).item())
            cs=[]
            for sensor in sensors[side]:
                sensor.update(dt,force_recompute=True)
                forces=sensor.data.net_forces_w[0].cpu().numpy()
                cm=contact_metrics(forces,len(sensor.body_names))
                cm['body_forces_n']=dict(zip(sensor.body_names,forces.tolist()))
                cs.append(cm)
            arm_speed=float(np.abs(v[arm_ids]).max())
            passed=hm['passed'] and np.isfinite(arm_error) and arm_error<np.deg2rad(1) and arm_speed<=.151 and mount_error<1e-4 and orientation_error<1e-3 and all(c['passed'] for c in cs)
            record['sides'][side]=dict(hand=hm,arm_error_rad=arm_error,mount_error_m=mount_error,
                mount_quaternion_error=orientation_error,contacts=cs,actual_joint_positions=q.tolist(),
                actual_joint_velocities=v.tolist(),joint_names=arm.joint_names,arm_speed_rad_s=arm_speed)
            record['passed']=bool(record['passed'] and passed)
            if point_refs is not None and step>=120 and step%120==0:
                active_ids=arm_ids+[arm.joint_names.index(n) for n in ACTIVE]
                active_q=q[active_ids]; active_v=v[active_ids]
                raw=arm.root_physx_view.get_jacobians()[0].cpu().numpy()
                if raw.shape!=(len(arm.body_names)-1,6,18):raise ValueError('unexpected fixed-base Jacobian layout')
                for ref in point_refs['references']:
                    if ref['side']!=side:continue
                    bi=arm.body_names.index(ref['body_name'])
                    if bi==0:raise ValueError('cannot probe fixed root with reduced body rows')
                    local=np.array(ref['point']);model=models[side]
                    errors=compare_point(raw[bi-1],arm.joint_names,side,
                        arm.data.body_link_state_w[0,bi,:7].cpu().numpy(),
                        arm.data.body_com_pos_w[0,bi].cpu().numpy(),
                        arm.data.body_com_vel_w[0,bi].cpu().numpy(),local,
                        model.point(active_q,ref['body_name'],local),
                        model.point_jacobian(active_q,ref['body_name'],local),active_v)
                    ok=(errors['position_error_m']<=1e-4 and errors['jacobian_error_m_rad']<=1e-4
                        and errors['velocity_error_m_s']<=1e-3)
                    point_rows.append(dict(step=step,side=side,digit=ref['digit'],passed=ok,**errors))
                    record['passed']=record['passed'] and ok
        if initial_actives is not None:
                source_guard_start=perf_counter()
                record['source_after_physics']=measured_source_guard(snapshot,wire_radius=float(engine.rod.radius.max()))
                source_post_s=perf_counter()-source_guard_start
        if grasp_mode:
            palm_poses={};mismatch={}
            for side,arm in arms.items():
                pose=arm.data.body_link_pose_w[0,arm.body_names.index('base_link')].cpu().numpy()
                t=np.eye(4);t[:3,:3]=normalized_wxyz_to_rotation(pose[3:7]);t[:3,3]=pose[:3]
                palm_poses[side]=t
            anchors=material_in_palm(snapshot['rod_position'].T,palm_poses,material_ids=fixed_material_ids)
            all_contact=all(s['valid'] for s in samples.values())
            now=(step+1)*dt
            if all_contact:
                if material_baseline is None:material_baseline=anchors.copy()
                if contact_since is None:contact_since=now
                longest_contact_s=max(longest_contact_s,now-contact_since)
            else:contact_since=None
            slip=None if material_baseline is None else float(np.linalg.norm(anchors-material_baseline,axis=1).max())
            if slip is not None:max_material_slip_m=max(max_material_slip_m,slip)
            for c,body in engine.mesh_bodies.items():
                arm=arms[c.split('_')[0]];pose=arm.data.body_link_pose_w[0,body_indices[c]].cpu().numpy()
                R=normalized_wxyz_to_rotation(pose[3:7]);actual=pose[:3]+R@np.array(offsets[c])
                predicted_R=body.director_collection[:,:,0].T
                mismatch[c]=dict(position_m=float(np.linalg.norm(actual-body.position_collection[:,0])),
                    rotation_rad=float(np.arccos(np.clip((np.trace(predicted_R.T@R)-1)/2,-1,1))))
            record['material_observation']=dict(material_ids=fixed_material_ids,anchors_in_base_link=anchors.tolist(),
                measured_base_link_poses={s:t.tolist() for s,t in palm_poses.items()},time_s=now,
                current_continuous_contact_s=0. if contact_since is None else now-contact_since,
                longest_continuous_contact_s=longest_contact_s,slip_from_first_valid_contact_m=slip,
                max_slip_since_first_valid_contact_m=max_material_slip_m,contact_pose_mismatch=mismatch,
                diagnostic_contact_hold_not_grasp=True,hold_qualified=False,lift_qualified=False)
            if args.grasp_sequence:
                point_errors=[];normal_errors=[];actual_normals=[]
                for c,ref in selected_refs.items():
                    arm=arms[ref['side']];bi=arm.body_names.index(ref['body_name'])
                    pose=arm.data.body_link_pose_w[0,bi].cpu().numpy();R=normalized_wxyz_to_rotation(pose[3:7])
                    point=pose[:3]+R@ref['point']
                    normal=(closure_normals[c] if closure_normals is not None
                            else R@ref['normal'])
                    goal_point,goal_normal=approach_goals[c]
                    point_errors.append(float(np.linalg.norm(point-goal_point)))
                    normal_errors.append(float(np.arccos(np.clip(normal@goal_normal,-1.,1.))))
                    actual_normals.append(normal)
                nodes=np.asarray(snapshot['rod_position']).T
                station_heights=[float(np.mean(nodes[np.asarray(fixed_material_ids[i:i+3]),2])) for i in (0,3)]
                radius=np.asarray(engine.rod.radius)
                node_radii=np.r_[radius[0],np.maximum(radius[:-1],radius[1:]),radius[-1]]
                surface_clearance=float(np.min(nodes[:,2]-node_radii)-(TABLE_CENTER[2]+TABLE_SIZE[2]/2))
                previous_stage=grasp_sequence.stage
                evidence=grasp_sequence.update(now,settled=bool(record['wire']['passed']),
                    approach_position_errors_m=point_errors,approach_normal_errors_rad=normal_errors,
                    joint_speeds_rad_s=np.array([arm.data.joint_vel[0].cpu().numpy() for arm in arms.values()]),
                    station_heights_m=station_heights,loads_n=previous_loads,
                    surface_gaps_m=[samples[c]['gap_m'] for c in BODY_NAMES],
                    valid_contacts=[samples[c]['valid'] for c in BODY_NAMES],normals=actual_normals,
                    anchors_local=anchors,min_surface_clearance_m=surface_clearance,
                    safety_ok=bool(record['passed']),active_constraints=snapshot['active_constraints'])
                if previous_stage=='approach' and evidence['stage']=='close':
                    approach_measured=dict(passed=True,time_s=now,position_errors_m=point_errors,
                        normal_errors_rad=normal_errors,grasp_qualified=False)
                if previous_stage in ('approach','restore_validate') and evidence['stage']=='close':
                    # Freeze the observed active pose, not the nominal approach endpoint.
                    # Preserve the last published command and rate-limit the takeover.
                    handoff_seed=np.array([[float(arm.data.joint_pos[0,arm.joint_names.index(n)])
                        for n in [f'{s}_panda_joint{i}' for i in range(1,8)]+list(ACTIVE)]
                        for s,arm in arms.items()])
                    closure_arm_targets={s:target_arms[s].copy() for s in arms}
                    closure_targets={s:target_hands[s].copy() for s in arms}
                    record['handoff_seed_active']=handoff_seed.tolist()
                record['handoff_active']=handoff_seed is not None
                record['grasp_sequence']=dict(evidence,actual_station_heights_m=station_heights,
                    min_surface_clearance_m=surface_clearance,approach_position_errors_m=point_errors,
                    approach_normal_errors_rad=normal_errors)
                record['material_observation'].update(diagnostic_contact_hold_not_grasp=False,
                    hold_qualified=evidence['stage'] in ('lift','finish'),lift_qualified=evidence['grasp_lift_passed'])
                if evidence['stage']=='abort':record['passed']=False
                if (args.save_stage_states and previous_stage in PHASES[:-1]
                        and evidence['stage']==PHASES[PHASES.index(previous_stage)+1]
                        and record['passed'] and point_rows and all(p['passed'] for p in point_rows)):
                    entry=evidence['stage'];statuses=grasp_sequence.stage_summary(final_passed=False)
                    predecessors=list(PHASES[:PHASES.index(entry)])
                    if not all(statuses[p] in ('PASS','IMPORTED') for p in predecessors):
                        raise ValueError('stage export lacks qualified predecessor evidence')
                    robot={}
                    for s,arm in arms.items():
                        limits=arm.data.joint_pos_limits[0].cpu().numpy()
                        targets=dict(zip(JOINTS,expand(target_hands[s])))
                        targets.update(zip([f'{s}_panda_joint{i}' for i in range(1,8)],target_arms[s]))
                        robot[s]=dict(joint_names=arm.joint_names,q=arm.data.joint_pos[0].cpu().tolist(),
                            qd=arm.data.joint_vel[0].cpu().tolist(),target=[float(targets[n]) for n in arm.joint_names],
                            lower=limits[:,0].tolist(),upper=limits[:,1].tolist(),root_state=arm.data.root_state_w[0].cpu().tolist())
                    payload=dict(schema='dh116-stage-state-v1',entry_stage=entry,pins=state_pins,
                        source=dict(run=str(args.output.resolve()),step=step,passed_stages=predecessors,
                            imported_from=None if restore_payload is None else args.restore_state_sha256),
                        robot=robot,rod=engine.export_physical_state(),material_ids=fixed_material_ids,
                        anchors_local=(grasp_sequence.gate.baseline.tolist() if grasp_sequence.gate is not None
                            and grasp_sequence.gate.baseline is not None else anchors.tolist()),
                        settled_heights_m=grasp_sequence.settled_heights.tolist(),target_loads_n=args.closure_loads)
                    path=args.output/f'stage_{entry}.json'
                    sha=save_state(path,payload)
                    record['saved_stage_state']=dict(path=str(path),sha256=sha,entry_stage=entry)
                    print('DH116_STAGE_SAVED',entry,str(path),sha,flush=True)
        record['post_snapshot_token']=post_snapshot.token
        evidence_start=perf_counter()
        evidence_stage=record.get('grasp_sequence',{}).get('stage',stage_name)
        phase_changed=(previous_evidence_stage is not None and evidence_stage!=previous_evidence_stage)
        if phase_changed or not record['passed']:
            rows.append(record,force_flush=True)
        else:
            rows.append(record)
        previous_evidence_stage=evidence_stage
        evidence_s=perf_counter()-evidence_start
        if perf_meter is not None:
            perf_meter.record_section('state_and_sensors',perf_counter()-state_start)
            if source_post_s is not None:
                perf_meter.record_section('source_guard_post',source_post_s)
            perf_meter.record_section('evidence',evidence_s)
            perf_meter.end_step(completed=bool(record['passed']))
        timing['control_s']+=control_end-tick_start
        timing['wire_s']+=wire_end-control_end
        timing['isaac_step_s']+=isaac_end-wire_end
        timing['validation_s']+=perf_counter()-isaac_end
        timing['steps']+=1
        timing['rendered_steps']+=int(do_render)
        if step%120==0:
            print('DH116_TIMING',step+1,json.dumps(timing),flush=True)
            print('DH116_PROGRESS',step+1,args.steps,record.get('wire'),flush=True)
            if grasp_mode:
                print('DH116_LOADS',step+1,record['contact']['normal_samples'],flush=True)
                print('DH116_MATERIAL',step+1,evidence_json(record['material_observation']),flush=True)
            if args.grasp_sequence:print('DH116_SEQUENCE',step+1,evidence_json(record['grasp_sequence']),flush=True)
        if not record['passed']: break
        if args.grasp_sequence and record['grasp_sequence']['grasp_lift_passed']:break
    completed=(grasp_sequence.stage=='finish') if args.grasp_sequence else len(rows)==args.steps
    result.update(passed=completed and all(r['passed'] for r in rows),steps=len(rows),
        asset_sha256=manifest['asset_sha256'],gravity=args.gravity_hold,mount='identity simulation-only',
        max_gravity_effort_nm=max_gravity_effort,
        arm_synthetic_gains=dict(stiffness=80.,damping=4.,effort_nm=args.arm_effort_limit,velocity_rad_s=.15),
        last=rows[-1],full_collision_clearance_validated=False)
    if engine is not None:
        result['wire']=dict(**rows[-1]['wire'],mode=snapshot['mode'],hand_contact_enabled=bool(args.contact_manifest),
            grasp_qualified=False,sim_time_s=snapshot['sim_time'])
        if not args.probe_channel and not grasp_mode:result['passed']=result['passed'] and rows[-1]['wire']['passed']
    if args.contact_manifest:
        result['contact']=dict(manifest_sha256=args.contact_manifest_sha256,probe_channel=args.probe_channel,
            peak_force_n=contact_peak,reaction_applied=not args.disable_contact_reaction,
            paired_response_validated=False,grasp_qualified=False)
        if args.probe_channel:result['passed']=result['passed'] and contact_peak>0
    if args.closure_pilot:
        result['closure_pilot']=dict(diagnostic_initial_touch_not_continuous_grasp=True,grasp_qualified=False,
            last_loads_n=previous_loads.tolist(),all_four_contact_steps=sum(all(v['valid'] for v in
                row['contact']['normal_samples'].values()) for row in rows),hold_qualified=False,lift_qualified=False)
    if args.grasp_sequence:
        continuous_qualified=bool(result['passed']) and restore_payload is None
        result['continuous_grasp']=dict(rows[-1]['grasp_sequence'],grasp_qualified=continuous_qualified)
        if restore_payload is not None:
            result['continuous_grasp']['grasp_lift_passed']=False
            result['continuous_grasp']['whole_wire_clear']=False
        result['approach']=approach_measured
        result['wire']['grasp_qualified']=continuous_qualified
        result['contact']['grasp_qualified']=continuous_qualified
    if approach is not None and not args.grasp_sequence:
        final_errors=[]
        for ref in point_refs['references']:
            side=ref['side'];arm=arms[side];model=models[side]
            pose=arm.data.body_link_pose_w[0,arm.body_names.index(ref['body_name'])].cpu().numpy()
            R=normalized_wxyz_to_rotation(pose[3:7]);local=np.array(ref['point']);normal=np.array(ref['normal'])
            target_pose=model.fk(approach[1][side])[0][model.bodies[ref['body_name']]]
            err=float(np.linalg.norm(pose[:3]+R@local-(target_pose[:3,3]+target_pose[:3,:3]@local)))
            angle=float(np.arccos(np.clip(np.dot(R@normal,target_pose[:3,:3]@normal),-1,1)))
            final_errors.append(dict(side=side,digit=ref['digit'],position_error_m=err,normal_error_rad=angle,
                                     passed=err<=.002 and angle<=np.deg2rad(5)))
        result['approach']=dict(passed=result['passed'] and all(e['passed'] for e in final_errors),
                               final_errors=final_errors,grasp_qualified=False)
        result['passed']=result['approach']['passed']
except Exception as exc:
    import traceback
    traceback.print_exc()
    result['passed']=False
    result['error']=repr(exc)
    result['failure_step']=locals().get('step')
    result['last_attempted_record']=locals().get('record')
    if args.grasp_sequence:
        result['continuous_grasp']=dict(stage='abort',reason=repr(exc),grasp_qualified=False,
            grasp_lift_passed=False,whole_wire_clear=False)
finally:
    if locals().get('point_refs') is not None:
        # Keep completed probes even when a later collision/contact guard aborts.
        point_rows=locals().get('point_rows',[])
        result['point_probe']=dict(samples=len(point_rows),passed=bool(point_rows) and all(r['passed'] for r in point_rows),
            reference_sha256=hashlib.sha256(args.point_references.read_bytes()).hexdigest(),
            max_position_error_m=max((r['position_error_m'] for r in point_rows),default=None),
            max_jacobian_error_m_rad=max((r['jacobian_error_m_rad'] for r in point_rows),default=None),
            max_velocity_error_m_s=max((r['velocity_error_m_s'] for r in point_rows),default=None))
        (args.output/'point_probe.json').write_text(evidence_json(point_rows,indent=2))
        result['passed']=result['passed'] and result['point_probe']['passed']
    if not result['passed']:
        for key in ('wire','contact','continuous_grasp'):
            if key in result:result[key]['grasp_qualified']=False
        if 'continuous_grasp' in result:
            result['continuous_grasp']['grasp_lift_passed']=False
            result['continuous_grasp']['whole_wire_clear']=False
    if args.grasp_sequence:
        if 'stage_reset' in result:result['stage_reset']['local_stages_passed']=bool(result['passed'])
        result['stage_status']=(grasp_sequence.stage_summary(final_passed=bool(result['passed']))
            if grasp_sequence is not None else dict.fromkeys(('settle','approach','close','hold','lift'),'NOT_RUN'))
    if perf_meter is not None:
        result['performance_profile']=perf_meter.close()
    if 'rows' in locals():rows.close()
    (args.output/'summary.json').write_text(evidence_json(result,indent=2)+'\n')
    print('DH116_MOUNTED',evidence_json(result),flush=True)
    app.close(wait_for_replicator=False,skip_cleanup=True)
