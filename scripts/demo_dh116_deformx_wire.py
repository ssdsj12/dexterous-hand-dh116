"""Single-environment real flexible wire import; run only with prior Isaac stopped."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose-summary',type=Path,default=Path('runs/reference_aligned_native_20261023_fix1/summary.json'))
    parser.add_argument('--grasp-initialization',type=Path,default=Path('runs/tabletop_pressure_geometry_20261024/initialization.json'))
    parser.add_argument('--load-profile',type=Path,default=Path('configs/dh116_load_profiles/v2_pressure_candidate.json'))
    parser.add_argument('--asset',type=Path,default=Path('assets/deformx_wire_v1/wire.usda'))
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--contact-probe-offset-x-m',type=float,default=0.,
        help='validation only: offset wire X by at most .01 m to probe contact; hands stay at their seed')
    parser.add_argument('--output',type=Path,default=Path('runs/deformx_import_'+time.strftime('%Y%m%d_%H%M%S')))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.contact_probe_offset_x_m) or abs(args.contact_probe_offset_x_m) > .01:
        parser.error('--contact-probe-offset-x-m must be finite and within [-.01,.01] m')
    if args.steps < 1:
        parser.error('--steps must be positive')
    if not str(args.device).startswith('cuda'):
        parser.error('the flexible runtime requires --device cuda:0')
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(passed=False,contact_detected=False,grasp_qualified=False,policy_executed=False,
        status='initializing',physics_hz=480,rope_self_contact=False,
        validation_only_contact_probe=dict(enabled=args.contact_probe_offset_x_m != 0.,
            wire_offset_x_m=args.contact_probe_offset_x_m,hand_roots_unchanged=True),
        visual_geometry='original DeformX skeleton skinning; stretch/shear and final tip only approximate',
        approximation='explicit penalty contact; Coulomb coefficient 1.0 approximates high friction, not identical PhysX static/dynamic law',
        contact_subsamples=10,contact_sample_spacing_m=.005,contact_stiffness_n_m=10000.,contact_damping_ns_m=10.,
        physics_radius_m=.005,upstream_example_physics_radius_m=.00635,material_status='example parameters, not calibrated',rows=[])
    def save():
        target=args.output/'monitor.json'
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(target)
    save()
    app = AppLauncher(args).app
    start=time.monotonic()
    try:
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        import isaaclab.sim as sim_utils
        from pxr import Usd,UsdGeom
        from go2_pvcnn.control.dh116.hand_only_scene import native_pose_hand_only_layout
        from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import adaptive_initialization
        from go2_pvcnn.control.dh116.tabletop_vec_isaac import spawn_scene,rotation
        from go2_pvcnn.control.dh116.tabletop_load_parameters import read_load_parameters,physics_load_readback
        from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,MIMIC
        from go2_pvcnn.control.dh116.tabletop_task import joint_safety_failure
        from go2_pvcnn.control.dh116.native_scene import _PAD_LOCAL_POINTS
        from go2_pvcnn.control.dh116.deformx_wire_asset import SkeletonWire
        from go2_pvcnn.control.dh116.deformx_wire_runtime import WireRuntime,paper_rod_config,link_origin_velocity
        from vendor.DeformX.co_sim.moving_contact_models import MeshMotion
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        root=Path(__file__).resolve().parents[1]
        layout=native_pose_hand_only_layout(json.loads(args.pose_summary.read_text()),roll_degrees=dict(left=76.,right=82.))
        manifest=json.loads((root/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
        sources={x['side']:x['path'] for x in manifest['sources'] if x['kind']=='dh116'}
        provenance=dict(pose_sha256=sha(args.pose_summary),source_sha256={s:sha(p) for s,p in sources.items()},
            physics_hz=480,control_hz=60,solver_type=0,position_iterations=16)
        layout,seed=adaptive_initialization(json.loads(args.grasp_initialization.read_text()),layout,provenance)
        table_top=float(layout['table_center'][2]+layout['table_size'][2]/2)
        layout['wire_center']=np.asarray(layout['wire_center']).copy()
        layout['wire_center'][2]=table_top+.005
        layout['wire_center'][0]+=args.contact_probe_offset_x_m
        args.physics_hz,args.control_mode,args.position_iterations,args.mimic_frequency=480,'adaptive_grasp',16,0
        args.load_parameters=read_load_parameters(args.load_profile)
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/480,device=args.device,
            physx=sim_utils.PhysxCfg(solver_type=0)))
        light=sim_utils.DomeLightCfg(intensity=1500.)
        light.func('/World/ImportLight',light)
        sim.set_camera_view(eye=(1.3,1.2,1.1),target=(.43,0,.3))
        scene=spawn_scene(sim,args,layout,sources,'/World/env',np.zeros(3),external_wire=True)
        visual=SkeletonWire(sim.stage,'/World/env/Wire',args.asset)
        sim.reset()
        tensor=lambda value:torch.as_tensor(value,dtype=torch.float32,device=args.device)
        hands=scene['hands']
        targets=[]
        for hi,hand in enumerate(hands):
            hand.reset()
            q=hand.data.default_joint_pos.clone()
            q[:,[hand.joint_names.index(n) for n in JOINTS]]=tensor(seed['joint_positions'][hi])
            hand.write_joint_state_to_sim(q,torch.zeros_like(q))
            hand.write_root_pose_to_sim(tensor(seed['roots'][hi:hi+1]))
            hand.write_root_velocity_to_sim(tensor(np.zeros((1,6))))
            ids=[hand.joint_names.index(n) for n in ACTIVE]
            targets.append((ids,q[:,ids].clone()))
            hand.set_joint_position_target(targets[-1][1],joint_ids=ids)
            hand.write_data_to_sim()
        sim.physics_sim_view.update_articulations_kinematic()
        for hand in hands:
            hand.update(1/480)
        report['drive_readback']=physics_load_readback(hands,args.load_parameters)
        meshes,body_channels={},{}
        cache=UsdGeom.XformCache(Usd.TimeCode.Default())
        for hi,side in enumerate(('left','right')):
            base=f'/World/env/{side}Hand'
            for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(base),Usd.TraverseInstanceProxies()):
                path=str(prim.GetPath())
                if '/collisions/' not in path or not prim.IsA(UsdGeom.Mesh):
                    continue
                link=path[len(base)+1:].split('/')[0]
                points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
                relative=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(sim.stage.GetPrimAtPath(base+'/'+link)).GetInverse()
                meshes[path]=(np.c_[points,np.ones(len(points))]@np.asarray(relative))[:,:3]
                body_channels[path]=(hi,hands[hi].body_names.index(link))
        if len(meshes)!=34:
            raise ValueError(f'expected all 34 collision meshes, got {len(meshes)}')
        pads=[]
        for hi,side in enumerate(('left','right')):
            pair=[]
            for link,digit in (('finger14_link','thumb'),('finger23_link','index')):
                pose=hands[hi].data.body_link_pose_w[0,hands[hi].body_names.index(link)].cpu().numpy()
                pair.append(pose[:3]+rotation(pose[3:])@_PAD_LOCAL_POINTS[side][digit])
            pads.append(pair)
        pads=np.asarray(pads)
        runtime=WireRuntime(paper_rod_config(layout['wire_center'],layout['table_center'],layout['table_size']),meshes,pads.mean(axis=1),args.device)
        report.update(source_contract=provenance,initialization_sha256=sha(args.grasp_initialization),load_profile_sha256=sha(args.load_profile),
            mesh_channels=list(meshes),initial_pad_points_w=pads.tolist(),initial_pad_surface_gap_m=(np.linalg.norm((pads-layout['wire_center'])[:,:,[0,2]],axis=2)-.005).tolist())
        def export_state(label,state):
            visual.update(state['rod_position'],state['rod_director'])
            sim.stage.Export(str((args.output/(label+'_scene.usda')).resolve()))
            (args.output/(label+'_rod_state.json')).write_text(json.dumps(runtime.engine.export_physical_state(),indent=2)+'\n')
            np.savez(args.output/(label+'_hand_state.npz'),roots=np.stack([h.data.root_pose_w.cpu().numpy() for h in hands]),
                joints=np.stack([h.data.joint_pos.cpu().numpy() for h in hands]),link_poses=np.stack([h.data.body_link_pose_w.cpu().numpy() for h in hands]))
        export_state('initial',runtime.engine.snapshot(device=False))
        report['status']='running';save()
        for tick in range(args.steps):
            report['active_tick']=tick;report['tick_status']='integrating';save()
            motions,coms={},{}
            for name,(hi,bi) in body_channels.items():
                hand=hands[hi]
                pose=hand.data.body_link_pose_w[0,bi].cpu().numpy()
                com=hand.data.body_com_pos_w[0,bi].cpu().numpy()
                velocity=hand.data.body_com_vel_w[0,bi].cpu().numpy()
                coms[(hi,bi)]=com
                # PhysX quaternion normalization avoids float32 orthogonality drift.
                matrix=Rotation.from_quat(np.r_[pose[4:7],pose[3]]).as_matrix()
                motions[name]=MeshMotion(pose[:3],matrix,link_origin_velocity(velocity[:3],velocity[3:],pose[:3],com),velocity[3:],np.zeros(3),np.zeros(3))
            state,reactions,measurement=runtime.advance(motions,coms,body_channels,1/480)
            for hi,hand in enumerate(hands):
                hand.set_joint_position_target(targets[hi][1],joint_ids=targets[hi][0])
                hand.write_data_to_sim()
                force=np.zeros((1,len(hand.body_names),3));torque=np.zeros_like(force)
                positions=hand.data.body_com_pos_w.clone()
                for (which,bi),(f,t) in reactions.items():
                    if which==hi:
                        force[0,bi]=f;torque[0,bi]=t
                hand.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=tensor(force),torque_data=tensor(torque),position_data=positions,
                    indices=torch.tensor([0],device=args.device,dtype=torch.int32),is_global=True)
            report['tick_status']='advancing_physx';save()
            sim.step(render=not args.headless)
            for hand in hands:
                hand.update(1/480)
                q=hand.data.joint_pos[0].cpu().numpy()
                errors=[q[hand.joint_names.index(f)]-r*q[hand.joint_names.index(m)]-o for f,m,r,o in MIMIC]
                failure=joint_safety_failure(q,hand.data.joint_pos_limits[0].cpu().numpy(),errors)
                if failure or not torch.isfinite(hand.data.joint_vel).all():
                    raise FloatingPointError(f'hand state guard: {failure or "nonfinite_joint_velocity"}')
                if not torch.isfinite(hand.data.body_link_pose_w).all():
                    raise FloatingPointError('nonfinite hand state')
            visual.update(state['rod_position'],state['rod_director'])
            measurement.update(tick=tick+1,sim_time_s=(tick+1)/480,wall_time_s=time.monotonic()-start)
            report['rows'].append(measurement)
            report['contact_detected'] |= measurement['contact_detected']
            report['tick_status']='complete';save()
        export_state('final',state)
        initial=json.loads((args.output/'initial_rod_state.json').read_text())['dynamic']['rod_position']
        report.update(passed=True,status='import_finite_balanced',simulated_duration_s=args.steps/480,
            max_node_motion_m=float(np.linalg.norm(state['rod_position']-np.asarray(initial),axis=0).max()))
    except KeyboardInterrupt:
        report.update(status='interrupted',interrupted=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        code=0 if report['passed'] else 2
        report['process_exit_code']=code
        report['wall_time_s']=time.monotonic()-start
        save()
        (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        import omni.kit.app
        omni.kit.app.get_app().post_quit(code)
        app.close(wait_for_replicator=False,skip_cleanup=True)
    raise SystemExit(code)


if __name__=='__main__':
    main()
