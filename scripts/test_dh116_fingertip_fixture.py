"""Simulation-only fixed-plate fingertip loading; never changes PPO parameters."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import traceback

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--initialization',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--efforts',type=float,nargs='+',default=[.2,.5,1.,2.])
parser.add_argument('--stiffness',type=float,default=20.)
parser.add_argument('--repeat',type=int,default=1)
parser.add_argument('--paired-error-cap',type=float,choices=(0.,.008),default=0.,help='diagnostic position lead cap, radians; requires paired-thumb')
parser.add_argument('--paired-solver',type=int,choices=(0,1),default=0,help='diagnostic PGS=0/TGS=1; nondefault requires paired-thumb')
parser.add_argument('--paired-effort-comparison',action='store_true',help='diagnostic 0.2/0.4 Nm comparison; requires paired-thumb')
parser.add_argument('--paired-thumb',action='store_true',help='paired unloaded/loaded thumb test using current training drive settings')
parser.add_argument('--disable-sleep',action='store_true',help='diagnostic comparison only: articulation sleep threshold zero')
parser.add_argument('--position-iterations',type=int,choices=(16,64,128),default=16)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if (args.paired_effort_comparison or args.paired_solver!=0 or args.paired_error_cap!=0) and not args.paired_thumb:
    parser.error('effort comparison requires --paired-thumb')
if args.paired_thumb:
    from go2_pvcnn.control.dh116.fingertip_fixture import paired_thumb_setup
    paired=paired_thumb_setup()
    args.efforts=[paired['effort_nm'],.4] if args.paired_effort_comparison else [paired['effort_nm']];args.stiffness=20.;args.position_iterations=paired['position_iterations']
if (args.output.exists() or (not args.paired_thumb and args.device!='cpu') or not 1<=args.repeat<=3
        or not 10<=args.stiffness<=200 or not args.efforts
        or not all(0<e<=2 for e in args.efforts)):
    parser.error('new output, CPU, bounded gains/efforts and 1..3 repeats required')
args.output.mkdir(parents=True)
app=AppLauncher(args).app
report=dict(status='initializing',hardware_calibrated=False,grasp_qualified=False,results=[])
try:
    import numpy as np
    import torch
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,MIMIC,expand,active_limits
    from go2_pvcnn.control.dh116.runtime_mimic import install_hand_mimic,configure_mimic_compliance
    from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
    from go2_pvcnn.control.dh116.fingertip_fixture import fixture_center,summarize_hold,bounded_position_target
    from go2_pvcnn.control.dh116.tabletop_training import write_json
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters,physics_load_readback
    torch.set_num_threads(2 if args.paired_thumb else 1);torch.set_num_interop_threads(2 if args.paired_thumb else 1)
    device=args.device
    tensor=lambda value:torch.as_tensor(value,dtype=torch.float32,device=device)
    mimic_frequency=paired['mimic_frequency'] if args.paired_thumb else 0.
    velocity_iterations=paired['velocity_iterations'] if args.paired_thumb else 4
    root=Path(__file__).resolve().parents[1]
    payload=json.loads(args.initialization.read_text())
    manifest=json.loads((root/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
    sources={x['side']:x['path'] for x in manifest['sources'] if x['kind']=='dh116'}
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/480,device=device,
        physx=sim_utils.PhysxCfg(solver_type=args.paired_solver)))
    dt=1/480
    friction=sim_utils.RigidBodyMaterialCfg(static_friction=0.,dynamic_friction=0.,
        restitution=0.,friction_combine_mode='min',restitution_combine_mode='min')
    if args.paired_thumb:
        friction.compliant_contact_stiffness=10000.
        friction.compliant_contact_damping=30.
    friction.func('/World/FixtureMaterial',friction)
    material=UsdShade.Material(sim.stage.GetPrimAtPath('/World/FixtureMaterial'))
    cases=[]
    case_specs=(paired['cases'] if args.paired_thumb else
        [dict(side=side,digit=digit,load='loaded',plate_outward_m=0.)
         for side,digit in [('left','thumb'),('left','index'),('right','thumb'),('right','index')]])
    for i,spec in enumerate(case_specs):
        side,digit=spec['side'],spec['digit']
        hi=0 if side=='left' else 1
        q=np.array(payload['grasp_command'][hi*6:hi*6+6],dtype=float)
        pose=np.array(payload['roots'][hi],dtype=float);pose[1]+=i*2.
        source=Path(sources[side]);sha=hashlib.sha256(source.read_bytes()).hexdigest()
        if sha!=payload['source_contract']['source_sha256'][side]:raise ValueError('source SHA mismatch')
        fk=HandPointFK(source,side,pose)
        link='finger14_link' if digit=='thumb' else 'finger23_link'
        joint='finger12' if digit=='thumb' else 'finger21'
        index=ACTIVE.index(joint);sign=-1 if digit=='thumb' else 1
        jac=fk.jacobian(q)[0 if digit=='thumb' else 1]
        if sign*jac[0,index]<=.001:raise ValueError('positive flexion must move toward fixture')
        prefix=f'/World/fixture_{i}'
        UsdGeom.Xform.Define(sim.stage,prefix)
        handpath=prefix+'/Hand'
        hand=Articulation(ArticulationCfg(prim_path=handpath,
            spawn=sim_utils.UsdFileCfg(usd_path=str(source),activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                    enabled_self_collisions=False,solver_position_iteration_count=args.position_iterations,solver_velocity_iteration_count=velocity_iterations,
                    sleep_threshold=0. if args.disable_sleep else None)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pose[:3]),rot=tuple(pose[3:]),
                joint_pos=dict(zip(JOINTS,expand(q).tolist()))),
            actuators={'active':ImplicitActuatorCfg(joint_names_expr=list(ACTIVE),stiffness=args.stiffness,
                damping=.2,effort_limit_sim=args.efforts[0],velocity_limit_sim=.3),
                'passive':ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE],
                    stiffness=0.,damping=0.,effort_limit_sim=0.,velocity_limit_sim=paired['passive_speed'] if args.paired_thumb else 3.)}))
        install_hand_mimic(sim.stage,handpath)
        configure_mimic_compliance(sim.stage,handpath,mimic_frequency,1.,allow_rigid=True)
        # Local runtime overrides only; source USD meshes remain untouched.
        while True:
            instances=[p for p in Usd.PrimRange(sim.stage.GetPrimAtPath(handpath)) if p.IsInstance()]
            if not instances:break
            for prim in instances:prim.SetInstanceable(False)
        vertices=[];collider_paths=[]
        cache=UsdGeom.XformCache(Usd.TimeCode.Default())
        for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(handpath),Usd.TraverseInstanceProxies()):
            keep=str(prim.GetPath()).startswith(handpath+'/'+link+'/')
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(keep)
                if keep:
                    collider_paths.append(str(prim.GetPath()))
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,materialPurpose='physics')
                    PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(.001)
                    PhysxSchema.PhysxCollisionAPI(prim).CreateRestOffsetAttr(0.)
            # Source applies CollisionAPI to a parent Xform, while its mesh
            # child carries the geometry but no CollisionAPI of its own.
            if not keep or '/collisions/' not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh):continue
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            transform=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(
                sim.stage.GetPrimAtPath(handpath+'/'+link)).GetInverse()
            points=(np.c_[points,np.ones(len(points))]@np.asarray(transform))[:,:3]
            t=next(v for k,v in fk.poses(q).items() if k.endswith('/'+link))
            vertices.append(points@t[:3,:3].T+t[:3,3])
        if not vertices:raise ValueError('no enabled pad collider')
        vertices=np.concatenate(vertices)
        center=fixture_center(vertices,sign)
        center[0]+=sign*spec['plate_outward_m']
        plate=RigidObject(RigidObjectCfg(prim_path=prefix+'/Plate',
            spawn=sim_utils.CuboidCfg(size=(.004,.06,.06),activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
                physics_material=friction,mass_props=sim_utils.MassPropertiesCfg(mass=1.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(center))))
        sensor=ContactSensor(ContactSensorCfg(prim_path=handpath+'/'+link,
            filter_prim_paths_expr=[prefix+'/Plate'],update_period=0.,track_contact_points=True))
        cases.append(dict(name=side+'_'+digit+('_'+spec['load'] if args.paired_thumb else ''),load_case=spec['load'],hand=hand,plate=plate,sensor=sensor,q=q,pose=pose,
            sign=sign,index=index,center=center,setup=dict(side=side,digit=digit,load_case=spec['load'],plate_outward_m=spec['plate_outward_m'],source_sha256=sha,
                root_pose=pose.tolist(),active_q_rad=q.tolist(),loaded_joint=joint,
                selected_pad=link,enabled_hand_colliders=collider_paths,plate_center_m=center.tolist(),
                initial_gap_m=.0005+spec['plate_outward_m'],plate_size_m=[.004,.06,.06],
                force_direction_w=[sign,0,0],point_jacobian_x_m_per_rad=float(jac[0,index]))))
    sim.reset();signal.signal(signal.SIGINT,signal.default_int_handler)
    for c in cases:
        h=c['hand'];c['ids']=[h.joint_names.index(n) for n in ACTIVE]
        c['joint_reset']=tensor([[dict(zip(JOINTS,expand(c['q'])))[n] for n in h.joint_names]])
        c['target']=tensor(c['q'][None])
    report.update(status='running',physics_hz=480,command_update_hz=480,trace_hz=60,hold_measurement_hz=480,stiffness_nm_per_rad=args.stiffness,
        damping_nms_per_rad=.2,mimic_frequency=mimic_frequency,position_iterations=args.position_iterations,velocity_iterations=velocity_iterations,
        paired_thumb=args.paired_thumb,device=device,solver_type=args.paired_solver,position_lead_cap_rad=args.paired_error_cap,
        reaction_semantics='total solver joint reaction, includes constraints; NOT isolated motor torque',
        joint_orders=[list(c['hand'].joint_names) for c in cases],body_orders=[list(c['hand'].body_names) for c in cases],
        fixture='fixed kinematic frictionless plate; only selected pad collision enabled',
        gravity_on_hand=False,command_rate_rad_s=.05,maximum_command_offset_rad=.15,
        sleep_threshold_override=0. if args.disable_sleep else None,
        scope='paired single-thumb control isolation' if args.paired_thumb else 'four loaded fingertips at current grasp posture, no PPO and no real hardware',
        contact_stiffness_n_m=10000. if args.paired_thumb else None,contact_damping_ns_m=30. if args.paired_thumb else None,
        torque_label='implicit PD estimate, not solver-measured actuator torque',
        cases=[c['setup'] for c in cases],effort_caps_nm=args.efforts,repeat=args.repeat,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initialization_sha256=hashlib.sha256(args.initialization.read_bytes()).hexdigest())
    (args.output/'diagnostic_source.py').write_text(Path(__file__).read_text())
    write_json(args.output/'configuration.json',report)
    for repeat in range(args.repeat):
      for effort in args.efforts:
        rows=[[] for _ in cases];holds=[[] for _ in cases]
        faults=[None]*len(cases);peak=np.zeros(len(cases));first=np.zeros(len(cases))
        for c in cases:
            h=c['hand'];h.reset();h.write_joint_state_to_sim(c['joint_reset'],torch.zeros_like(c['joint_reset']))
            h.write_root_pose_to_sim(tensor(c['pose'][None]))
            h.write_root_velocity_to_sim(torch.zeros((1,6),device=device))
            h.write_joint_effort_limit_to_sim(torch.full((1,6),effort,device=device),joint_ids=c['ids'])
            h.actuators['active'].effort_limit[:]=effort
            h.set_joint_position_target(c['target'],joint_ids=c['ids']);h.write_data_to_sim()
            c['sensor'].reset()
            c['command']=c['q'].copy()
        readback=physics_load_readback([c['hand'] for c in cases],LoadParameters(
            stiffness_max=max(10.01,args.stiffness),effort_limit_nm=effort))
        write_json(args.output/f'readback_r{repeat}_effort{effort:g}.json',readback)
        for step in range(9*480):
            t=(step+1)*dt
            # .5s unloaded,3s ramp,2s hold,3s unload,.5s released.
            offset=min(.15,max(0.,(t-.5)*.05)) if t<=5.5 else max(0.,.15-(t-5.5)*.05)
            phase='unloaded' if t<=.5 else 'ramp' if t<=3.5 else 'hold' if t<=5.5 else 'unload' if t<=8.5 else 'released'
            for i,c in enumerate(cases):
                h=c['hand'];target=c['q'].copy();target[c['index']]+=offset if faults[i] is None else 0.
                if args.paired_error_cap:
                    k=c['index'];actual=float(h.data.joint_pos[0,c['ids'][k]].item())
                    target[k]=bounded_position_target(target[k],actual,c['command'][k],dt,args.paired_error_cap)
                expand(target)
                c['command']=target.copy()
                h.set_joint_position_target(tensor(target[None]),joint_ids=c['ids'])
                h.write_data_to_sim()
            sim.step(render=False)
            for i,c in enumerate(cases):
                h=c['hand'];h.update(dt);c['plate'].update(dt);c['sensor'].update(dt,force_recompute=True)
                f=c['sensor'].data.force_matrix_w[0,0,0].cpu().numpy()
                q=h.data.joint_pos[0].cpu().numpy();qd=h.data.joint_vel[0].cpu().numpy()
                byname=dict(zip(h.joint_names,q))
                mimic=max(abs(byname[follower]-ratio*byname[master]-shift) for follower,master,ratio,shift in MIMIC)
                normal=max(0.,float(-c['sign']*f[0]));peak[i]=max(peak[i],normal)
                drift=float(np.max(abs(c['plate'].data.root_pos_w[0].cpu().numpy()-c['center'])))
                failure=None
                if not np.isfinite(np.r_[f,q,qd,drift]).all():failure='nonfinite_state'
                elif args.paired_thumb and abs(qd).max()>3.+1e-5:failure='joint_velocity'
                elif mimic>.02:failure='mimic_error'
                elif np.linalg.norm(f)>30:failure='excess_force'
                elif drift>1e-6:failure='fixture_moved'
                lo,up=active_limits();qa=q[c['ids']]
                if not args.paired_thumb and (np.any(qa<lo-.01) or np.any(qa>up+.01)):failure='joint_limit'
                if args.paired_thumb:
                    bounds=h.data.joint_pos_limits[0].cpu().numpy()
                    if np.any(q<bounds[:,0]-.02) or np.any(q>bounds[:,1]+.02):failure='joint_limit'
                    if c['load_case']=='unloaded' and np.linalg.norm(f)>.05:failure='unexpected_unloaded_contact'
                if failure and faults[i] is None:faults[i]=failure
                if phase=='unloaded':first[i]=max(first[i],normal)
                if 4.5<=t<=5.5:holds[i].append((t,normal,float(mimic)))
                if step%8==7 or failure:
                    reaction={}
                    if args.paired_thumb:
                        reaction=dict(solver_projected_joint_force_nm=h.root_physx_view.get_dof_projected_joint_forces()[0].cpu().tolist(),
                            solver_incoming_joint_wrench_child_frame=h.root_physx_view.get_link_incoming_joint_force()[0].cpu().tolist())
                    contact=c['sensor'].data.contact_pos_w[0,0,0].cpu().numpy()
                    rows[i].append(dict(t=t,phase=phase,command_offset_rad=offset,
                        normal_force_n=normal,force_on_pad_w_n=f.tolist(),
                        contact_position_w_m=contact.tolist() if np.isfinite(contact).all() else None,
                        active_q_rad=qa.tolist(),max_joint_speed_rad_s=float(abs(qd).max()),
                        joint_positions_rad={n:float(v) for n,v in byname.items()},
                        command_error_rad=(c['command']-qa).tolist(),active_command_rad=c['command'].tolist(),
                        signed_mimic_errors_rad={follower:float(byname[follower]-ratio*byname[master]-shift) for follower,master,ratio,shift in MIMIC},
                        estimated_unclipped_active_pd_torque_nm=h.data.computed_torque[0,c['ids']].cpu().tolist(),
                        mimic_error_rad=float(mimic),fixture_drift_m=drift,
                        estimated_active_pd_torque_nm=h.data.applied_torque[0,c['ids']].cpu().tolist(),
                        safety_failure=faults[i],**reaction))
            if step%120==119:
                write_json(args.output/'monitor.json',dict(status='loading',repeat=repeat,effort_limit_nm=effort,
                    simulation_time_s=t,phase=phase,normal_force_n=[r[-1]['normal_force_n'] for r in rows],
                    peak_force_n=peak.tolist(),safety_failures=faults))
            if all(faults) or (args.paired_thumb and any(faults)):break
        for i,c in enumerate(cases):
            hold=holds[i]
            stats=summarize_hold([r[0] for r in hold],[r[1] for r in hold],[r[2] for r in hold],
                max_sample_gap_s=dt+1e-9)
            valid=bool(faults[i] is None and first[i]<.05 and (not args.paired_thumb or t>=9.))
            result=dict(name=c['name'],repeat=repeat,effort_limit_nm=effort,
                valid=valid,completed=bool(t>=9.),simulation_time_s=t,load_case=c['load_case'],safety_failure=faults[i],unloaded_peak_n=float(first[i]),
                peak_n=float(peak[i]),hold=stats,sustained_14n=valid and stats['sustained_14n'])
            report['results'].append(result)
            write_json(args.output/f'{c["name"]}_r{repeat}_effort{effort:g}.json',rows[i])
            write_json(args.output/f'{c["name"]}_r{repeat}_effort{effort:g}_hold480hz.json',
                dict(columns=['time_s','normal_force_n','mimic_error_rad'],samples=hold))
        write_json(args.output/'results.json',report)
        print('FIXTURE_TRIAL',json.dumps(report['results'][-len(cases):]),flush=True)
        if args.paired_thumb and any(faults):raise RuntimeError('paired fixture stopped on first physical fault: '+str(faults))
    report['status']='completed'
except KeyboardInterrupt:
    report['status']='interrupted'
except Exception:
    report.update(status='error',error=traceback.format_exc());print(report['error'],flush=True)
finally:
    from go2_pvcnn.control.dh116.tabletop_training import write_json
    report['process_exit_code']=0 if report['status']=='completed' else 2
    write_json(args.output/'summary.json',report)
    write_json(args.output/'monitor.json',dict(status=report['status'],results=report['results'],
        error=report.get('error'),hardware_calibrated=False))
    print('FIXTURE_RESULT',json.dumps(dict(status=report['status'],results=report['results'])),flush=True)
    import omni.kit.app
    omni.kit.app.get_app().post_quit(report['process_exit_code'])
    app.close(wait_for_replicator=False,skip_cleanup=True)
raise SystemExit(report['process_exit_code'])
