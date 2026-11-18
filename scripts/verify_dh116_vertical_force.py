"""GPU contact-force sign verification using a falling block against a wall."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.output.exists():parser.error('new output file required')
app=AppLauncher(args).app
report=dict(passed=False)
try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject,RigidObjectCfg
    from isaaclab.sensors import ContactSensor,ContactSensorCfg
    torch.set_num_threads(6)
    dt=1/480
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=args.device,
        physx=sim_utils.PhysxCfg(solver_type=0)))
    material=sim_utils.RigidBodyMaterialCfg(static_friction=.2,dynamic_friction=.2,restitution=0.)
    def box(path,size,pos,kinematic=False):
        return RigidObject(RigidObjectCfg(prim_path=path,spawn=sim_utils.CuboidCfg(size=size,
            activate_contact_sensors=True,rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=kinematic,disable_gravity=kinematic,linear_damping=0.,angular_damping=0.),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=.12),physics_material=material),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos)))
    wall=box('/World/Hand',(.1,1.,3.),(0.,0.,0.),True)
    block=box('/World/Wire',(.1,.1,.1),(.1001,0.,.5))
    sensors=[ContactSensor(ContactSensorCfg(prim_path=a,filter_prim_paths_expr=[b],
        track_friction_forces=True,max_contact_data_count_per_prim=32))
        for a,b in [('/World/Hand','/World/Wire'),('/World/Wire','/World/Hand')]]
    sim.reset();wall.reset();block.reset()
    velocity=torch.zeros((1,6),device=args.device);velocity[:,2]=-.1
    block.write_root_velocity_to_sim(velocity)
    force=torch.tensor([[[-2.,0.,0.]]],device=args.device)
    rows=[]
    for step in range(180):
        previous=block.data.root_lin_vel_w.clone()
        block.set_external_force_and_torque(force,torch.zeros_like(force),is_global=True)
        block.write_data_to_sim();sim.step(render=False)
        wall.update(dt);block.update(dt)
        for sensor in sensors:sensor.update(dt,force_recompute=True)
        n=[s.data.force_matrix_w[0,0,0].clone() for s in sensors]
        f=[s.data.friction_forces_w[0,0,0].clone() for s in sensors]
        if step>=60:
            acceleration=(block.data.root_lin_vel_w-previous)[0]/dt
            expected=.12*(acceleration+torch.tensor([0.,0.,9.81],device=args.device))-force[0,0]
            rows.append(torch.cat((-(n[0]+f[0]),n[1]+f[1],expected,-n[0],-f[0])).cpu())
    data=torch.stack(rows).double();avg=data.mean(0)
    reciprocal=(data[:,:3]-data[:,3:6]).abs().max().item()
    balance=(data[:,:3]-data[:,6:9]).abs().max().item()
    report.update(normal_on_object_mean_n=avg[9:12].tolist(),friction_on_object_mean_n=avg[12:15].tolist(),
        hand_to_object_mean_n=avg[:3].tolist(),momentum_balance_mean_n=avg[6:9].tolist(),
        max_action_reaction_error_n=reciprocal,max_momentum_balance_error_n=balance,
        physics_dt_s=dt,samples=len(rows),device=args.device)
    assert torch.isfinite(data).all() and avg[0]>1.5 and avg[2]>.1,report
    assert reciprocal<.03 and balance<.05,report
    report['passed']=True
finally:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('VERTICAL_FORCE_VALIDATION',json.dumps(report),flush=True)
    app.close()
