"""Single-finger physics probe. Explicit synthetic gains; no hardware claims."""
import argparse
import hashlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--side', choices=('left','right'), default='left')
parser.add_argument('--sweep', action='store_true', help='Six sequential 8-second active joint probes')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise ValueError('refusing to overwrite evidence')
args.output.mkdir(parents=True)
app = AppLauncher(args).app
result = dict(passed=False, physics_stepped=False, hardware_validated=False)
try:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand
    from go2_pvcnn.control.dh116.no_load import reference, metrics, contact_metrics, STIFFNESS, DAMPING, EFFORT_LIMIT, VELOCITY_LIMIT
    from isaaclab.sensors import ContactSensor, ContactSensorCfg

    torch.set_num_threads(1)
    manifest_path = Path(__file__).resolve().parents[1]/'assets/fixed_dual_panda_dh116_v1/manifest.json'
    record = next(r for r in json.loads(manifest_path.read_text())['sources']
                  if r['side']==args.side and r['kind']=='dh116')
    source = Path(record['path'])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != record['sha256']:
        raise ValueError('asset SHA mismatch')
    dt = 1/240
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    cfg = ArticulationCfg(
        prim_path='/World/Hand',
        spawn=sim_utils.UsdFileCfg(usd_path=str(source),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, enabled_self_collisions=True,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(zip(JOINTS,map(float,expand(reference(0)))))),
        actuators={
            'active': ImplicitActuatorCfg(joint_names_expr=list(ACTIVE), stiffness=STIFFNESS,
                damping=DAMPING, effort_limit_sim=EFFORT_LIMIT, velocity_limit_sim=VELOCITY_LIMIT),
            'passive': ImplicitActuatorCfg(joint_names_expr=[n for n in JOINTS if n not in ACTIVE],
                stiffness=0., damping=0., effort_limit_sim=0., velocity_limit_sim=VELOCITY_LIMIT)},
    )
    hand = Articulation(cfg)
    contacts = ContactSensor(ContactSensorCfg(prim_path='/World/Hand/.*', update_period=0.0))
    # Independent positive control, two metres from the hand: falling ball on
    # a static block. Never use zero hand readings alone to qualify the sensor.
    block_cfg = sim_utils.CuboidCfg(size=(.2,.2,.1), collision_props=sim_utils.CollisionPropertiesCfg())
    block_cfg.func('/World/ContactControlBlock', block_cfg, translation=(2.,0.,.7))
    ball = RigidObject(RigidObjectCfg(prim_path='/World/ContactControlBall',
        spawn=sim_utils.SphereCfg(radius=.025, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.01)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.,0.,.8))))
    control_sensor = ContactSensor(ContactSensorCfg(prim_path='/World/ContactControlBall',update_period=0.0))
    sim.reset()
    order = [hand.joint_names.index(n) for n in JOINTS]
    active_ids = [hand.joint_names.index(n) for n in ACTIVE]
    hand.write_joint_state_to_sim(hand.data.default_joint_pos.clone(), torch.zeros_like(hand.data.default_joint_vel))
    hand.reset()
    contacts.reset()
    control_sensor.reset()
    control_peak = 0.
    control_first = None
    if set(contacts.body_names) != set(hand.body_names):
        raise ValueError('contact sensor body coverage differs from articulation')
    rows = []
    total_steps = 1920 * (6 if args.sweep else 1)
    for step in range(total_steps):
        t = (step+1)*dt
        index = step//1920 if args.sweep else 2
        q = reference((step % 1920+1)*dt, active_index=index)
        hand.set_joint_position_target(torch.tensor(q,device=args.device,dtype=torch.float32)[None], joint_ids=active_ids)
        hand.write_data_to_sim()
        sim.step(render=False)
        hand.update(dt)
        contacts.update(dt, force_recompute=True)
        control_sensor.update(dt, force_recompute=True)
        control_force = float(torch.linalg.vector_norm(control_sensor.data.net_forces_w).item())
        if not np.isfinite(control_force):
            raise ValueError('invalid positive-control contact data')
        if control_first is None:
            control_first = control_force
        control_peak = max(control_peak, control_force)
        result['physics_stepped'] = True
        full = hand.data.joint_pos[0,order].detach().cpu().numpy()
        vel = hand.data.joint_vel[0,order].detach().cpu().numpy()
        m = metrics(full, vel, q)
        c = contact_metrics(contacts.data.net_forces_w[0].detach().cpu().numpy(), len(hand.body_names))
        rows.append(dict(step=step,time=t,active_joint=ACTIVE[index],q=full.tolist(),contact=c,**m))
        if (step+1) % 1920 == 0:
            print('DH116_JOINT_COMPLETE', ACTIVE[index], step+1, flush=True)
        if not m['passed'] or not c['passed']:
            result['failure'] = rows[-1]
            break
    result.update(side=args.side, asset_sha256=digest, steps=len(rows),
        passed=len(rows)==total_steps and all(r['passed'] and r['contact']['passed'] for r in rows)
            and control_first <= .01 and control_peak > .01,
        contact_positive_control=dict(first_n=control_first, peak_n=control_peak,
                                      passed=control_first <= .01 and control_peak > .01),
        sweep=args.sweep, contact_body_names=contacts.body_names,
        max_net_contact_n=max(r['contact'].get('max_net_contact_n',0) for r in rows),
        gains=dict(stiffness=STIFFNESS,damping=DAMPING,effort_nm=EFFORT_LIMIT,velocity_rad_s=VELOCITY_LIMIT),
        gravity=False, self_collision_enabled=True, contact_clearance_validated=False,
        max_tracking_rad=max(r.get('tracking_rad',0) for r in rows),
        max_coupling_rad=max(r.get('coupling_rad',0) for r in rows))
    with (args.output/'trajectory.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
except Exception as exc:
    import traceback
    traceback.print_exc()
    result['error'] = repr(exc)
    raise
finally:
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DH116_NO_LOAD',json.dumps(result),flush=True)
    # Evidence is durable before Kit teardown; stage stop/close can deadlock
    # with active PhysX contact views in this installed build.
    app.close(wait_for_replicator=False, skip_cleanup=True)
