"""Display the authored rigid DH116 scene, paused; no trained policy or grasp claim."""
import argparse
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--duration', type=float, default=3600.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS

    root = Path(__file__).resolve().parents[1]
    pose = json.loads((root/'configs/dh116_training/grasp_initialization_pos64.json').read_text())
    manifest = json.loads((root/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
    sources = {r['side']:r['path'] for r in manifest['sources'] if r['kind']=='dh116'}
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/480, device=args.device))
    sim.set_camera_view(eye=(1.0, 0.9, 0.85), target=(0.5, 0., 0.34))
    light = sim_utils.DomeLightCfg(intensity=2200.)
    light.func('/World/Light', light)
    table = sim_utils.CuboidCfg(size=(0.8, 1.2, 0.1),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32,0.38,0.44)))
    table.func('/World/Table', table, translation=(0.6,0.,0.25))
    rod = sim_utils.CylinderCfg(radius=0.008, height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85,0.45,0.12)))
    rod.func('/World/Rod', rod, translation=(0.43,0.,0.308),
             orientation=(0.70710678,-0.70710678,0.,0.))
    hands=[]
    for i, side in enumerate(('left','right')):
        q=pose['joint_positions'][i]
        p=pose['roots'][i]
        cfg=ArticulationCfg(prim_path=f'/World/{side}Hand',
            spawn=sim_utils.UsdFileCfg(usd_path=sources[side],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(p[:3]),rot=tuple(p[3:]),
                joint_pos=dict(zip(JOINTS,map(float,q)))),
            actuators={'pose':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=10.,damping=0.2)})
        hands.append(Articulation(cfg))
    sim.reset()
    for hand in hands:
        hand.write_joint_state_to_sim(hand.data.default_joint_pos,torch.zeros_like(hand.data.default_joint_pos))
        hand.update(sim.get_physics_dt())
    sim.render()
    sim.pause()
    print('HAND_DEMO_READY: authored pose, paused, no policy running',flush=True)
    deadline=time.monotonic()+args.duration
    while app.is_running() and time.monotonic()<deadline:
        app.update()
        time.sleep(.01)
finally:
    app.close(wait_for_replicator=False)
