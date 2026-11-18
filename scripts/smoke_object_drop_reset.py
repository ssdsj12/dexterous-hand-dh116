"""Isolated Isaac regression: drop one object, verify reset and continue PPO."""
import json
from pathlib import Path
import runpy
import sys
import torch
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv

output=Path('runs/object_drop_reset_8env_smoke_v2')
original=TensorEnv.step
injected=False


def step(self,actions):
    global injected
    check=not injected
    if check:
        injected=True
        ids=torch.tensor([0],device=self.device,dtype=torch.long)
        pose=self.world.wire.data.root_pose_w[ids].clone()
        pose[:,0]+=.8  # Off this environment's tabletop, away from hands.
        pose[:,2]-=.02
        self.world.wire.write_root_pose_to_sim(pose,env_ids=ids)
        self.world.w.sim.physics_sim_view.update_articulations_kinematic()
    result=original(self,actions)
    if check:
        _,_,done,extra=result
        assert done[0] and not done[1:].any()
        assert int(self.object_drop_resets)==1
        assert extra['terms']['failure'][0]==-25.
        assert self.world.measure()['bottom'][0]>=.29
        proof=dict(only_env_zero_reset=True,object_drop_resets=int(self.object_drop_resets),
                   failure_penalty=float(extra['terms']['failure'][0]),
                   post_reset_bottom_m=float(self.world.measure()['bottom'][0]),
                   training_continues=True)
        (output/'drop_reset_validation.json').write_text(json.dumps(proof,indent=2))
        print('OBJECT_DROP_RESET_VALIDATED',json.dumps(proof),flush=True)
    return result


TensorEnv.step=step
sys.argv=['scripts/train_dh116_tabletop_gpu_ppo.py',
    '--pose-summary','configs/dh116_training/pose_reference.json',
    '--adaptive-grasp','--force-feedback','--full-hand-contact',
    '--joint-envelope','configs/dh116_joint_envelopes/free_grasp_geometry_v1.json',
    '--grasp-initialization','configs/dh116_training/grasp_initialization_below_axis_2mm_v1.json',
    '--require-initial-below-axis','--load-profile','configs/dh116_load_profiles/v2_pressure_12n_instant_candidate.json',
    '--object-kind','solid_cylinder','--object-material','extreme_grip','--hand-material','grip_compliant',
    '--num-envs','8','--rollout-steps','24','--max-iterations','1',
    '--physics-hz','480','--mimic-frequency','200','--position-iterations','64','--velocity-iterations','16',
    '--reset-height-jitter-m','0','--cpu-threads','6','--headless','--device','cuda:0',
    '--direct-training','--tensor-pipeline','--training-episode-resets','--pressure-relief',
    '--grasp-controller','policy','--palm-transport','driven_carriage','--palm-exploration-axis','xyz',
    '--passive-velocity-limit-rad-s','2.97','--contact-reward-window','close_start',
    '--grasp-readiness','measured_load','--skip-hold','--close-downward-penalty','--grasp-shaping',
    '--vertical-force-reward','--vertical-force-reward-mode','continuous',
    '--force-reward-start','downward_after_close','--grasp-acquisition-shaping',
    '--free-grasp-lift','--stable-lift-reward','--force-region-control','--phase-force-reward',
    '--output',str(output),'--kit_args=--allow-root']
runpy.run_path(sys.argv[0],run_name='__main__')
