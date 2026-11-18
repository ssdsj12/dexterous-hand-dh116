"""Diagnostic digit-contact position servo around a verified pinch seed.

Reuses the existing physical driver in this process only. No requested force,
extra torque, attachment, or PPO update; lift remains scheduled independently.
"""
import argparse
import hashlib
from pathlib import Path
import runpy
import sys
import torch
from go2_pvcnn.control.dh116 import tabletop_grip_lift as program
from go2_pvcnn.control.dh116 import tabletop_tensor_world as adapter
from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
from go2_pvcnn.control.dh116.tabletop_surface_kinematics import pose_matrix
from go2_pvcnn.control.dh116.tabletop_surface_objectives import finite_cylinder_sdf
from go2_pvcnn.control.dh116.tabletop_pinch_reference import contact_seek,manifold_depth

diagnostic_parser=argparse.ArgumentParser(add_help=False)
diagnostic_parser.add_argument('--depth-feedback',choices=('source','native'),default='source')
diagnostic_parser.add_argument('--contact-statistic',choices=('mean','minimum'),default='mean')
diagnostic_parser.add_argument('--hold-only',action='store_true')
diagnostic_options,remaining=diagnostic_parser.parse_known_args()
sys.argv=[sys.argv[0],*remaining]

ROOT=Path(__file__).resolve().parents[1]
state={};samples=[]
original_init=adapter.TensorWorld.__init__;original_advance=adapter.TensorWorld.advance
original_reset=adapter.TensorWorld.reset;original_program=program.advance_program
original_write=training.write_json


def initialize(self,world):
    original_init(self,world);state['world']=self
    self.seek_history=torch.zeros(8,self.n,2,2,device=self.device,dtype=torch.float64)
    self.seek_count=torch.zeros(self.n,device=self.device,dtype=torch.long)
    self.seek_depth=torch.zeros(self.n,2,2,device=self.device,dtype=torch.float64)
    self.seek_source_depth=torch.zeros_like(self.seek_depth)
    self.seek_native_observed=torch.zeros_like(self.seek_depth,dtype=torch.bool)
    self.seek_feedback=torch.zeros_like(self.seek_depth);self.seek_tick=0;self.seek_dirty=True


def reset(self,ids):
    measured=original_reset(self,ids)
    if hasattr(self,'seek_history'):
        self.seek_history[:,ids]=0.;self.seek_count[ids]=0;self.seek_depth[ids]=0.;self.seek_dirty=True
    return measured


def advance(self,task,allowed,live):
    # Preserve the exact inputs used to choose the current command.
    samples.append(dict(feedback_digit_load_n=self.seek_feedback.clone(),decision_digit_depth_m=self.seek_depth.clone(),
        source_digit_depth_m=self.seek_source_depth.clone(),native_contacts_observed=self.seek_native_observed.clone()))
    measured=original_advance(self,task,allowed,live)
    self.seek_history[self.seek_tick%8]=measured['grouped_object_loads'][:,:,:2]
    self.seek_count=(self.seek_count+1).clamp_max(8);self.seek_tick+=1
    return measured


def targets(command,seed,closed,elapsed_s,dt,**options):
    world=state['world'];_,height=original_program(command,seed,closed,elapsed_s,dt,**options)
    if world.seek_dirty or world.seek_tick%world.decimation==0:
        world.seek_source_depth.zero_()
        for h,body,local in world.shapes:
            name=world.hands[h].body_names[body]
            digit=0 if name.startswith('finger1') else (1 if name.startswith('finger2') else None)
            if digit is None:continue
            transform=pose_matrix(world.poses[:,h,body])
            points=local[None]@transform[:,:3,:3].transpose(-1,-2)+transform[:,:3,3][:,None]
            depth=(-finite_cylinder_sdf(points,world.wp,world.wq)).clamp_min(0).amax(-1)
            world.seek_source_depth[:,h,digit]=torch.maximum(world.seek_source_depth[:,h,digit],depth)
        world.seek_dirty=False
    if diagnostic_options.depth_feedback=='native':
        world.seek_depth.zero_();world.seek_native_observed.zero_()
        for i,sensor in enumerate(world.w.sensors):
            name=world.w.sensor_links[i%len(world.w.sensor_links)]
            digit=0 if name.startswith('finger1') else (1 if name.startswith('finger2') else None)
            if digit is None:continue
            hand=i//len(world.w.sensor_links)
            _,_,_,separation,count,start=sensor.contact_physx_view.get_contact_data(dt=dt)
            observed=count.reshape(-1)>0
            # The native buffer may still contain the preceding episode's last
            # contact at reset. Wait for this episode's first real physics step.
            fresh=world.seek_count>0
            count=torch.where(fresh,count.reshape(-1),0)
            start=torch.where(fresh,start.reshape(-1),0)
            depth=manifold_depth(separation.reshape(-1).double(),count,start)
            world.seek_depth[:,hand,digit]=torch.maximum(world.seek_depth[:,hand,digit],depth)
            world.seek_native_observed[:,hand,digit] |= observed&fresh
    else:world.seek_depth.copy_(world.seek_source_depth)
    if diagnostic_options.contact_statistic=='minimum':
        world.seek_feedback=world.seek_history.amin(0)
    else:world.seek_feedback=world.seek_history.sum(0)/world.seek_count.clamp_min(1)[:,None,None]
    command=contact_seek(command,seed,world.seek_feedback,world.seek_depth,dt)
    return command,torch.zeros_like(height) if diagnostic_options.hold_only else height


def write(path,data):
    path=Path(path)
    if path.name=='resolved_grip_lift.json':
        data.update(mode='right_pinch_contact_presence_position_seek',force_controller_executed=True,
            force_control_mode='binary_contact_presence;no_force_goal',force_goal=None,
            contact_seek=dict(rate_rad_s=.02,contact_presence_n=.1,max_digit_depth_m=.0008,
                active_indices=[1,2,7,8],geometry_sampling_hz=60,load_history_raw_steps=8,
                lift_admission_gate=False,depth_feedback=diagnostic_options.depth_feedback,
                native_sampling_hz=1/state['world'].dt if diagnostic_options.depth_feedback=='native' else None,
                contact_statistic=diagnostic_options.contact_statistic,
                missing_manifold='unknown;zero_observed_depth_is_not_full_geometry_clearance'),
            diagnostic_entry_lf_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
            contact_seek_helper_lf_sha256=hashlib.sha256((ROOT/'src/go2_pvcnn/control/dh116/tabletop_pinch_reference.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
        if diagnostic_options.hold_only:
            data.update(mode='right_pinch_stationary_force_hold_probe',lift_goal_m=0.,lift_speed_m_s=0.,
                acceptance='measure_force_stability;no_lift_attempt')
    elif path.name=='initial_cases.json':
        for row in data:
            row['nominal_closed_targets']=row.pop('closed_targets');row['controller']='contact_seek'
    elif path.name.startswith('lift_case_'):
        data.update(use='replay_stateful_contact_seek_controller;not_fixed_joint_program;not_plain_PPO_BC',
            replay_validation='not_executed',
            replay_requires_original_reset_schedule=data['first_qualified']['episode']>0)
        data['first_qualified']['replay_stop']='replay_exact_contact_seek_controller_for_episode_physics_steps;then_keep_controller_running'
        if data['first_qualified']['episode']>0:
            data['first_qualified']['replay_stop']='replay_original_reset_schedule_and_sampling_phase;fresh_episode_replay_not_verified'
    elif path.name=='grip_lift_result.json':
        if samples:
            torch.save({key:torch.stack([row[key] for row in samples]).detach().cpu() for key in samples[0]},path.parent/'contact_seek_trace.pt')
        data.update(controller='right_pinch_contact_presence_position_seek',force_goal=None)
        if diagnostic_options.hold_only:data.update(force_hold_probe=True,commanded_wrist_lift_m=0.)
    return original_write(path,data)


adapter.TensorWorld.__init__=initialize;adapter.TensorWorld.reset=reset;adapter.TensorWorld.advance=advance
program.advance_program=targets;training.write_json=write
runpy.run_path(str(ROOT/'scripts/run_dh116_grip_lift.py'),run_name='__main__')
