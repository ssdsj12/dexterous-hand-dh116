"""Feedback/history-aware distillation of measured v2 contact decisions.

The actor predicts the next bounded command, not an unvalidated complete
grasp plan. Teacher-only geometry/dynamics never enter student features.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contact_collection import _observation
from .contact_imitation import ContactImitationPolicy


def _frame(row):
    o=_observation(row['observation'])
    mask=np.tile([1.,1.,0.] if o['goal']['contact_mode']=='pinch' else [1.,1.,1.],2)
    relative=(np.asarray(o['target_points_w'])-o['pad_positions_w']).reshape(6,3)
    phase=np.array([row['phase']==p for p in ('close','hold','lift')],dtype=float)
    if not phase.any():raise ValueError('close/hold/lift feature phase required')
    return np.concatenate((np.asarray(o['hand_q']).ravel(),np.asarray(row['previous_action']).reshape(12),
        np.asarray(o['hand_qd']).ravel()*10.,(relative*mask[:,None]).ravel()*1000.,
        np.asarray(o['contact_loads_n'])*mask,np.asarray(o['contact_gaps_m']).ravel()*mask*1000.,
        o['contact_valid'],o['contact_valid_known'],np.asarray(o['target_loads_n'])*mask,phase,
        [o['goal']['lift_height_m']*10.])).astype(np.float32)


def complete_features(row,history,*,rod_nodes):
    """Three 88D observed frames plus history mask; full current rod for teacher.

    Inputs currently originate in simulation, including pad-to-wire geometry.
    This feature contract does not assert an installed real-world vision or
    tactile estimator. No future outcome or Panda joint is read here.
    """
    frames=list(history[-2:])+[row]
    known=np.r_[np.zeros(3-len(frames)),np.ones(len(frames))]
    frames=[frames[0]]*(3-len(frames))+frames
    student=np.concatenate([*(_frame(r) for r in frames),known]).astype(np.float32)
    o=_observation(row['observation'])
    if np.asarray(o['rod_position']).shape!=(3,rod_nodes):
        raise ValueError('complete rod node count differs from checkpoint')
    privileged=[]
    for key,scale in (('palm_poses_w',1.),('pad_positions_w',10.),('target_points_w',10.),
        ('pad_jacobians',10.),('contact_normals_w',1.),('rod_position',10.),
        ('rod_velocity',10.),('rod_director',1.),('rod_omega',.1),('rod_radius_m',1000.)):
        privileged.append(np.asarray(o[key]).ravel()*scale)
    # The nearest four constraints have an explicit occupancy mask. Full raw
    # source rows remain in replay; padding is not a fabricated measurement.
    constraints=np.zeros((4,13));present=np.zeros(4)
    for slot,index in enumerate(np.argsort(o['clearances_m'])[:4]):
        constraints[slot]=np.r_[o['clearances_m'][index]*1000.,
                               np.asarray(o['clearance_jacobians'][index])*10.]
        present[slot]=1.
    teacher=np.concatenate((student,*privileged,constraints.ravel(),present)).astype(np.float32)
    if student.shape!=(267,) or teacher.shape!=(486+19*rod_nodes,):
        raise ValueError('complete contact feature dimensions mismatch')
    return student,teacher


class CompleteContactPolicy(ContactImitationPolicy):
    requires_complete_observation=True

    def __init__(self,*,role,control_dt,rod_nodes,device='cpu'):
        super().__init__(role=role,control_dt=control_dt,device=device)
        if not isinstance(rod_nodes,int) or rod_nodes<2:raise ValueError('rod node count required')
        self.rod_nodes=rod_nodes;self.history=[]
        self.input_dim=267 if role=='student' else 486+19*rod_nodes
        self.net=nn.Sequential(nn.Linear(self.input_dim,128),nn.SiLU(),nn.Linear(128,128),
                               nn.SiLU(),nn.Linear(128,12),nn.Tanh()).to(self.device)

    def reset(self):
        """Clear recurrent observation history at the start of an episode."""
        self.history.clear()

    @torch.no_grad()
    def plan_contact(self,*,active_joints,previous_action,complete_observation=None,phase='close',
                     pad_jacobians=None,**unused):
        if complete_observation is None:raise ValueError('complete measured observation is required')
        if not np.allclose(np.asarray(active_joints),complete_observation['hand_q'],atol=1e-7,rtol=0):
            raise ValueError('complete observation does not match current active joints')
        row=dict(observation=complete_observation,previous_action=previous_action,phase=phase)
        student,teacher=complete_features(row,self.history,rod_nodes=self.rod_nodes)
        self.net.eval()
        delta=self.net(torch.as_tensor(teacher if self.role=='teacher' else student,
                                      device=self.device))*self.max_step
        if pad_jacobians is not None:
            j=torch.as_tensor(pad_jacobians,dtype=delta.dtype,device=self.device)
            if j.shape!=(2,3,3,6) or not bool(torch.isfinite(j).all()):
                raise ValueError('hand-only pad Jacobians required')
            delta*=((j.abs().sum((1,2))>1e-12).reshape(12))
        previous=torch.as_tensor(previous_action,dtype=delta.dtype,device=self.device)
        action=(previous+delta).clamp(self.lower,self.upper)
        self.history=(self.history+[deepcopy(row)])[-2:]
        return action.reshape(2,6),dict(model='learned_contact_imitation_v2',role=self.role,
            input_dim=self.input_dim,action_dim=12,control_dt_s=self.control_dt,history_frames=3)

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(schema='dh116-contact-imitation-v2',role=self.role,input_dim=self.input_dim,
            action_dim=12,arm_state_included=False,control_dt_s=self.control_dt,
            rod_nodes=self.rod_nodes,history_frames=3,training_metadata=dict(self.training_metadata),
            state_dict=self.net.state_dict()),path)

    @classmethod
    def load(cls,path,*,expected_role='student',device='cpu'):
        value=torch.load(path,map_location='cpu',weights_only=True)
        if (value.get('schema')!='dh116-contact-imitation-v2' or value.get('role')!=expected_role
                or value.get('action_dim')!=12 or value.get('arm_state_included') is not False
                or value.get('history_frames')!=3):
            raise ValueError('expected complete hand-only contact checkpoint')
        model=cls(role=value['role'],control_dt=value['control_dt_s'],rod_nodes=value['rod_nodes'],device=device)
        if model.input_dim!=value['input_dim']:raise ValueError('complete feature dimensions mismatch')
        model.net.load_state_dict(value['state_dict']);model.net.eval()
        model.training_metadata=dict(value.get('training_metadata',model.training_metadata))
        return model


def fit_complete_pair(episodes,*,control_dt,epochs=500,device='cpu',seed=0):
    """Clone actual submitted expert targets; do not use unexecuted future plans.

    Source qualification belongs to load_complete_contact_run. Episodes stay
    separate so student history cannot leak across runs.
    """
    if not episodes or any(not rows for rows in episodes) or epochs<1:
        raise ValueError('nonempty measured episodes and positive epochs required')
    nodes=np.asarray(episodes[0][0]['observation']['rod_position']).shape[1]
    torch.manual_seed(seed)
    models={r:CompleteContactPolicy(role=r,control_dt=control_dt,rod_nodes=nodes,device=device)
            for r in ('teacher','student')}
    examples=[];labels=[]
    for rows in episodes:
        history=[]
        for row in rows:
            examples.append(complete_features(row,history,rod_nodes=nodes))
            labels.append((np.asarray(row['action'])-row['previous_action'])/models['student'].max_step)
            history=(history+[row])[-2:]
    target=torch.as_tensor(np.asarray(labels),dtype=torch.float32,device=device).clamp(-1.,1.)
    metrics=dict(samples=len(labels),episodes=len(episodes),epochs=epochs,rod_nodes=nodes,
                 action_dim=12,arm_state_included=False,control_dt_s=control_dt)
    for role,column in (('teacher',1),('student',0)):
        x=torch.as_tensor(np.asarray([pair[column] for pair in examples]),device=device)
        net=models[role].net
        with torch.no_grad():metrics[role+'_loss_before']=float((net(x)-target).square().mean())
        if role=='student':
            privileged=torch.as_tensor(np.asarray([pair[1] for pair in examples]),device=device)
            with torch.no_grad():desired=.5*target+.5*models['teacher'].net(privileged)
        else:desired=target
        optimizer=torch.optim.Adam(net.parameters(),lr=.003)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss=(net(x)-desired).square().mean();loss.backward();optimizer.step()
        net.eval()
        with torch.no_grad():metrics[role+'_loss_after']=float((net(x)-target).square().mean())
    return models,metrics
