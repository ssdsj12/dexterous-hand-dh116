"""Small teacher/student behavior-cloning policies for physical contact MPC."""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contact_replay import contact_features
from .linear_coupling import active_limits


class ContactImitationPolicy:
    """Predict a bounded hand-command residual relative to previous_action."""

    def __init__(self,*,role,control_dt,device='cpu'):
        if role not in ('teacher','student'):raise ValueError('role must be teacher or student')
        if not math.isfinite(control_dt) or control_dt<=0:raise ValueError('control_dt must be positive')
        self.role=role;self.control_dt=float(control_dt);self.device=torch.device(device)
        self.training_metadata=dict(training_scope='unspecified',validated_source_grasp_lift=False)
        self.input_dim=78 if role=='teacher' else 42
        self.max_step=.3*self.control_dt
        self.net=nn.Sequential(nn.Linear(self.input_dim,64),nn.SiLU(),nn.Linear(64,64),
                               nn.SiLU(),nn.Linear(64,12),nn.Tanh()).to(self.device)
        lower,upper=active_limits()
        lo=lower.astype(np.float32);hi=upper.astype(np.float32)
        lo=np.where(lo.astype(float)<lower,np.nextafter(lo,np.full_like(lo,np.inf)),lo)
        hi=np.where(hi.astype(float)>upper,np.nextafter(hi,np.full_like(hi,-np.inf)),hi)
        self.lower=torch.as_tensor(np.tile(lo,2),device=self.device)
        self.upper=torch.as_tensor(np.tile(hi,2),device=self.device)

    @torch.no_grad()
    def plan_contact(self,*,active_joints,previous_action,pad_positions,target_points,pad_jacobians=None,
                     actuator_load_offset=None,clearances=None,clearance_jacobians=None):
        student,teacher=contact_features(dict(observed_active_joints=active_joints,
            previous_action=previous_action,pad_positions_w=pad_positions,target_points_w=target_points))
        observation=teacher if self.role=='teacher' else student
        self.net.eval()
        delta=self.net(torch.as_tensor(observation,device=self.device))*self.max_step
        if pad_jacobians is not None:
            jac=torch.as_tensor(pad_jacobians,dtype=delta.dtype,device=self.device)
            if jac.shape!=(2,3,3,6) or not bool(torch.isfinite(jac).all()):
                raise ValueError('hand-only pad Jacobians must have shape (2,3,3,6)')
            delta*=((jac.abs().sum((1,2))>1e-12).reshape(12))
        previous=torch.as_tensor(student[12:24],device=self.device)
        action=(previous+delta).clamp(self.lower,self.upper)
        return action.reshape(2,6),dict(model='learned_contact_imitation',role=self.role,
            input_dim=self.input_dim,action_dim=12,control_dt_s=self.control_dt)

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(schema='dh116-contact-imitation-v1',role=self.role,
            input_dim=self.input_dim,action_dim=12,arm_state_included=False,
            control_dt_s=self.control_dt,training_metadata=dict(self.training_metadata),
            state_dict=self.net.state_dict()),path)

    @classmethod
    def load(cls,path,*,expected_role='student',device='cpu'):
        value=torch.load(path,map_location='cpu',weights_only=True)
        if (value.get('schema')!='dh116-contact-imitation-v1' or value.get('action_dim')!=12
                or value.get('arm_state_included') is not False or value.get('role')!=expected_role):
            raise ValueError(f'expected hand-only {expected_role} contact imitation checkpoint')
        model=cls(role=value['role'],control_dt=value['control_dt_s'],device=device)
        if model.input_dim!=value.get('input_dim'):raise ValueError('contact feature dimensions mismatch')
        model.net.load_state_dict(value['state_dict']);model.net.eval()
        model.training_metadata=dict(value.get('training_metadata',model.training_metadata))
        return model


def load_mounted_policy(path,*,control_dt,device='cpu',role='student'):
    """Load current neural contact policies or explicit legacy JSON policies."""
    path=Path(path)
    if path.suffix=='.json':
        from .teacher_student import LinearImitationPolicy
        return LinearImitationPolicy.load(path,expected_role=role)
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value.get('schema')=='dh116-contact-imitation-v2':
        from .complete_contact_policy import CompleteContactPolicy
        model=CompleteContactPolicy.load(path,expected_role=role,device=device)
    else:model=ContactImitationPolicy.load(path,expected_role=role,device=device)
    if not math.isclose(model.control_dt,control_dt,rel_tol=0.,abs_tol=1e-12):
        raise ValueError('contact policy control interval does not match the mounted runner')
    return model


def fit_contact_pair(transitions,*,control_dt,epochs=500,device='cpu',learning_rate=.003,seed=0):
    """Fit a privileged teacher to MPC labels and distill into the student.

    The caller selects successful physical runs before invoking this optimizer.
    Losses are normalized command-residual MSE, not grasp-success evidence.
    """
    if not transitions or any(not row.physics_valid for row in transitions):
        raise ValueError('physical MPC transitions are required')
    if epochs<1 or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('positive training epochs and learning rate required')
    torch.manual_seed(seed)
    models={role:ContactImitationPolicy(role=role,control_dt=control_dt,device=device)
            for role in ('teacher','student')}
    target_device=models['student'].device
    student=torch.as_tensor(np.array([row.obs_student for row in transitions],dtype=np.float32),device=target_device)
    teacher=torch.as_tensor(np.array([row.obs_teacher_privileged for row in transitions],dtype=np.float32),device=target_device)
    if student.shape!=(len(transitions),42) or teacher.shape!=(len(transitions),78):
        raise ValueError('expected physical contact replay with 42 student/78 teacher inputs')
    previous=torch.as_tensor(np.array([row.previous_action for row in transitions],dtype=np.float32),device=target_device)
    actions=torch.as_tensor(np.array([row.mpc_action for row in transitions],dtype=np.float32),device=target_device)
    labels=((actions-previous)/models['student'].max_step).clamp(-1.,1.)
    metrics=dict(samples=len(transitions),epochs=epochs,control_dt_s=control_dt,
                 action_dim=12,arm_state_included=False)
    for role,features in (('teacher',teacher),('student',student)):
        net=models[role].net
        with torch.no_grad():metrics[role+'_loss_before']=float((net(features)-labels).square().mean())
        target=labels
        if role=='student':
            with torch.no_grad():target=.5*labels+.5*models['teacher'].net(teacher)
        optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
        net.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss=(net(features)-target).square().mean()
            loss.backward();optimizer.step()
        net.eval()
        with torch.no_grad():metrics[role+'_loss_after']=float((net(features)-labels).square().mean())
    return models,metrics
