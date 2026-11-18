"""Fail-closed teacher qualification from actual geometry and measured loads.

This is offline acceptance, never an admission gate for the PPO controller.
Candidate optimization and wrist movement alone cannot publish teacher data.
"""
import json
import hashlib
import math
from pathlib import Path
import re
import torch
from .linear_coupling import active_limits

REQUIRED_IMPLEMENTATIONS=tuple('src/go2_pvcnn/control/dh116/'+name for name in (
    'tabletop_surface_candidates.py','tabletop_surface_kinematics.py','tabletop_surface_objectives.py',
    'tabletop_surface_teacher.py','tabletop_surface_teacher_control.py','tabletop_surface_support.py',
    'tabletop_surface_force_goals.py','tabletop_grip_lift.py','tabletop_tensor_env.py',
    'tabletop_tensor_task.py','tabletop_tensor_world.py','tabletop_free_wrist_grasp.py','tabletop_policy_control.py'))+(
    'scripts/generate_dh116_surface_candidates.py','scripts/generate_dh116_phalanx_candidates.py',
    'scripts/generate_dh116_initial_pinch_candidate.py',
    'scripts/run_dh116_grip_lift.py',
    'scripts/search_dh116_surface_teacher.py')
QUALIFIER_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def implementation_identity(root):
    root=Path(root)
    return {name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for name in REQUIRED_IMPLEMENTATIONS}

CONTRACT=dict(version='dh116_surface_teacher_v2_force_tracking_support',active_dof_count=12,
    passive_dof_count=10,finger_dof_count=22,action_size=26,observation_size=261,
    table_z_m=.30,radius_m=.008,length_m=.5,minimum_height_m=.01,
    minimum_digit_normal_load_n=.05,hand_normal_force_range_n=[2.,12.],
    maximum_digit_ratio=1.5,maximum_linear_speed_m_s=.03,hold_s=.3,
    maximum_sampled_penetration_m=.001,maximum_bottom_disagreement_m=1e-6,
    fault_or_reset='reject_entire_candidate',geometry_only_qualified=False,
    original_formal_task_success='unchanged_30mm_and_0.5s')
CONTRACT.update(force_tracking_tolerance='max(0.1N,0.05*current_requested_hand_normal_force)',
    minimum_measured_total_support_weight_ratio=.9,force_goals='explored2to12N_not_fixed')

SHAPES=dict(object_pose_w=(7,),bottom=(),digit_normal_load_n=(2,2),hand_normal_load_n=(2,),
    object_velocity=(6,),hand_clearance_m=(2,),object_penetration_m=(),self_penetration_m=(),
    fault=(),reset=(),requested_hand_force_n=(2,),object_weight_n=(),vertical_components_n=(2,2),actions=(26,))
PHYSICS_SHAPES=dict(physics_object_pose_w=(7,),physics_bottom=(),physics_digit_normal_load_n=(2,2),
    physics_hand_normal_load_n=(2,),physics_object_velocity=(6,),physics_fault=(),
    physics_requested_hand_force_n=(2,),physics_object_weight_n=(),physics_vertical_components_n=(2,2))


def physics_interval_evidence(m,n):
    """Recheck every recorded substep; cached qualification flags are insufficient."""
    if 'physics_bottom' not in m or m['physics_bottom'].ndim!=2:raise ValueError('complete raw physics evidence required')
    samples=m['physics_bottom'].shape[1]
    if samples<1:raise ValueError('nonempty raw physics interval required')
    finite=torch.ones((n,samples),device=m['physics_bottom'].device,dtype=torch.bool)
    for key,shape in PHYSICS_SHAPES.items():
        value=m.get(key)
        if not isinstance(value,torch.Tensor) or value.shape!=(n,samples,*shape):
            raise ValueError('raw physics evidence shape mismatch: '+key)
        finite &= torch.isfinite(value).reshape(n,samples,-1).all(-1)
    pose=m['physics_object_pose_w'];q=pose[...,3:];norm=q.norm(dim=-1)
    valid_quaternion=(norm-1).abs()<1e-3
    q=q/norm[...,None].clamp_min(1e-12);qw,qx,qy,qz=q.unbind(-1)
    az=(2*(qy*qz+qw*qx)).abs().clamp(0,1)
    bottom=pose[...,2]-.25*az-.008*(1-az.square()).clamp_min(0).sqrt()
    coherent=(bottom-m['physics_bottom']).abs()<=1e-6
    invalid=~finite|~valid_quaternion|~coherent|m['physics_fault'].bool()
    loads=m['physics_digit_normal_load_n'];hands=m['physics_hand_normal_load_n']
    requested=m['physics_requested_hand_force_n'];weight=m['physics_object_weight_n']
    invalid |= (weight<=0)|((requested<2)|(requested>12)).any(-1)
    executed=2+5*(m['actions'][:,-2:]+1)
    invalid |= ((requested-executed[:,None]).abs()>1e-6).any(-1)
    good=~invalid&(bottom>=.31-1e-10)&(m['physics_object_velocity'][...,:3].norm(dim=-1)<.03)
    good &= (loads>.05).flatten(2).all(-1)&(loads.amax(-1)<=1.5*loads.amin(-1)).all(-1)
    good &= ((hands>=2)&(hands<=12)).all(-1)&(loads.sum(-1)<=hands+1e-6).all(-1)
    tolerance=torch.maximum(torch.full_like(requested,.1),.05*requested)
    good &= ((hands-requested).abs()<=tolerance).all(-1)
    good &= m['physics_vertical_components_n'].sum((-1,-2))>=.9*weight
    return good.all(-1),invalid.any(-1)


def initial_geometry_mask(hand_in_object,object_in_hand,self_depth,clearances):
    n=hand_in_object.shape[0]
    if any(v.shape!=(n,) for v in (hand_in_object,object_in_hand,self_depth)) or clearances.shape!=(n,2):
        raise ValueError('matching measured initial depths and hand clearances required')
    valid=torch.isfinite(clearances).all(-1)&(clearances>=.00029).all(-1)
    for value in (hand_in_object,object_in_hand,self_depth):
        valid &= torch.isfinite(value)&(value>=0)&(value<=.0003)
    return valid


class Qualification:
    def __init__(self,n,device):
        if not isinstance(n,int) or n<=0:raise ValueError('positive candidate count required')
        self.n=n;self.device=torch.device(device)
        self.hold_s=torch.zeros(n,device=device,dtype=torch.float64)
        self.longest_hold_s=torch.zeros_like(self.hold_s)
        self.maximum_height_m=torch.zeros_like(self.hold_s)
        self.poisoned=torch.zeros(n,device=device,dtype=torch.bool)
        self.qualified=torch.zeros_like(self.poisoned)
        self.last_good=torch.zeros_like(self.poisoned)

    def update(self,m,dt):
        if not math.isfinite(dt) or not 0<dt<=1/60+1e-12:raise ValueError('positive physics/control dt at most1/60 required')
        finite=torch.ones(self.n,device=self.device,dtype=torch.bool)
        for key,shape in SHAPES.items():
            value=m.get(key)
            if not isinstance(value,torch.Tensor) or value.shape!=(self.n,*shape) or value.device!=self.device:
                raise ValueError('raw evidence shape/device mismatch: '+key)
            finite &= torch.isfinite(value).reshape(self.n,-1).all(-1)
        pose=m['object_pose_w'];quat=pose[:,3:];norm=quat.norm(dim=-1)
        valid_quat=torch.isfinite(norm)&((norm-1).abs()<1e-3)
        q=quat/norm[:,None].clamp_min(1e-12)
        qw,qx,qy,qz=q.unbind(-1)
        az=(2*(qy*qz+qw*qx)).abs().clamp(0,1)
        bottom=pose[:,2]-.25*az-.008*(1-az.square()).clamp_min(0).sqrt()
        coherent=(bottom-m['bottom']).abs()<=CONTRACT['maximum_bottom_disagreement_m']
        geometry=(m['object_penetration_m']>=0)&(m['object_penetration_m']<=.001)
        geometry &= (m['self_penetration_m']>=0)&(m['self_penetration_m']<=.001)
        geometry &= (m['hand_clearance_m']>=0).all(-1)
        invalid=(~finite|~valid_quat|~coherent|~geometry|m['fault'].bool()|m['reset'].bool())
        requested=m['requested_hand_force_n'];weight=m['object_weight_n']
        invalid |= (weight<=0)|((requested<2)|(requested>12)).any(-1)
        invalid |= ((requested-(2+5*(m['actions'][:,-2:]+1))).abs()>1e-6).any(-1)
        self.poisoned |= invalid
        raw_good=torch.ones_like(self.poisoned)
        if any(key in m for key in PHYSICS_SHAPES):
            raw_good,raw_invalid=physics_interval_evidence(m,self.n)
            self.poisoned |= raw_invalid
        height=(bottom-.30).clamp_min(0)
        loads=m['digit_normal_load_n'];hands=m['hand_normal_load_n']
        bearing=(loads>.05).flatten(1).all(-1)
        balance=(loads.amax(-1)<=1.5*loads.amin(-1)).all(-1)
        force=((hands>=2.)&(hands<=12.)).all(-1)
        # Digit loads are a subset of each hand's independently measured total.
        load_consistent=(loads.sum(-1)<=hands+1e-6).all(-1)
        good=(~self.poisoned)&bearing&balance&force&load_consistent&(height>=.01-1e-10)
        good &= m['object_velocity'][:,:3].norm(dim=-1)<.03
        tolerance=torch.maximum(torch.full_like(requested,.1),.05*requested)
        good &= ((hands-requested).abs()<=tolerance).all(-1)
        good &= m['vertical_components_n'].sum((-1,-2))>=.9*weight
        good &= raw_good
        if 'physics_interval_qualified' in m:
            interval=m['physics_interval_qualified']
            if interval.shape!=(self.n,) or interval.device!=self.device or interval.dtype!=torch.bool:
                raise ValueError('physics interval qualification must be N boolean values')
            good &= interval
        self.hold_s=torch.where(good,self.hold_s+dt,0.)
        self.longest_hold_s=torch.maximum(self.longest_hold_s,self.hold_s)
        self.maximum_height_m=torch.maximum(self.maximum_height_m,torch.where(finite&valid_quat&coherent,height,0.))
        self.qualified=(self.qualified|(self.hold_s>=.3-1e-12))&~self.poisoned
        self.last_good=good
        return self.qualified.clone()

    def summary(self):
        return dict(contract=CONTRACT,qualified=self.qualified.cpu().tolist(),
            poisoned=self.poisoned.cpu().tolist(),maximum_height_mm=(self.maximum_height_m*1000).cpu().tolist(),
            longest_qualified_hold_s=self.longest_hold_s.cpu().tolist())


def export_qualified(output,trace,initial_cases,metadata,*,dt):
    """Recompute qualification; write no teacher file when every trial fails."""
    for key,value in (('active_dof_count',12),('passive_dof_count',10),('finger_dof_count',22),
                      ('action_size',26),('observation_size',261)):
        if metadata.get(key)!=value:raise ValueError('teacher metadata mismatch: '+key)
    hashes=metadata.get('source_hashes',{})
    if not hashes or any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in hashes.values()):
        raise ValueError('source hashes required')
    implementation=metadata.get('implementation_hashes',{})
    if (not set(REQUIRED_IMPLEMENTATIONS)<=set(implementation)
            or any(not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None for h in implementation.values())):
        raise ValueError('complete implementation source identity required')
    metadata=dict(metadata,qualification_source_hashes={
        'src/go2_pvcnn/control/dh116/tabletop_surface_teacher.py':QUALIFIER_SOURCE_SHA256})
    physics=metadata.get('physics',{})
    gravity=physics.get('gravity',[]);mass=physics.get('object_mass_kg',0)
    if len(gravity)!=3 or gravity[:2]!=[0.,0.] or not math.isfinite(gravity[2]) or gravity[2]>=0 or not math.isfinite(mass) or mass<=0:
        raise ValueError('original downward gravity and positive object mass required')
    if not trace or 'actions' not in trace or 'observations' not in trace:raise ValueError('actual trajectory required')
    steps,n=trace['actions'].shape[:2]
    if steps<1 or len(initial_cases)!=n:raise ValueError('matching nonempty cases and trajectory required')
    finite_candidates=torch.ones(n,device=trace['actions'].device,dtype=torch.bool)
    for key,value in trace.items():
        if not isinstance(value,torch.Tensor) or value.shape[:2]!=(steps,n):
            raise ValueError('T x N trace required: '+key)
        finite_candidates &= torch.isfinite(value).reshape(steps,n,-1).all(-1).all(0)
    if trace['actions'].shape!=(steps,n,26) or trace['observations'].shape!=(steps,n,261):
        raise ValueError('26D action and261D observation contract required')
    finite_candidates &= (trace['actions'].abs()<=1.).all(-1).all(0)
    substeps=metadata.get('physics_substeps_per_action')
    if not isinstance(substeps,int) or isinstance(substeps,bool) or substeps<1:
        raise ValueError('actual physics substep count required')
    for key,shape in PHYSICS_SHAPES.items():
        if key not in trace or trace[key].shape!=(steps,n,substeps,*shape):
            raise ValueError('complete physics evidence required: '+key)
    if metadata.get('surface_support',{}).get('enabled',False):
        torque=trace.get('physics_support_torque_command_nm')
        if not isinstance(torque,torch.Tensor) or torque.shape!=(steps,n,substeps,2,6):
            raise ValueError('complete raw support torque commands required')
        cap=physics.get('effort_limit_nm')
        if not isinstance(cap,(float,int)) or not math.isfinite(cap) or cap<=0 or cap>.2+1e-8:
            raise ValueError('original support torque budget required')
        finite_candidates &= (torque.abs()<=cap+1e-7).flatten(2).all(-1).all(0)
    low,high=map(torch.as_tensor,active_limits())
    for index,case in enumerate(initial_cases):
        q=torch.as_tensor(case['active_angles'],dtype=torch.float64)
        poses=torch.as_tensor(case['wrist_pose_w'],dtype=torch.float64)
        if (q.shape!=(12,) or not torch.isfinite(q).all() or (q.reshape(2,6)<low).any()
                or (q.reshape(2,6)>high).any() or poses.shape!=(2,7) or not torch.isfinite(poses).all()
                or ((poses[:,3:].norm(dim=-1)-1).abs()>1e-5).any()):
            finite_candidates[index]=False
    qualifier=Qualification(n,trace['actions'].device)
    for i in range(steps):qualifier.update({key:value[i] for key,value in trace.items()},dt)
    qualifier.qualified &= finite_candidates
    selected=qualifier.qualified.nonzero().flatten().cpu().tolist()
    if not selected:return []
    output=Path(output);output.mkdir(parents=True,exist_ok=True);paths=[]
    for index in selected:
        target=output/f'teacher_case_{index:03d}.json';tensor_path=target.with_suffix('.pt')
        if target.exists() or tensor_path.exists():raise FileExistsError('teacher evidence already exists')
        data={key:value[:,index].detach().cpu().contiguous() for key,value in trace.items()}
        tensor_temp=tensor_path.with_suffix('.pt.tmp');torch.save(data,tensor_temp);tensor_temp.replace(tensor_path)
        case=dict(version=CONTRACT['version'],qualified=True,candidate_index=index,metadata=metadata,
            initial_case=initial_cases[index],qualification=qualifier.summary(),trace_file=tensor_path.name,
            observations='actual_pre_action_policy_observation',actions='actual_generated_26D_policy_commands',
            use='teacher_trajectory_or_diverse_initial_case;does_not_lock_PPO_posture')
        if metadata.get('surface_support',{}).get('enabled',False):
            case.update(use='diverse_initial_case_with_recorded_support_executor',
                support_executor_required=True,
                trajectory_commands='26D_policy_commands_plus_recorded_continuous_support_motor_feedforward')
        temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(case,indent=2,allow_nan=False)+'\n');temp.replace(target)
        paths.append(target)
    return paths
