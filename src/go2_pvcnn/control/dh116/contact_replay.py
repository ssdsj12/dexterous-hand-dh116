"""Read measured full-scene contact-MPC evidence for hand policy distillation."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .mpc_feedback import to_numpy
from .teacher_student import make_transition


def contact_features(sample):
    """42 student inputs; teacher additionally sees absolute privileged geometry.

    The shared inputs are measured DH116 q, the previous hand command and
    six relative pad-to-goal displacements in mm. No Panda coordinates enter.
    """
    fields=[]
    for name,shape in (('observed_active_joints',(2,6)),('previous_action',(12,)),
                       ('pad_positions_w',(2,3,3)),('target_points_w',(2,3,3))):
        value=np.asarray(to_numpy(sample[name]),dtype=np.float32)
        if value.shape!=shape or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite with shape {shape}')
        fields.append(value)
    q,previous,points,targets=fields
    student=np.concatenate((q.reshape(12),previous,(targets-points).reshape(18)*1000.))
    teacher=np.concatenate((student,points.reshape(18)*10.,targets.reshape(18)*10.))
    return student,teacher


def load_contact_run(path,*,require_success=True):
    """Select only physical expert decisions; incomplete runs require opt-in."""
    path=Path(path)
    manifest=json.loads((path/'run_manifest.json').read_text())
    config=manifest['run_config']
    if config.get('rod_backend')!='torch_cuda' or config.get('hand_mpc') is not True:
        raise ValueError('expected full-scene CUDA rod with contact MPC')
    summary_path=path/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    outcome=summary.get('grasp_only',summary.get('continuous_grasp',{}))
    success=bool(summary.get('passed') is True and outcome.get('grasp_lift_passed') is True)
    if require_success and not success:
        raise ValueError('successful measured grasp/lift is required for final imitation training')
    transitions=[];phase_counts=Counter();steps=[]
    pinch=('left_thumb','left_index','right_thumb','right_index')
    with (path/'trajectory.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'):break  # A live writer may not have finished the final row.
            row=json.loads(line)
            phase=row.get('grasp_sequence',{}).get('stage','unknown')
            phase_counts[phase]+=1
            sample=row.get('hand_mpc_training_sample')
            if (row.get('passed') is not True or sample is None
                    or sample.get('rod_backend')!='torch_cuda'
                    or sample.get('arm_state_included') is not False
                    or sample.get('planner_diagnostics',{}).get('model')!='linearized_pad_kinematics'
                    or row.get('contact',{}).get('reaction_applied') is not True):
                continue
            student,teacher=contact_features(sample)
            loads=row['contact']['normal_samples']
            load=max(0.,max(float(loads[channel]['normal_load_n']) for channel in pinch))
            rise=min(row['grasp_sequence'].get('station_rise_m',[0.,0.]))
            transitions.append(make_transition(obs_student=student,obs_teacher_privileged=teacher,
                previous_action=sample['previous_action'],mpc_action=sample['mpc_action'],
                action=sample['mpc_action'],reward=float(success),done=False,
                phase='lift' if phase=='lift' else 'grasp',contact_force_n=load,
                wire_lift_m=rise,physics_valid=True))
            steps.append(row['step'])
    if not transitions:raise ValueError('no valid physical contact-MPC expert decisions in this run')
    if summary:transitions[-1]=replace(transitions[-1],done=True)
    return transitions,dict(source_run=str(path.resolve()),rod_backend='torch_cuda',
        grasp_lift_passed=success,source_terminal=bool(summary),phase_counts=dict(phase_counts),
        decision_steps=steps,action_dim=12,student_input_dim=42,teacher_input_dim=78,
        arm_state_included=False,
        control_dt_s=float(config['physics_dt'])*int(config['hand_mpc_update_steps']))


def load_complete_contact_run(path,*,require_success=True):
    """Load v2 measured intervals without reducing away privileged feedback.

    This is distinct from the v1 42/78-feature diagnostic replay. A passed
    summary alone cannot qualify missing hold/lift data or student labels.
    """
    from .contact_collection import _array,_observation,_state
    path=Path(path)
    manifest=json.loads((path/'run_manifest.json').read_text())
    config=manifest['run_config']
    if (config.get('rod_backend')!='torch_cuda' or config.get('hand_mpc') is not True
            or config.get('hand_mpc_policy') is not None):
        raise ValueError('complete expert replay requires actual CUDA rod and online MPC, not learned labels')
    control_dt=float(config['physics_dt'])*int(config['hand_mpc_update_steps'])
    summary_file=path/'summary.json'
    summary=json.loads(summary_file.read_text()) if summary_file.exists() else {}
    result=summary.get('grasp_only',summary.get('continuous_grasp',{}))
    source_success=bool(summary.get('passed') is True and result.get('grasp_lift_passed') is True
                        and result.get('whole_wire_clear') is True)
    if require_success and not source_success:
        raise ValueError('successful measured grasp/lift and whole-wire clearance are required for complete expert replay')
    records=[];counts=Counter();previous=None
    with (path/'contact_decisions.jsonl').open() as stream:
        for line in stream:
            if not line.endswith('\n'):break
            row=json.loads(line)
            phase=row.get('phase')
            if (row.get('schema')!='dh116-contact-decision-v2'
                    or row.get('action_dim')!=12 or row.get('arm_state_included') is not False
                    or phase not in ('close','hold','lift')):
                raise ValueError('complete hand-only decision schema required')
            expected_owner='hand_mpc' if phase=='close' else 'hand_mpc_hold'
            if row.get('owner')!=expected_owner:
                raise ValueError('expert labels must come from MPC or explicit hand hold')
            obs=_observation(row['observation'])
            _array(row['previous_action'],(12,),'previous_action')
            _array(row['action'],(12,),'action')
            outcome=row.get('outcome')
            if (not isinstance(outcome,dict) or outcome['step']<=row['step']
                    or outcome['time_s']<=row['time_s']
                    or not np.isclose(outcome['elapsed_s'],outcome['time_s']-row['time_s'],atol=1e-12,rtol=0)
                    or not np.isclose(row['control_dt_s'],control_dt,atol=1e-12,rtol=0)):
                raise ValueError('an aligned measured control-interval outcome is required')
            _state(outcome['state'])
            if previous is not None:
                end=previous['outcome']
                if (end.get('terminal') or end['step']!=row['step'] or end['token']!=row['token']
                        or not np.isclose(end['time_s'],row['time_s'],atol=1e-12,rtol=0)):
                    raise ValueError('broken expert interval alignment')
                for key,value in end['state'].items():
                    if not np.array_equal(value,obs[key]):
                        raise ValueError(f'expert next-state mismatch: {key}')
                if ('close','hold','lift').index(phase)<('close','hold','lift').index(previous['phase']):
                    raise ValueError('expert phase sequence must not reset')
            prediction=row.get('prediction')
            if phase=='close':
                if not isinstance(prediction,dict) or prediction.get('model')!='linearized_pad_kinematics':
                    raise ValueError('physical expert MPC prediction required')
                trajectory=np.asarray(prediction['trajectory'])
                if trajectory.ndim!=2 or trajectory.shape[1]!=12 or len(trajectory)<1 or not np.isfinite(trajectory).all():
                    raise ValueError('finite hand-only MPC prediction trajectory required')
            elif prediction is not None:
                raise ValueError('hold/lift must not fabricate new MPC predictions')
            records.append(row);counts[phase]+=1;previous=row
    if not records:raise ValueError('no complete physical control intervals found')
    has_phases=all(counts[phase]>0 for phase in ('close','hold','lift'))
    terminal=records[-1]['outcome']
    qualified=bool(source_success and has_phases and terminal.get('terminal') is True
        and terminal.get('grasp_lift_passed') is True
        and all(row['outcome'].get('passed') is True for row in records))
    if require_success and not qualified:
        raise ValueError('complete successful close/hold/lift intervals are required')
    return records,dict(schema='dh116-contact-decision-v2',source_run=str(path.resolve()),
        rod_backend='torch_cuda',grasp_lift_passed=qualified,source_terminal=bool(summary),
        phase_counts=dict(counts),action_dim=12,arm_state_included=False,control_dt_s=control_dt,
        whole_wire_clear=bool(result.get('whole_wire_clear') is True),
        training_scope='successful_complete_expert' if qualified else 'partial_action_diagnostic')
