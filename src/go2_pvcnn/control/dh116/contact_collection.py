"""Time-aligned physical hand decisions, separate from MPC predictions.

The next decision closes the previous action's measured control interval.
Keeping a pending record avoids labeling a single 100 us physics step as a
whole 10 ms action outcome. No successful phase is inferred by this module.
"""
from copy import deepcopy
import math

import numpy as np


def _array(value,shape,name,*,boolean=False):
    a=np.asarray(value)
    if (a.shape!=shape or (boolean and a.dtype!=np.bool_)
            or not np.isfinite(a).all()):
        raise ValueError(f'{name} requires finite {shape}' + (' booleans' if boolean else ''))
    return a.tolist()


def _state(value):
    try:
        position=np.asarray(value['rod_position'])
        if position.ndim!=2 or position.shape[0]!=3 or position.shape[1]<2:
            raise ValueError('rod_position requires (3,N) nodes')
        nodes=position.shape[1]
        shapes=dict(hand_q=(2,6),hand_qd=(2,6),contact_loads_n=(6,),
            contact_gaps_m=(2,3),contact_valid=(6,),contact_valid_known=(6,),
            rod_position=(3,nodes),rod_velocity=(3,nodes),rod_director=(3,3,nodes-1),
            rod_omega=(3,nodes-1),rod_radius_m=(nodes-1,))
        return {k:_array(value[k],shape,k,boolean=k in ('contact_valid','contact_valid_known'))
                for k,shape in shapes.items()}
    except KeyError as e:
        raise ValueError(f'missing measured feedback: {e.args[0]}') from e


def _observation(value):
    result=_state(value)
    try:
        shapes=dict(pad_positions_w=(2,3,3),target_points_w=(2,3,3),
                    pad_jacobians=(2,3,3,6),target_loads_n=(6,),
                    palm_poses_w=(2,7),contact_normals_w=(2,3,3))
        result.update({k:_array(value[k],shape,k) for k,shape in shapes.items()})
        names=list(value['clearance_body_names'])
        result['clearance_body_names']=names
        result['clearances_m']=_array(value['clearances_m'],(len(names),),'clearances_m')
        result['clearance_jacobians']=_array(value['clearance_jacobians'],(len(names),12),
                                               'clearance_jacobians')
        result['goal']=deepcopy(value['goal'])
        if (not isinstance(result['goal'],dict)
                or not math.isfinite(result['goal']['lift_height_m'])
                or result['goal']['lift_height_m']<=0
                or result['goal']['contact_mode'] not in ('pinch','pinch_hook')):
            raise ValueError('explicit lift/contact goal required')
    except KeyError as e:
        raise ValueError(f'missing decision observation: {e.args[0]}') from e
    return result


def _json_value(value):
    if isinstance(value,dict):return {k:_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_value(v) for v in value]
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    if isinstance(value,(np.ndarray,np.generic)):return value.tolist()
    return deepcopy(value)


class ContactDecisionCollector:
    def __init__(self,*,control_dt_s):
        if not math.isfinite(control_dt_s) or control_dt_s<=0:
            raise ValueError('positive control interval required')
        self.control_dt_s=float(control_dt_s)
        self.pending=None

    def begin(self,*,step,time_s,token,phase,owner,observation,previous_action,action,prediction):
        if phase not in ('close','hold','lift'):
            raise ValueError('close/hold/lift decision required')
        if owner not in ('hand_mpc','hand_mpc_hold','hand_teacher','hand_student'):
            raise ValueError('explicit hand-only controller owner required')
        if not isinstance(step,int) or step<0 or not math.isfinite(time_s) or time_s<0:
            raise ValueError('measured decision time required')
        obs=_observation(observation)
        new=dict(schema='dh116-contact-decision-v2',step=step,time_s=float(time_s),token=token,
            phase=phase,owner=owner,control_dt_s=self.control_dt_s,action_dim=12,
            arm_state_included=False,observation=obs,
            previous_action=_array(previous_action,(12,),'previous_action'),
            action=_array(action,(12,),'action'),prediction=_json_value(prediction))
        completed=self.finish(state=observation,step=step,time_s=time_s,token=token,
            passed=True,terminal=False,grasp_lift_passed=False)
        self.pending=new
        return completed

    def finish(self,*,state,step,time_s,token,passed,terminal,grasp_lift_passed):
        if self.pending is None:return None
        if (step<=self.pending['step'] or not math.isfinite(time_s)
                or time_s<=self.pending['time_s']):
            raise ValueError('measured outcome must follow its decision')
        if grasp_lift_passed and not (passed and terminal):
            raise ValueError('successful outcome requires a passed terminal episode')
        outcome=dict(step=step,time_s=float(time_s),token=token,
            elapsed_s=float(time_s-self.pending['time_s']),state=_state(state),
            passed=bool(passed),terminal=bool(terminal),grasp_lift_passed=bool(grasp_lift_passed))
        result=dict(self.pending,outcome=outcome)
        self.pending=None
        return result
