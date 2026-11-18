"""Opt-in bounded failed-episode resets; never suppress original transitions.

Episode limits remain 30 N / .02 rad. The separate 45 N / .03 rad bounds
restrict eligibility for training-only recovery; these are not hardware ratings.
Reset does not advance physics. Every remaining shared substep is checked; a
last-substep reset must pass the first next physical step before any recovery.
"""
import numpy as np
from .tabletop_task import joint_safety_failure
from .tabletop_training import PHYSICAL_FAILURES
from .tabletop_protected_vec_env import UnrecoverablePhysicsError

VERSION = 'validated_gpu_episode_reset_v1'


def eligible(reason, force, mimic):
    return bool(np.isfinite([force,mimic]).all() and force >= 0 and mimic >= 0 and (
        (reason == 'excess_force' and 30. < force < 45. and mimic <= .02) or
        (reason == 'mimic_error' and .02 < mimic < .03 and force <= 30.)))


def validated_recovery(info, done, contract, observation_shape):
    """Fail-closed callback check. Evaluation deliberately does not use this."""
    if (contract.get('training_failure_handling') != VERSION or not bool(done)
            or info.get('recovered_physical') is not True or info.get('global_stop') is not False
            or info.get('global_stop_failure') is not None or info.get('TimeLimit.truncated',False)
            or info.get('is_success',False) or info.get('stage_success',False)
            or info.get('failure') != info.get('safety_failure')):
        return False
    v,p=info.get('reset_validation'),info.get('physical')
    if not isinstance(v,dict) or not isinstance(p,dict):return False
    if (v.get('version') != VERSION or any(v.get(k) is not True for k in
            ('verified','passed','other_guards_safe','reset_guards_safe','post_reset_guards_safe'))):
        return False
    try:
        terminal=np.asarray(info['terminal_observation'])
        force,mimic,bottom,slip=(float(p[k]) for k in
            ('max_force_n','max_mimic_error_rad','object_bottom_m','slip_m'))
        clearance=float(v['terminal_clearance_m']);velocity=float(v['terminal_joint_velocity_rad_s'])
        decimation=contract['physics_hz']/contract['control_hz']
        step,remaining,checked=(v[k] for k in
            ('terminal_physics_substep','remaining_physics_substeps','checked_physics_substeps'))
    except (KeyError,TypeError,ValueError,OverflowError,ZeroDivisionError):return False
    if any(type(x) is not int for x in (step,remaining,checked)):return False
    scope='immediate_reset_only' if remaining==0 else 'reset_and_remaining_substeps'
    return bool(terminal.shape==tuple(observation_shape)
        and tuple(observation_shape)==(contract.get('observation_size',observation_shape[0]),)
        and terminal.dtype.kind in 'fiu' and np.isfinite(terminal).all()
        and np.isfinite([bottom,slip,clearance,velocity]).all()
        and bottom>=.29 and slip>=0 and clearance>=-.001 and 0<=velocity<=3.
        and 1<=step<=decimation and remaining==decimation-step and checked==remaining
        and v.get('validation_scope')==scope
        and v.get('post_reset_physics_checked') is (checked>0)
        and eligible(info.get('failure'),force,mimic))


class GpuEpisodeRecovery:
    def __init__(self,vec):
        self.vec=vec;self.world=vec.world
        self.pending=set();self.records={};self.stopped=False
        if any(e.reset_jitter!=0 for e in vec.envs):
            raise ValueError('GPU episode recovery requires deterministic zero-jitter resets')

    def begin(self):
        if self.stopped:raise RuntimeError('shared physics was globally stopped')
        self.records={}

    def abort(self,index,failure,detail=None):
        self.stopped=True
        info=dict(failure=failure,safety_failure=failure,recovery_rejection=detail,
                  reset_validation=self.records.get(index))
        raise UnrecoverablePhysicsError(failure,info,index)

    def checked(self,index,m,terminal_reason=None):
        w=self.world
        # Inspect all numeric measurement fields, not a single prioritized label.
        for key,value in m.items():
            if key.startswith('_') or key=='joint_failure':continue
            a=np.asarray(value[index])
            if a.dtype.kind in 'fiu' and not np.isfinite(a).all():
                self.abort(index,'nonfinite_state',key)
        for key in ('q','qd','limits','errors'):
            if not np.isfinite(getattr(w,key)[index]).all():
                self.abort(index,'nonfinite_joint_state',key)
        velocity=float(np.max(abs(w.qd[index])))
        if velocity>3.:self.abort(index,'joint_velocity')
        for hand in range(2):
            failure=joint_safety_failure(w.q[index,hand],w.limits[index,hand],w.errors[index,hand])
            if failure and not (terminal_reason=='mimic_error' and failure=='mimic_error'):
                self.abort(index,failure,'independent_joint_guard')
        if m['joint_failure'][index] and m['joint_failure'][index]!=terminal_reason:
            self.abort(index,m['joint_failure'][index])
        clearance=float(m['clearance'][index]);bottom=float(m['bottom'][index])
        loads=m.get('object_link_loads',m['forces'])[index]
        force=float(np.max(loads));mimic=float(np.max(abs(m['mimic_errors'][index])))
        if np.any(loads<0):self.abort(index,'nonfinite_state','negative_contact_load')
        if clearance<-.001:self.abort(index,'table_penetration')
        if bottom<.29:self.abort(index,'object_dropped')
        if terminal_reason:
            if not eligible(terminal_reason,force,mimic):
                self.abort(index,terminal_reason,'outside_recovery_bounds_or_multiple_guards')
        else:
            if force>30.:self.abort(index,'excess_force','unsafe_post_reset')
            if mimic>.02:self.abort(index,'mimic_error','unsafe_post_reset')
        return clearance,velocity

    def after_substep(self,task,m,step):
        # Check previously reset slots even though their original episode ended.
        for index in list(self.pending):
            self.checked(index,m);self.pending.remove(index)
        for index,record in self.records.items():
            self.checked(index,m)
            record['checked_physics_substeps']+=1
            record['post_reset_physics_checked']=True
        indices=[]
        for index in np.flatnonzero((task.failure != None) | task.success):
            index=int(index)
            if index in self.records:continue
            failure=task.failure[index]
            if failure not in PHYSICAL_FAILURES:
                # Ordinary timeout/success also freezes task measurements.
                # Inspect the live world until tick-end reset, not that snapshot.
                self.checked(index,m)
                continue
            if failure not in ('excess_force','mimic_error'):
                self.abort(index,failure)
            clearance,velocity=self.checked(index,m,terminal_reason=failure)
            self.vec.envs[index].measurement['diagnostics']=self.world.backends[index].measure().get('diagnostics',[])
            remaining=self.vec.decimation-step
            self.records[index]=dict(version=VERSION,verified=False,passed=False,
                other_guards_safe=True,reset_guards_safe=False,post_reset_guards_safe=False,
                terminal_clearance_m=clearance,terminal_joint_velocity_rad_s=velocity,
                terminal_physics_substep=step,remaining_physics_substeps=remaining,
                checked_physics_substeps=0,post_reset_physics_checked=False,
                validation_scope='immediate_reset_only' if remaining==0 else 'reset_and_remaining_substeps')
            indices.append(index)
        if indices:
            try:self.world.reset_slots(indices,[self.vec.envs[i].initial_offset for i in indices])
            except Exception as error:self.abort(indices[0],'reset_failed',str(error))
            reset_m=self.world.measure_batch()
            for index in indices:
                self.checked(index,reset_m)
                self.records[index].update(reset_guards_safe=True,post_reset_guards_safe=True)

    def finish(self,results):
        for index,v in self.records.items():
            v.update(verified=True,passed=True)
            if v['checked_physics_substeps']==0:self.pending.add(index)
            info=results[index][4]
            info.update(recovered_physical=True,global_stop=False,reset_validation=dict(v))
        self.vec._already_reset=set(self.records)
