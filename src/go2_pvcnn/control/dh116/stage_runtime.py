"""Stage-local restore orchestration; never certifies a continuous whole run."""
import numpy as np
from .stage_state import validate_state
from .restore_gate import RestoreGate
from .grasp_sequence import TabletopGraspSequence
from .grasp_evidence import GraspEvidenceGate


def robot_restore_arrays(saved, joint_names, live_limits, live_root):
    """Map by name and verify the current fixed-base asset before any write."""
    if len(joint_names)!=18 or set(joint_names)!=set(saved['joint_names']):
        raise ValueError('live joint identity mismatch')
    ids=[saved['joint_names'].index(n) for n in joint_names]
    result={k:np.asarray(saved[k],dtype=float)[ids].copy() for k in ('q','qd','target','lower','upper')}
    limits=np.asarray(live_limits,dtype=float)
    if (limits.shape!=(18,2) or not np.isfinite(limits).all()
            or not np.allclose(limits,np.stack([result['lower'],result['upper']],axis=1),atol=1e-6,rtol=0)
            or not np.allclose(live_root,saved['root_state'],atol=1e-6,rtol=0)
            or np.any(result['q']<limits[:,0]) or np.any(result['q']>limits[:,1])):
        raise ValueError('live limits or fixed root mismatch')
    caps=np.array([.151 if '_panda_joint' in n else .31 for n in joint_names])
    if np.any(np.abs(result['qd'])>caps):raise ValueError('saved velocity violates runtime guard')
    return result


class StageResetSequence(TabletopGraspSequence):
    def __init__(self,payload,expected_pins,*,state_triggered=False):
        self.saved=validate_state(payload,expected_pins)
        super().__init__(material_ids=self.saved['material_ids'],target_loads_n=self.saved['target_loads_n'],
                         state_triggered=state_triggered)
        self.entry_stage=self.saved['entry_stage']
        self.imported_evidence=self.saved['source']['passed_stages'][:]
        self.restore_gate=RestoreGate(self.entry_stage,minimum_loads_n=self.minimum_loads,
                                     saved_anchors_local=self.saved['anchors_local'])
        self.stage='restore_validate';self.released_at=None

    @property
    def control_contact_active(self):
        return self.stage in ('close','hold','lift') or (
            self.stage=='restore_validate' and self.entry_stage in ('hold','lift'))

    def _result(self):
        result=super()._result()
        result.update(imported_evidence=self.imported_evidence[:],continuous_grasp_qualified=False)
        return result

    def update(self,time_s,**m):
        if self.stage!='restore_validate':return super().update(time_s,**m)
        speed_ok=True
        if self.state_triggered:
            speeds=np.asarray(m.get('joint_speeds_rad_s'),dtype=float)
            if speeds.shape!=(2,18) or not np.isfinite(speeds).all():
                raise ValueError('actual joint speeds required during restore')
            speed_ok=bool(np.all(np.abs(speeds)<=.02))
        for key,shape in (('station_heights_m',(2,)),('approach_position_errors_m',(4,)),
                          ('approach_normal_errors_rad',(4,))):
            value=np.asarray(m[key],dtype=float)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('invalid restore measurement')
            if key!='station_heights_m' and np.any(value<0):raise ValueError('negative approach error')
        if not np.isfinite(m['min_surface_clearance_m']):raise ValueError('invalid restore clearance')
        r=self.restore_gate.update(time_s,safety_ok=m['safety_ok'],active_constraints=m['active_constraints'],
            settled=m['settled'],approach_ok=bool(np.all(np.asarray(m['approach_position_errors_m'])<=.002)
                and np.all(np.asarray(m['approach_normal_errors_rad'])<=np.deg2rad(5.))
                and speed_ok),
            loads_n=m['loads_n'],valid_contacts=m['valid_contacts'],normals=m['normals'],anchors_local=m['anchors_local'])
        self.last_time=float(time_s)
        if r['aborted']:
            self.stage='abort';self.reason=r['reason']
        elif r['ready']:
            self.stage=self.entry_stage;self.released_at=float(time_s)
            self.settled_heights=np.array(self.saved['settled_heights_m'])
            self.last_rise=np.asarray(m['station_heights_m'])-self.settled_heights
            if not self.state_triggered:self.approach_deadline=time_s+3.5
            if self.entry_stage!='approach':
                self.gate=GraspEvidenceGate(material_ids=self.material_ids,approach_qualified=True,
                                            minimum_hold_loads_n=self.minimum_loads)
                self.gate.stage=self.entry_stage
                if self.entry_stage in ('hold','lift'):
                    self.gate.baseline=np.array(self.saved['anchors_local'])
                    self.gate.hold_since=float(time_s)
                if self.entry_stage=='lift':self.lift_started=float(time_s)
        return dict(self._result(),restore_validation=r)

    def motion_time(self,time_s):
        # Only trajectory lookup uses this local phase coordinate. Physical
        # evidence and backend time remain fresh and contiguous from zero.
        if self.released_at is None:return 0.
        return .5+time_s-self.released_at if self.entry_stage=='approach' else 4.

    def stage_summary(self,*,final_passed):
        result=super().stage_summary(final_passed=final_passed)
        for phase in self.imported_evidence:result[phase]='IMPORTED'
        return result
