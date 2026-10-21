"""Measured sequence coordinator, without robot commands or wire attachments.

Caller must provide actual, contiguous post-physics measurements and retain
all contact, source-clearance, effort and speed guards. A desired trajectory
or independent initial-touch diagnostic cannot qualify approach here.
"""
import numpy as np
from .grasp_evidence import GraspEvidenceGate
from .touch_first import ApproachReadiness, TouchFirst
from .symmetric_transport import SynchronizedTransport


def validate_continuous_options(args):
    """Fail closed before launching Isaac; legacy diagnostic options stay separate."""
    if not args.grasp_sequence:return
    required=('active_pose','active_pose_sha256','approach_certificate',
              'approach_certificate_sha256','point_references','contact_manifest',
              'contact_manifest_sha256','coordinated_closure','gravity_hold','table_wire')
    loads=np.asarray([] if args.closure_loads is None else args.closure_loads,dtype=float)
    if (args.closure_pilot or args.probe_channel or args.disable_contact_reaction
            or not all(getattr(args,k) for k in required)
            or not np.isfinite(args.physics_dt) or not 0<args.physics_dt<=.0001
            or args.steps*args.physics_dt<=5.1 or loads.shape!=(6,)
            or not np.isfinite(loads).all() or np.any(loads<.05) or np.any(loads>3.)):
        raise ValueError('continuous grasp requires pinned approach, six loads in [.05,3] N, coordinated reaction control, dt<=100us and >5.1s')


class TabletopGraspSequence:
    def __init__(self,*,material_ids,target_loads_n,state_triggered=False,symmetric_lift=False,speed_profile=None):
        from .speed_profile import GraspSpeedProfile
        if speed_profile is not None and (not isinstance(speed_profile,GraspSpeedProfile)
                or not state_triggered or speed_profile.lift_m_s>.01):
            raise ValueError('staged speed requires measured control and lift <=10 mm/s')
        self.speed_profile=speed_profile
        targets=np.asarray(target_loads_n,dtype=float)
        if (targets.shape!=(6,) or not np.isfinite(targets).all()
                or np.any(targets<.005/.9) or np.any(targets>3.)):
            raise ValueError('six finite simulation targets in [.005/.9,3] N required')
        # Validate IDs/thresholds without granting approach or retaining a gate.
        if (len(material_ids)!=6 or any(type(i) is not int or not 0<=i<=40 for i in material_ids)
                or set(material_ids[:3])&set(material_ids[3:])):
            raise ValueError('six fixed channel IDs in separate hand regions required')
        self.material_ids=tuple(material_ids);self.minimum_loads=.9*targets
        self.pinch_indices=(0,1,3,4)
        self.support_indices=(2,5)
        self.stage='settle';self.reason=None;self.last_time=None;self.gate=None
        self.approach_deadline=4.
        if type(state_triggered) is not bool:raise ValueError('state_triggered must be bool')
        self.state_triggered=state_triggered
        self.readiness=ApproachReadiness() if state_triggered else None
        self.touch=TouchFirst(targets.reshape(2,3),speed_profile=speed_profile) if state_triggered else None
        self.touch_output=None
        self.settled_heights=None;self.last_rise=np.zeros(2);self.lift_started=None;self.last_grasp={}
        if type(symmetric_lift) is not bool:raise ValueError('symmetric_lift must be bool')
        self.symmetric_lift=symmetric_lift
        self.lift_speed=.005 if speed_profile is None else speed_profile.lift_m_s
        self.transport_coordinator=SynchronizedTransport(max_speed_m_s=self.lift_speed) if symmetric_lift else None
        self.synchronization_output=None

    def _result(self):
        result=dict(self.last_grasp)
        result.update(stage=self.stage,reason=self.reason,
            permit_close=self.stage in ('close','hold','lift'),permit_lift=self.stage=='lift',
            grasp_lift_passed=self.stage=='finish',
            whole_wire_clear=bool(self.stage=='finish' and self.last_grasp.get('whole_wire_clear',False)),
            station_rise_m=self.last_rise.tolist(),material_ids=list(self.material_ids),
            minimum_hold_loads_n=self.minimum_loads.tolist())
        if self.state_triggered:
            result['approach_ready_since']=self.readiness.ready_since
            result['touch_first']=None if self.touch_output is None else {
                k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in self.touch_output.items()}
        if self.symmetric_lift:
            result['lift_synchronization']=None if self.synchronization_output is None else {
                k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in self.synchronization_output.items()}
        return result

    def update(self,time_s,*,settled,approach_position_errors_m,approach_normal_errors_rad,
               station_heights_m,loads_n,valid_contacts,normals,anchors_local,
               min_surface_clearance_m,safety_ok,active_constraints,joint_speeds_rad_s=None,surface_gaps_m=None):
        if self.speed_profile is not None:
            gaps=np.asarray(surface_gaps_m,dtype=float)
            if gaps.shape!=(6,) or not np.isfinite(gaps).all():
                raise ValueError('finite measured six-channel surface gaps required')
        position=np.asarray(approach_position_errors_m,dtype=float)
        angles=np.asarray(approach_normal_errors_rad,dtype=float)
        heights=np.asarray(station_heights_m,dtype=float);load=np.asarray(loads_n,dtype=float)
        normal=np.asarray(normals,dtype=float);anchors=np.asarray(anchors_local,dtype=float)
        valid=np.asarray(valid_contacts)
        if (position.shape!=(6,) or angles.shape!=(6,) or heights.shape!=(2,) or load.shape!=(6,)
                or normal.shape!=(6,3) or anchors.shape!=(6,3) or valid.shape!=(6,)
                or valid.dtype.kind!='b' or type(settled) is not bool or type(safety_ok) is not bool
                or any(not np.isfinite(x).all() for x in (position,angles,heights,load,normal,anchors))
                or np.any(position<0) or np.any(angles<0) or np.any(load<0)
                or not np.allclose(np.linalg.norm(normal,axis=1),1.,atol=1e-6,rtol=0)
                or not np.isfinite([time_s,min_surface_clearance_m]).all() or time_s<0):
            raise ValueError('invalid measured sequence evidence')
        if self.state_triggered:
            speeds=np.asarray(joint_speeds_rad_s,dtype=float)
            if speeds.shape!=(2,18) or not np.isfinite(speeds).all():
                raise ValueError('actual joint speeds required for state-triggered grasp')
        if ((self.last_time is None and time_s>.01+1e-12)
                or (self.last_time is not None and not 0<time_s-self.last_time<=.01+1e-12)):
            raise ValueError('sequence evidence must be contiguous and advance')
        self.last_time=float(time_s)
        if self.stage=='abort':return self._result()
        if not safety_ok or active_constraints:
            self.stage='abort';self.reason='safety failure or forbidden wire constraint'
            return self._result()
        if self.stage=='settle' and time_s>=.5-1e-12:
            if not settled:
                self.stage='abort';self.reason='wire did not settle before approach'
                return self._result()
            self.settled_heights=heights.copy();self.stage='approach'
        if self.settled_heights is not None:self.last_rise=heights-self.settled_heights
        ready=False
        if self.stage=='approach':
            if self.state_triggered:
                try:
                    ready=self.readiness.update(time_s,position_errors_m=position,
                        normal_errors_rad=angles,joint_speeds_rad_s=speeds)
                except TimeoutError as error:
                    self.stage='abort';self.reason=str(error)
                    return self._result()
            else:ready=time_s>=self.approach_deadline-1e-12
        if ready:
            if np.any(position>.002) or np.any(angles>np.deg2rad(5.)):
                self.stage='abort';self.reason='measured approach position or normal failed'
                return self._result()
            self.gate=GraspEvidenceGate(material_ids=self.material_ids,approach_qualified=True,
                minimum_hold_loads_n=self.minimum_loads)
            self.stage='close'
        if self.gate is not None:
            previous=self.stage
            gate_valid=valid
            if self.state_triggered and self.stage=='close':
                self.touch_output=self.touch.update(time_s,load.reshape(2,3),valid.reshape(2,3),
                    surface_gaps_m=None if self.speed_profile is None else gaps.reshape(2,3))
                # Preserve measured contacts in logs; withhold qualification until
                # touch/opposition and target ramp have actually completed.
                if not self.touch_output['ready_for_hold']:gate_valid=np.zeros(6,dtype=bool)
            self.last_grasp=self.gate.update(time_s,loads_n=load,valid_contacts=gate_valid,normals=normal,
                anchors_local=anchors,station_rise_m=self.last_rise,
                min_surface_clearance_m=min_surface_clearance_m,safety_ok=safety_ok,active_constraints=active_constraints,
                lift_alignment_ok=bool(not self.symmetric_lift or abs(self.last_rise[0]-self.last_rise[1])<=.001))
            self.stage=self.last_grasp['stage'];self.reason=self.last_grasp['reason']
            if previous!='lift' and self.stage=='lift':self.lift_started=float(time_s)
            if self.symmetric_lift and self.stage=='lift':
                try:
                    self.synchronization_output=self.transport_coordinator.update(time_s,
                        rises_m=self.last_rise,contacts_valid=valid.reshape(2,3).all(axis=1),direction=1)
                except ValueError as error:
                    self.stage='abort';self.reason=str(error)
                    self.gate.stage='abort';self.gate.reason=self.reason
        return self._result()

    def transport_velocity(self,time_s):
        if (not np.isfinite(time_s) or time_s<0 or (self.last_time is not None
                and not 0<=time_s-self.last_time<=.01+1e-12)):
            raise ValueError('transport time must be finite and not precede evidence')
        if self.symmetric_lift:
            if self.stage!='lift' or np.all(self.last_rise>=.035) or self.synchronization_output is None:
                return np.zeros((2,3))
            return self.synchronization_output['velocity_w'].copy()
        velocity=np.zeros(3)
        if self.stage=='lift' and np.any(self.last_rise<.035):
            velocity[2]=self.lift_speed*float(np.clip((time_s-self.lift_started)/.2,0.,1.))
        return velocity

    def stage_summary(self,*,final_passed):
        """Final qualification, not an in-progress status or a command history.

        Intermediate passes require measured transition evidence. The last
        lift PASS additionally requires the runner's final physical/J gates.
        """
        if type(final_passed) is not bool or (final_passed and self.stage!='finish'):
            raise ValueError('final success requires a completed measured sequence')
        phases=('settle','approach','close','hold','lift')
        if self.last_time is None:return dict.fromkeys(phases,'NOT_RUN')
        completed=(self.settled_heights is not None,self.gate is not None,
                   self.gate is not None and self.gate.baseline is not None,
                   self.lift_started is not None,final_passed and self.stage=='finish')
        result={};attempted=True
        for phase,passed in zip(phases,completed):
            result[phase]='PASS' if passed else ('FAIL' if attempted else 'NOT_RUN')
            attempted=bool(passed)
        return result
