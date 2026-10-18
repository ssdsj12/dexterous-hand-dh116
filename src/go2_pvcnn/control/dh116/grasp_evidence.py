"""DH116 measured-evidence observer; never commands or attaches the wire.

Contact validity must come from the backend load/separation/region checks.
Four contact-channel material anchors are expressed in their respective hand
frames. Thumb and index may share a material section within one hand; the two
hands must observe disjoint material regions. IDs never change during a trial.
The 5 mN noise floor is explicit simulation configuration, not hardware data.
"""
import numpy as np


class GraspEvidenceGate:
    def __init__(self,*,material_ids,approach_qualified,minimum_hold_loads_n=None):
        if (approach_qualified is not True or len(material_ids)!=4
                or any(type(i) is not int or not 0<=i<=40 for i in material_ids)
                or set(material_ids[:2]) & set(material_ids[2:])):
            raise ValueError('measured approach and four fixed channel IDs in separate hand regions required')
        minimum=np.full(4,.005) if minimum_hold_loads_n is None else np.asarray(minimum_hold_loads_n,dtype=float)
        if (minimum.shape!=(4,) or not np.isfinite(minimum).all()
                or np.any(minimum<.005) or np.any(minimum>3.)):
            raise ValueError('hold preload must be four finite loads in [.005,3] N')
        self.minimum_hold_loads=minimum.copy()
        self.material_ids=tuple(material_ids)
        self.stage='close';self.last_time=None;self.contact_since=None
        self.hold_since=None;self.lift_since=None;self.baseline=None;self.reason=None

    def update(self,time_s,*,loads_n,valid_contacts,normals,anchors_local,
               station_rise_m,min_surface_clearance_m,safety_ok,active_constraints):
        load=np.asarray(loads_n,dtype=float);normal=np.asarray(normals,dtype=float)
        anchor=np.asarray(anchors_local,dtype=float);rise=np.asarray(station_rise_m,dtype=float)
        valid=np.asarray(valid_contacts)
        if (load.shape!=(4,) or normal.shape!=(4,3) or anchor.shape!=(4,3) or rise.shape!=(2,)
                or valid.shape!=(4,) or valid.dtype.kind!='b' or type(safety_ok) is not bool
                or any(not np.isfinite(x).all() for x in (load,normal,anchor,rise))
                or not np.isfinite([time_s,min_surface_clearance_m]).all() or time_s<0
                or not np.allclose(np.linalg.norm(normal,axis=1),1.,atol=1e-6,rtol=0)
                or any(self.material_ids[a]==self.material_ids[b] and
                       not np.allclose(anchor[a],anchor[b],atol=1e-10,rtol=0)
                       for a,b in ((0,1),(2,3)))):
            raise ValueError('invalid measured grasp evidence')
        if self.last_time is not None and not 0<time_s-self.last_time<=.01+1e-12:
            raise ValueError('grasp evidence time must advance with no missing interval')
        self.last_time=float(time_s)
        contact=bool(valid.all() and np.all(load>=.005)
                     and np.dot(normal[0],normal[1])<=-.5 and np.dot(normal[2],normal[3])<=-.5)
        slip=0. if self.baseline is None else float(np.linalg.norm(anchor-self.baseline,axis=1).max())
        if not safety_ok or active_constraints:
            self.stage='abort';self.reason='safety failure or forbidden wire constraint'
        elif self.stage in ('hold','lift','finish') and (not contact or slip>.002):
            self.stage='abort';self.reason='contact lost or material slip exceeds 2 mm'
        elif self.stage=='close':
            if not contact or np.any(load<self.minimum_hold_loads):self.contact_since=None
            elif self.contact_since is None:self.contact_since=float(time_s)
            elif time_s-self.contact_since>=.1-1e-12:
                self.stage='hold';self.hold_since=float(time_s);self.baseline=anchor.copy()
        elif self.stage=='hold' and time_s-self.hold_since>=1.-1e-12:
            self.stage='lift'
        elif self.stage=='lift':
            if np.any(rise<.03):self.lift_since=None
            elif self.lift_since is None:self.lift_since=float(time_s)
            elif time_s-self.lift_since>=1.-1e-12:self.stage='finish'
        return dict(stage=self.stage,reason=self.reason,permit_lift=self.stage=='lift',
                    grasp_lift_passed=self.stage=='finish',material_ids=list(self.material_ids),
                    max_material_slip_m=slip,station_rise_m=rise.tolist(),
                    whole_wire_clear=bool(self.stage=='finish' and min_surface_clearance_m>=.002))
