"""Pure NumPy mounted17D FK/J hypotheses, never a controller or simulator.

Inputs are owned descriptor values plus an explicit coordinate hypothesis.
The actual 17 SDK rows remain independent (including measured followers); no
affine mimic, reflection, RNEA, force prediction or future-state qualification
is provided.  The old 11D HandPhysicalKinematics and all controls stay separate.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import math
from numbers import Real
from types import MappingProxyType

import numpy as np

from .tabletop_bimanual_mpc_model_descriptor import (
    MountedModelDescriptorSample,assemble_mounted_graph,_digest,_freeze,_thaw)


VERSION='dh116-mounted-geometry-hypothesis-v1'
_QUALIFICATION=dict(native_geometry_calibrated=False,dynamics_qualified=False,
    candidate_future_dynamics_qualified=False,planning_eligible=False,
    native_attested=False,native_success_qualification_issued=False,
    hardware_measurement_qualified=False)


def descriptor_digest(sample):
    """Content identity, not Native provenance or physical model validation."""
    if not isinstance(sample,MountedModelDescriptorSample):
        raise ValueError('owned MountedModelDescriptorSample required')
    return _digest(sample.descriptor)


def _vector(value,size,name):
    a=np.asarray(value)
    if (a.shape!=(size,) or not np.issubdtype(a.dtype,np.number)
            or np.issubdtype(a.dtype,np.complexfloating) or not np.isfinite(a).all()):
        raise ValueError(f'{name} must contain {size} finite real values')
    return np.array(a,dtype=np.float64,copy=True)


def _skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def _quaternion_matrix(wxyz):
    q=_vector(wxyz,4,'wxyz quaternion'); norm=float(np.linalg.norm(q))
    if not math.isclose(norm,1.,rel_tol=0.,abs_tol=1e-5):
        raise ValueError('unit quaternion required; no zero/sentinel replacement')
    # Roundoff normalization is explicit in the mandatory hypothesis policy;
    # original quaternion bytes remain available in its receipt.
    w,x,y,z=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _frame(position,quaternion):
    t=np.eye(4); t[:3,3]=_vector(position,3,'joint frame position')
    t[:3,:3]=_quaternion_matrix(quaternion)
    return t


def _attr(j,name):
    r=j['attributes'].get('physics:'+name)
    if not isinstance(r,dict) or r.get('available') is not True:
        raise ValueError('missing physical joint frame/axis: '+name)
    return r['value']


@dataclass(frozen=True)
class MountedFKResult:
    side: str
    clock: object
    input_q_sdk: tuple
    dof_names: tuple
    link_paths: tuple
    link_poses_world: tuple
    assumption_receipt: object
    qualification: object
    kind: str='geometry_hypothesis'


@dataclass(frozen=True)
class MountedPointJacobian:
    side: str
    clock: object
    input_q_sdk: tuple
    link_row: int
    point_world_m: tuple
    jacobian_world: tuple
    coordinate_units: tuple
    assumption_receipt: object
    qualification: object
    kind: str='geometry_hypothesis'


class MountedTreeKinematics:
    """Full17D fixed-base tree under explicit, uncalibrated q/frame hypotheses.

    Reversed internal edges, floating roots and unsupported topologies are
    rejected, not reconstructed from failed descriptor raw evidence.  The
    fixed world attachment may itself list world as body0 OR body1; actual
    root-link readback selects the root pose without guessing a world joint
    rest transform from USD defaults.
    """

    @classmethod
    def from_descriptor(cls,sample,*,side,coordinate_hypothesis):
        if not isinstance(sample,MountedModelDescriptorSample) or sample.structure_valid is not True:
            raise ValueError('structure-valid owned mounted descriptor required; no raw bypass')
        if side not in ('left','right'): raise ValueError('explicit left/right side required')
        if _thaw(sample.descriptor.get('clock'))!=asdict(sample.clock):
            raise ValueError('descriptor physical clock does not match its receipt')
        h=_thaw(coordinate_hypothesis)
        if (not isinstance(h,dict) or h.get('version')!=VERSION or h.get('side')!=side
                or h.get('descriptor_sha256')!=descriptor_digest(sample)):
            raise ValueError('coordinate hypothesis version/side/descriptor SHA is unbound')
        if (h.get('root_pose_source')!='existing_sdk_link_transform_readback'
                or h.get('root_frame')!='world_link' or h.get('root_quaternion_order')!='xyzw'
                or h.get('quaternion_policy')!='unit_tolerance_1e-5_normalize_roundoff'
                or h.get('edge_direction')!='authored_body0_to_body1'
                or not isinstance(h.get('basis'),str) or not h['basis'].strip()):
            raise ValueError('explicit root frame/order/edge/sign basis required; unsupported hypothesis')
        hand=_thaw(sample.descriptor['hands'][side]); b=hand['binding']
        g=assemble_mounted_graph(hand['graph'],b)
        root=g['root_link_row']
        if type(h.get('root_link_row')) is not int or h['root_link_row']!=root:
            raise ValueError('hypothesis root row is not bound to actual graph/SDK link rows')
        coordinates=h.get('joint_coordinates')
        if not isinstance(coordinates,list) or len(coordinates)!=17:
            raise ValueError('explicit per-SDK-row 17D coordinate hypothesis required')
        signs=[]; offsets=[]; units=[]
        for row,c in enumerate(coordinates):
            unit='m' if b['dof_types'][row]==1 else 'rad'
            if (not isinstance(c,dict) or c.get('dof_name')!=b['dof_names'][row]
                    or c.get('dof_path')!=b['dof_paths'][row] or c.get('unit')!=unit
                    or isinstance(c.get('sign'),bool) or c.get('sign') not in (-1,1)
                    or isinstance(c.get('zero_offset'),bool) or not isinstance(c.get('zero_offset'),Real)
                    or not math.isfinite(c['zero_offset'])):
                raise ValueError('SDK row/name/path/unit/sign/zero reference unbound or unsupported')
            signs.append(c['sign']); offsets.append(c['zero_offset']); units.append(unit)
        readback=hand['readbacks'].get('get_link_transforms',{})
        if readback.get('selected_valid') is not True:
            raise ValueError('actual root-link transform readback unavailable')
        pose=_vector(readback['selected'][root],7,'actual root link pose')
        troot=_frame(pose[:3],pose[[6,3,4,5]])
        self=cls.__new__(cls)
        self.side,self.clock=side,sample.clock
        self.dof_names,self.link_paths=tuple(b['dof_names']),tuple(b['link_paths'])
        self.coordinate_units=tuple(units); self._root=root; self._root_transform=troot
        self._sign=np.array(signs,dtype=float); self._offset=np.array(offsets,dtype=float)
        self._edges=[]; pending=list(enumerate(g['joints'])); known={root}
        meta=b['metatype']; linkrows={path:i for i,path in enumerate(b['link_paths'])}
        while pending:
            ready=[(i,j) for i,j in pending if linkrows[j['body0'][0]] in known]
            if not ready: raise ValueError('unsupported reversed/disconnected/cyclic internal edge')
            for i,j in ready:
                parent,child=linkrows[j['body0'][0]],linkrows[j['body1'][0]]
                if child in known: raise ValueError('duplicate/reversed tree edge')
                t0=_frame(_attr(j,'localPos0'),_attr(j,'localRot0'))
                t1=_frame(_attr(j,'localPos1'),_attr(j,'localRot1'))
                count=meta['joint_dof_counts'][i]
                qrow=meta['joint_dof_offsets'][i] if count else None
                axis=np.eye(3)['XYZ'.index(_attr(j,'axis'))] if count else None
                kind='fixed' if not count else 'prismatic' if b['dof_types'][qrow]==1 else 'revolute'
                self._edges.append((parent,child,t0,np.linalg.inv(t1),axis,qrow,kind))
                known.add(child); pending.remove((i,j))
        if len(known)!=23: raise ValueError('mounted link coverage incomplete')
        receipt=dict(h,kind='geometry_hypothesis',native_calibrated=False,
            root_raw_pose_xyzw=pose.tolist(),root_transform_source_api=readback.get('source'),
            world_attachment=_thaw(g['world_attachments'][0]),
            source_identity=hand.get('source_identity'),
            actual_q17_independent=True,affine_mimic_applied=False,
            actual_pose_tracking_and_collision_qualified=False,
            point_jacobian_frame='world point linear XYZ; mixed SDK-coordinate columns')
        self.assumption_receipt=_freeze(receipt)
        self.qualification=MappingProxyType(dict(_QUALIFICATION))
        return self

    def _evaluate(self,q_sdk):
        q=_vector(q_sdk,17,'q_sdk17'); angles=self._sign*q+self._offset
        if not np.isfinite(angles).all(): raise ValueError('coordinate hypothesis arithmetic overflow; finite geometry required')
        poses={self._root:self._root_transform.copy()}; axes={}; ancestors={self._root:()}
        for parent,child,t0,t1,axis,row,kind in self._edges:
            before=poses[parent]@t0; motion=np.eye(4)
            if kind!='fixed':
                axes[row]=(before[:3,3].copy(),before[:3,:3]@axis,kind)
                theta=angles[row]
                if kind=='revolute':
                    k=_skew(axis); motion[:3,:3]=np.eye(3)+math.sin(theta)*k+(1-math.cos(theta))*(k@k)
                else: motion[:3,3]=axis*theta
            poses[child]=before@motion@t1
            if not np.isfinite(poses[child]).all(): raise ValueError('FK overflow; finite geometry required')
            ancestors[child]=ancestors[parent]+(() if row is None else (row,))
        return q,poses,axes,ancestors

    def fk(self,q_sdk17):
        q,poses,_,_=self._evaluate(q_sdk17)
        return MountedFKResult(self.side,self.clock,tuple(q),self.dof_names,self.link_paths,
            _freeze([poses[i].tolist() for i in range(23)]),self.assumption_receipt,self.qualification)

    def point_jacobian(self,q_sdk17,*,link_row,point_link_xyz):
        if type(link_row) is not int or not 0<=link_row<23: raise ValueError('actual SDK link row required')
        point=_vector(point_link_xyz,3,'point_link_xyz'); q,poses,axes,ancestors=self._evaluate(q_sdk17)
        t=poses[link_row]; world=t[:3,:3]@point+t[:3,3]; jac=np.zeros((3,17))
        for row in ancestors[link_row]:
            origin,axis,kind=axes[row]
            jac[:,row]=(axis if kind=='prismatic' else np.cross(axis,world-origin))*self._sign[row]
        if not np.isfinite(world).all() or not np.isfinite(jac).all():
            raise ValueError('point/Jacobian overflow; finite geometry required')
        return MountedPointJacobian(self.side,self.clock,tuple(q),link_row,tuple(world),
            _freeze(jac.tolist()),self.coordinate_units,self.assumption_receipt,self.qualification)

    def audit_point_jacobian(self,q_sdk17,*,link_row,point_link_xyz,epsilon_by_dof):
        q=_vector(q_sdk17,17,'q_sdk17'); eps=_vector(epsilon_by_dof,17,'epsilon_by_dof')
        if np.any(eps<=0): raise ValueError('strictly positive per-coordinate FD epsilon required')
        exact=self.point_jacobian(q,link_row=link_row,point_link_xyz=point_link_xyz)
        point=np.asarray(exact.point_world_m); analytical=np.asarray(exact.jacobian_world)
        forward=[]; backward=[]
        for row,e in enumerate(eps):
            delta=np.zeros(17); delta[row]=e
            plus=self.point_jacobian(q+delta,link_row=link_row,point_link_xyz=point_link_xyz).point_world_m
            minus=self.point_jacobian(q-delta,link_row=link_row,point_link_xyz=point_link_xyz).point_world_m
            forward.append((np.asarray(plus)-point)/e); backward.append((point-np.asarray(minus))/e)
        forward,backward=np.array(forward).T,np.array(backward).T
        central=(forward+backward)/2
        if not all(np.isfinite(a).all() for a in (forward,backward,central)):
            raise ValueError('finite difference arithmetic overflow')
        return _freeze(dict(kind='geometry_hypothesis',side=self.side,clock=asdict(self.clock),
            analytical=analytical.tolist(),forward=forward.tolist(),backward=backward.tolist(),central=central.tolist(),
            forward_error=(forward-analytical).tolist(),backward_error=(backward-analytical).tolist(),
            central_error=(central-analytical).tolist(),epsilon_by_dof=eps.tolist(),
            coordinate_units=self.coordinate_units,assumption_receipt=self.assumption_receipt,
            qualification=self.qualification))
