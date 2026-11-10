"""Owned numeric batch FK for an already compiled mounted-model hypothesis.

The existing scalar FK and its coordinate contract remain unchanged. This
kernel batches numerical evaluation only: no SDK, joint response, Jacobian,
collision, dynamics, source acquisition or execution is provided. Its root
pose stays at the source sample; forecast rows are never future measurements.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .tabletop_bimanual_mpc_mounted_kinematics import MountedTreeKinematics,_skew


VERSION='dh116-mounted-numeric-batch-fk-v1'


def _owned(value,shape=None):
    a=np.array(value,dtype=np.float64,copy=True)
    if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():
        raise ValueError('finite compiled mounted-model parameter shape required')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class MountedFKBatchResult:
    side:str
    source_clock:object
    input_q_sdk:object
    dof_names:tuple
    link_paths:tuple
    link_poses_world:object
    assumption_receipt:object
    qualification:object
    root_pose_assumed_constant:bool=True
    kind:str='geometry_forecast_hypothesis'
    version:str=VERSION


class MountedFKBatchKernel:
    """Copy a validated scalar tree once, then evaluate independent q17 rows.

    Results are owned read-only NumPy arrays, not a serialized evidence DTO.
    Changing an input or the original tree cannot change an existing kernel.
    Recompile from fresh actual root/source context before a new feedback
    origin; this class supplies no freshness or control-admission decision.
    """
    def __init__(self,tree):
        if type(tree) is not MountedTreeKinematics or any(v is not False for v in tree.qualification.values()):
            raise ValueError('exact uncalibrated MountedTreeKinematics required')
        self.side,self.source_clock=tree.side,tree.clock
        self.dof_names,self.link_paths=tuple(tree.dof_names),tuple(tree.link_paths)
        if len(self.dof_names)!=17 or len(self.link_paths)!=23:
            raise ValueError('complete full17/23link mounted model required')
        self._root=tree._root
        self._root_transform=_owned(tree._root_transform,(4,4))
        self._sign=_owned(tree._sign,(17,));self._offset=_owned(tree._offset,(17,))
        self.assumption_receipt=tree.assumption_receipt
        self.qualification=MappingProxyType(dict(tree.qualification))
        edges=[];known={self._root};rows=set()
        for parent,child,t0,t1,axis,row,kind in tree._edges:
            if (parent not in known or child in known or type(child) is not int or not 0<=child<23
                    or kind not in ('fixed','prismatic','revolute')):
                raise ValueError('complete directed mounted-tree topology required')
            if kind=='fixed':
                if row is not None or axis is not None:raise ValueError('fixed edge must not consume a coordinate')
                unit_axis=skew=skew2=None
            else:
                if type(row) is not int or not 0<=row<17 or row in rows:raise ValueError('independent unique full17 coordinate rows required')
                rows.add(row);unit_axis=_owned(axis,(3,))
                if not np.isclose(np.linalg.norm(unit_axis),1.,rtol=0,atol=1e-12):raise ValueError('actual unit joint axis required')
                skew=_owned(_skew(unit_axis),(3,3));skew2=_owned(skew@skew,(3,3))
            edges.append((parent,child,_owned(t0,(4,4)),_owned(t1,(4,4)),unit_axis,row,kind,skew,skew2))
            known.add(child)
        if len(known)!=23 or rows!=set(range(17)):raise ValueError('complete link and coordinate coverage required')
        self._edges=tuple(edges)

    def evaluate(self,q_sdk17):
        if isinstance(q_sdk17,(tuple,list)) and any(
                (isinstance(row,np.ndarray) and row.dtype.kind=='b') or
                (isinstance(row,(tuple,list)) and any(isinstance(v,(bool,np.bool_)) for v in row))
                for row in q_sdk17):
            raise ValueError('boolean coordinates are not physical joint values')
        try:q=np.asarray(q_sdk17)
        except (TypeError,ValueError) as exc:raise ValueError('finite real batch of shape (N,17) required') from exc
        if q.ndim!=2 or q.shape[1]!=17 or not q.shape[0] or q.dtype.kind not in 'fiu' or not np.isfinite(q).all():
            raise ValueError('finite real batch of shape (N,17) required')
        q=_owned(q);n=len(q)
        with np.errstate(over='ignore',invalid='ignore'):
            angles=q*self._sign+self._offset
            if not np.isfinite(angles).all():raise ValueError('coordinate hypothesis arithmetic overflow')
            poses=np.empty((n,23,4,4),dtype=np.float64)
            poses[:,self._root]=self._root_transform
            for parent,child,t0,t1,axis,row,kind,skew,skew2 in self._edges:
                before=poses[:,parent]@t0
                if kind=='fixed':poses[:,child]=before@t1
                else:
                    motion=np.broadcast_to(np.eye(4),(n,4,4)).copy()
                    theta=angles[:,row]
                    if kind=='prismatic':motion[:,:3,3]=theta[:,None]*axis
                    else:
                        motion[:,:3,:3]=(np.eye(3)+np.sin(theta)[:,None,None]*skew
                            +(1-np.cos(theta))[:,None,None]*skew2)
                    poses[:,child]=before@motion@t1
            if not np.isfinite(poses).all():raise ValueError('batch FK arithmetic overflow')
        poses.setflags(write=False)
        return MountedFKBatchResult(self.side,self.source_clock,q,self.dof_names,self.link_paths,
            poses,self.assumption_receipt,self.qualification)
