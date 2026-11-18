"""USD-joint-frame FK of one mounted Panda/DH116 side, not a physics solver."""
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
from .linear_coupling import JOINTS,expand
from .contact_frames import point_jacobian
from .usd_contract import axis_vector


def _frame(position,quaternion):
    t=np.eye(4);t[:3,3]=position
    t[:3,:3]=Rotation.from_quat([*quaternion.GetImaginary(),quaternion.GetReal()]).as_matrix()
    return t


class SideKinematics:
    def __init__(self,stage,side):
        if side not in ('left','right'):raise ValueError('invalid side')
        self.side=side;self.prefix=f'/FixedDualPandaDH116/{side}_arm/'
        self.names=tuple(f'{side}_panda_joint{i}' for i in range(1,8))+JOINTS
        edges=[]
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):continue
            j=UsdPhysics.Joint(prim);a=j.GetBody0Rel().GetTargets();b=j.GetBody1Rel().GetTargets()
            if len(a)!=1 or len(b)!=1:continue  # world fixed joint, not a kinematic edge
            a,b=str(a[0]),str(b[0])
            if not a.startswith(self.prefix) or not b.startswith(self.prefix):continue
            moving=prim.IsA(UsdPhysics.RevoluteJoint)
            if not moving and not prim.IsA(UsdPhysics.FixedJoint):raise ValueError('unsupported joint')
            edges.append((prim.GetName(),a,b,
                _frame(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get()),
                np.linalg.inv(_frame(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get())),
                axis_vector(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()) if moving else None))
        if {e[0] for e in edges if e[5] is not None}!=set(self.names):raise ValueError('joint coverage mismatch')
        roots={e[1] for e in edges}-{e[2] for e in edges}
        if len(roots)!=1 or len({e[2] for e in edges})!=len(edges):raise ValueError('invalid joint tree')
        self.root=roots.pop()
        self.root_pose=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(self.root))).T
        self.edges=[];known={self.root}
        while edges:
            ready=[e for e in edges if e[1] in known]
            if not ready:raise ValueError('disconnected/cyclic tree')
            for e in ready:
                self.edges.append(e);known.add(e[2]);edges.remove(e)
        self.bodies={p.rsplit('/',1)[-1]:p for p in known}
        if len(self.bodies)!=len(known):raise ValueError('ambiguous body names')

    def fk(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(13,) or not np.isfinite(q).all():raise ValueError('expected finite 13D active state')
        full=dict(zip(self.names,np.r_[q[:7],expand(q[7:])]))
        poses={self.root:self.root_pose};axes={};ancestors={self.root:[]}
        for name,parent,child,t0,t1,axis in self.edges:
            before=poses[parent]@t0;motion=np.eye(4)
            if axis is not None:
                motion[:3,:3]=Rotation.from_rotvec(axis*full[name]).as_matrix()
                axes[name]=(before[:3,3],before[:3,:3]@axis)
            poses[child]=before@motion@t1
            ancestors[child]=ancestors[parent]+([name] if axis is not None else [])
        return poses,axes,ancestors

    def _point(self,q,body,local):
        if body not in self.bodies:raise ValueError('unknown body')
        local=np.asarray(local,dtype=float)
        if local.shape!=(3,) or not np.isfinite(local).all():raise ValueError('invalid local point')
        poses,axes,ancestors=self.fk(q);path=self.bodies[body];t=poses[path]
        return t[:3,:3]@local+t[:3,3],axes,ancestors[path]

    def point(self,q,body,local):
        return self._point(q,body,local)[0]

    def point_jacobian(self,q,body,local):
        point,axes,ancestors=self._point(q,body,local)
        spatial=np.zeros((6,18))
        for name in ancestors:
            origin,axis=axes[name];i=self.names.index(name)
            spatial[:3,i]=np.cross(axis,point-origin);spatial[3:,i]=axis
        return point_jacobian(spatial,np.zeros(3),self.names,self.side)
