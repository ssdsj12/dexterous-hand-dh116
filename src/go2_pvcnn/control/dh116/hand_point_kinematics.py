"""Offline source USD point FK; never starts Isaac or modifies a USD stage."""
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics
from go2_pvcnn.control.dh116.usd_kinematics import _frame
from go2_pvcnn.control.dh116.usd_contract import axis_vector
from go2_pvcnn.control.dh116.linear_coupling import JOINTS, expand
from go2_pvcnn.control.dh116.native_scene import _PAD_LOCAL_POINTS

class HandPointFK:
    def __init__(self, source, side, root_pose):
        r = np.asarray(root_pose, dtype=float)
        if (side not in ('left', 'right') or r.shape != (7,)
                or not np.isfinite(r).all()
                or not np.isclose(np.linalg.norm(r[3:]), 1., atol=1e-6, rtol=0)):
            raise ValueError('finite root pose with unit quaternion and valid hand side required')
        self.side = side
        stage = Usd.Stage.Open(str(source))
        edges = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(a) != 1 or len(b) != 1:
                continue
            if not prim.IsA(UsdPhysics.RevoluteJoint) and not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError('unsupported hand joint')
            axis = (axis_vector(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
                    if prim.IsA(UsdPhysics.RevoluteJoint) else None)
            edges.append((prim.GetName(), str(a[0]), str(b[0]),
                _frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                np.linalg.inv(_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())), axis))
        roots = set(e[1] for e in edges)-set(e[2] for e in edges)
        if {e[0] for e in edges if e[5] is not None} != set(JOINTS):
            raise ValueError('expected eleven DH116 joints')
        if len(roots) != 1:
            raise ValueError('expected one hand root')
        self.root = roots.pop()
        self.root_pose = np.eye(4)
        self.root_pose[:3, 3] = r[:3]
        self.root_pose[:3, :3] = Rotation.from_quat(np.r_[r[4:], r[3]]).as_matrix()
        self.edges = []
        known = {self.root}
        while edges:
            ready = [e for e in edges if e[1] in known]
            if not ready:
                raise ValueError('disconnected/cyclic hand tree')
            for e in ready:
                self.edges.append(e)
                known.add(e[2])
                edges.remove(e)

    def poses(self, q):
        full = dict(zip(JOINTS, expand(q)))
        poses = {self.root: self.root_pose}
        for name, parent, child, t0, t1, axis in self.edges:
            motion = np.eye(4)
            if axis is not None:
                motion[:3, :3] = Rotation.from_rotvec(axis*full[name]).as_matrix()
            poses[child] = poses[parent]@t0@motion@t1
        return poses

    def points(self, q):
        poses, out = self.poses(q), []
        for body, digit in [('finger14_link', 'thumb'), ('finger23_link', 'index')]:
            t = next(v for k, v in poses.items() if k.endswith('/'+body))
            out.append(t[:3, :3]@_PAD_LOCAL_POINTS[self.side][digit]+t[:3, 3])
        return np.array(out)

    def jacobian(self, q, epsilon=1e-6):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('positive finite difference step required')
        q = np.asarray(q)
        return np.stack([(self.points(q+np.eye(6)[i]*epsilon)-
                          self.points(q-np.eye(6)[i]*epsilon))/(2*epsilon)
                         for i in range(6)], axis=-1)
