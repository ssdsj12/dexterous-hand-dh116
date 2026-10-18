"""Read USD frames, verify linear-model point Jacobians; no physics stepping.

Writes a NEW disabled-drive linear-mimic derivative and evidence, not a Play
asset qualification. Probe points are distal-link origins, not calibrated pads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--asset-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise RuntimeError('Refusing to overwrite output')
app = AppLauncher(args).app

try:
    from pxr import Usd, UsdPhysics, PhysxSchema
    from scipy.spatial.transform import Rotation
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS, MIMIC, expand, fold_jacobian
    from go2_pvcnn.control.dh116.usd_contract import axis_vector

    def frame(position, quaternion):
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
        t[:3, 3] = position
        return t

    args.output.mkdir(parents=True)
    reports = []
    source_pins = {
        'left': '31ba8d541058e89fc17e20af03c317499726b41f4a304f6c0af315a37dcf8f32',
        'right': '63d98413dd870ab58f0775170b47e1db45146db0ab9f24fa839d80c28bd3ad60',
    }
    for side in ('left', 'right'):
        source = args.asset_root / f'DH116_{side}.usd'
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_pins[side]:
            raise ValueError('Unexpected DH116 source; coupling requires re-audit')
        s = Usd.Stage.Open(str(source))
        joints = {p.GetName(): UsdPhysics.Joint(p) for p in s.Traverse()
                  if p.IsA(UsdPhysics.Joint) and p.GetName() != 'root_joint'}
        assert set(JOINTS) == {n for n, j in joints.items() if j.GetPrim().IsA(UsdPhysics.RevoluteJoint)}
        edges = []
        for name, j in joints.items():
            a, b = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            assert len(a) == len(b) == 1
            axis = np.zeros(3)
            if name in JOINTS:
                axis_name = UsdPhysics.RevoluteJoint(j.GetPrim()).GetAxisAttr().Get()
                axis = axis_vector(axis_name)
            edges.append((name, str(a[0]), str(b[0]),
                          frame(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                          frame(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()), axis))
        roots = {e[1] for e in edges} - {e[2] for e in edges}
        assert len(roots) == 1
        root = roots.pop()

        def fk(q):
            full = dict(zip(JOINTS, expand(q)))
            poses, ancestors, axes = {root: np.eye(4)}, {root: []}, {}
            pending = list(edges)
            while pending:
                ready = [e for e in pending if e[1] in poses]
                if not ready:
                    raise ValueError('Disconnected or cyclic USD joint tree')
                for name, parent, child, t0, t1, axis in ready:
                    before = poses[parent] @ t0
                    r = np.eye(4)
                    if name in full:
                        r[:3, :3] = Rotation.from_rotvec(axis*full[name]).as_matrix()
                        axes[name] = (before[:3, 3], before[:3, :3]@axis)
                    poses[child] = before @ r @ np.linalg.inv(t1)
                    ancestors[child] = ancestors[parent]+([name] if name in full else [])
                    pending.remove(next(e for e in pending if e[0] == name))
            return poses, ancestors, axes

        max_error = 0.
        probes = 0
        for value in (.1, .3, .7):
            q = np.full(6, value)
            poses, ancestors, axes = fk(q)
            for suffix in ('finger14_link','finger23_link','finger33_link','finger43_link','finger53_link'):
                body = next(b for b in poses if b.endswith('/'+suffix))
                point = poses[body][:3, 3]
                j = np.zeros((3,11))
                for name in ancestors[body]:
                    origin, axis = axes[name]
                    j[:,JOINTS.index(name)] = np.cross(axis, point-origin)
                reduced = fold_jacobian(j, JOINTS)
                fd = np.column_stack([(fk(q+np.eye(6)[i]*1e-6)[0][body][:3,3]-fk(q-np.eye(6)[i]*1e-6)[0][body][:3,3])/2e-6 for i in range(6)])
                max_error = max(max_error, float(np.abs(reduced-fd).max()))
                probes += 1
        assert max_error < 1e-7, max_error
        for follower, master, ratio, offset in MIMIC:
            j = joints[follower].GetPrim()
            for coupled_name in (follower, master):
                if UsdPhysics.RevoluteJoint(joints[coupled_name].GetPrim()).GetAxisAttr().Get() != 'X':
                    raise ValueError('Expected X axis for this mimic pair')
            # PhysX relation: q_follower = -gearing*q_reference - offset.
            api = PhysxSchema.PhysxMimicJointAPI.Apply(j, 'rotX')
            api.CreateReferenceJointRel().SetTargets([joints[master].GetPath()])
            api.CreateReferenceJointAxisAttr('rotX')
            api.CreateGearingAttr(-ratio)
            api.CreateOffsetAttr(-float(np.degrees(offset)))
        for name in JOINTS:
            drive = UsdPhysics.DriveAPI.Apply(joints[name].GetPrim(), 'angular')
            drive.CreateStiffnessAttr(0.)
            drive.CreateDampingAttr(0.)
            drive.CreateMaxForceAttr(0.)
        target = args.output / f'DH116_{side}.usd'
        s.GetRootLayer().Export(str(target))
        check = Usd.Stage.Open(str(target))
        for follower, master, ratio, offset in MIMIC:
            p = check.GetPrimAtPath(joints[follower].GetPath())
            api = PhysxSchema.PhysxMimicJointAPI(p, 'rotX')
            assert api and abs(api.GetGearingAttr().Get()+ratio)<1e-7
            assert abs(api.GetOffsetAttr().Get()+np.degrees(offset))<1e-6
            assert api.GetReferenceJointRel().GetTargets() == [joints[master].GetPath()]
        reports.append(dict(side=side,point_probes=probes,max_jacobian_error_m_per_rad=max_error,
                            usd_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        print('DH116_JACOBIAN', reports[-1], flush=True)
    (args.output/'verification.json').write_text(json.dumps(dict(
        qualification='USD-derived kinematics; linear simulation approximation',
        physics_stepped=False, motion_authorized=False, drives_enabled=False,
        points='five distal link origins, not calibrated pads', reports=reports),indent=2)+'\n')
    print('DH116_KINEMATICS_COMPLETE', flush=True)
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
