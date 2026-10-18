"""Create a NEW disabled-drive derivative; no physics or motion qualification.

Requires pxr from the installed Isaac environment. Source assets are immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics
from go2_pvcnn.control.dh116.linear_coupling import JOINTS
from go2_pvcnn.control.dh116.usd_contract import axis_vector, full_limits_degrees


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--urdf-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite output')
    pins = {'left': 'e5d6bd3a060dfb6cf4dc8bb2037b3001d87188aacf67dd61d29793821d72a483',
            'right': 'd1d6a53d46820e50ea789b0d5ccb41088ea317ecd355ad145ec4624d84e5ed5e'}
    prepared = []
    lower, upper = full_limits_degrees()
    for side, letter in (('left', 'L'), ('right', 'R')):
        source = args.source / f'DH116_{side}.usd'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != pins[side]:
            raise ValueError('source SHA mismatch')
        urdf = args.urdf_root / f'DH116-V2.0-{letter}/urdf/DH116-V2.0-{letter}.urdf'
        joints = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint')}
        stage = Usd.Stage.Open(str(source))
        # All edits stay in memory; never Save() the source layer.
        rows = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
            name = prim.GetName()
            index = JOINTS.index(name)
            joint = UsdPhysics.RevoluteJoint(prim)
            rotation = joint.GetLocalRot1Attr().Get()
            matrix = Rotation.from_quat([*rotation.GetImaginary(), rotation.GetReal()]).as_matrix()
            original_axis = joint.GetAxisAttr().Get()
            source_axis = np.fromstring(joints[name].find('axis').get('xyz'), sep=' ')
            source_axis /= np.linalg.norm(source_axis)
            if side == 'left' and name == 'finger11' and original_axis == '':
                if np.linalg.norm(matrix @ axis_vector('X') - source_axis) > 1e-7:
                    raise ValueError('left thumb axis repair lacks source evidence')
                joint.CreateAxisAttr('X')
            axis = axis_vector(joint.GetAxisAttr().Get())
            error = float(np.linalg.norm(matrix @ axis - source_axis))
            # Imported four-finger axes were cardinalized. Record, do not claim exact FK.
            if error > .001:
                raise ValueError(f'axis discrepancy {side}/{name}: {error}')
            joint.GetLowerLimitAttr().Set(float(lower[index]))
            joint.GetUpperLimitAttr().Set(float(upper[index]))
            drive = UsdPhysics.DriveAPI(prim, 'angular')
            if not drive or any(a.Get() != 0 for a in
                    (drive.GetStiffnessAttr(), drive.GetDampingAttr(), drive.GetMaxForceAttr())):
                raise ValueError('expected disabled drives')
            rows.append(dict(name=name, original_axis=original_axis,
                             axis=joint.GetAxisAttr().Get(), axis_error=error))
        if len(rows) != 11 or {r['name'] for r in rows} != set(JOINTS):
            raise ValueError('joint set mismatch')
        prepared.append((stage, side, dict(side=side, source_sha256=digest,
                         urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(), axes=rows)))
    args.output.mkdir(parents=True)
    reports = []
    for stage, side, report in prepared:
        target = args.output / f'DH116_{side}.usd'
        stage.GetRootLayer().Export(str(target))
        check = Usd.Stage.Open(str(target))
        for prim in check.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint = UsdPhysics.RevoluteJoint(prim)
                axis_vector(joint.GetAxisAttr().Get())
                i = JOINTS.index(prim.GetName())
                np.testing.assert_allclose([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()],
                                           [lower[i], upper[i]], atol=1e-5, rtol=0)
        report['usd_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        reports.append(report)
    result = dict(qualification='URDF-linear simulation approximation; axis/limit audit only',
                  physics_stepped=False, drives_enabled=False, motion_authorized=False,
                  lower_deg=lower.tolist(), upper_deg=upper.tolist(), reports=reports)
    (args.output / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
