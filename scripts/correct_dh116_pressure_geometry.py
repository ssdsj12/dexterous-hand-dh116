"""Recompute pressure direction from source USD; no physics or asset mutation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
from go2_pvcnn.control.dh116.tabletop_pressure_geometry import pressure_direction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('new output path required; preserve original initialization')
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(args.initialization.read_text())
    if payload.get('format') != 'dh116-adaptive-grasp-initialization-v1':
        raise ValueError('expected adaptive initialization')
    manifest = json.loads((root/'assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
    compensation = []
    derived = {}
    for hi, side in enumerate(('left', 'right')):
        source = Path(next(x['path'] for x in manifest['sources'] if x['kind']=='dh116' and x['side']==side))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if payload['source_contract']['source_sha256'][side] != digest:
            raise ValueError('initialization/source USD mismatch')
        model = HandPointFK(source, side, payload['roots'][hi])
        q = np.asarray(payload['grasp_command'][6*hi:6*hi+6])
        jac = model.jacobian(q)
        points = model.points(q)
        np.testing.assert_allclose(points, payload['diagnostics']['hands'][side]['closed_pad_positions'], atol=1e-7, rtol=0)
        direction, palm = pressure_direction(jac, [1.,0.,0.])
        delta = model.points(q+.08*direction)+.08*palm-points
        if np.max(abs(delta[:,2])) > .0005:
            raise ValueError('pressure compensation exceeds 0.5 mm local height error')
        payload['diagnostics']['hands'][side]['closure_direction_active_unit'] = direction.tolist()
        compensation.append(palm.tolist())
        derived[side] = dict(source_sha256=digest, point_jacobian_m_per_rad=jac.tolist(),
            nonlinear_delta_at_offset_008_m=delta.tolist(), reference_active_rad=q.tolist())
    payload['pressure_geometry'] = dict(version='diametric_midpoint_feedforward_v1',
        coupling='source_urdf_linear_approximation', palm_compensation_m_per_rad=compensation,
        parent_initialization_sha256=hashlib.sha256(args.initialization.read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        approximation='local first order; actual palm motion remains rate/clearance/tracking limited',
        derivation=derived)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), pressure_geometry=payload['pressure_geometry'])))


if __name__ == '__main__':
    main()
