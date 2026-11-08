"""Audit both actual source USDs without creating an Isaac/Kit instance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import active_limits
from go2_pvcnn.control.dh116.tabletop_bimanual_physical_action_adapter import HandPhysicalKinematics


def quantiles(values):
    values = np.asarray(values, dtype=float).ravel()
    return dict(zip(('p50', 'p90', 'p95', 'max'), np.quantile(values, [.5, .9, .95, 1.]).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path('assets/fixed_dual_panda_dh116_v1/dependencies'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-hand', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1165)
    args = parser.parse_args()
    if args.samples_per_hand < 1 or args.output.exists():
        raise ValueError('positive sample count and new independent output required')
    lower, upper = active_limits()
    rng = np.random.default_rng(args.seed)
    states = rng.uniform(lower, upper, (args.samples_per_hand, 6))
    poses = []
    for _ in states:
        quaternion = Rotation.from_rotvec(rng.uniform(-.5, .5, 3)).as_quat()
        poses.append([*rng.uniform(-1., 1., 3), quaternion[3], *quaternion[:3]])
    result = dict(qualification='CPU_source_geometry_only', native_physics_episodes=0,
                  seed=args.seed, samples_per_hand=args.samples_per_hand,
                  same_q_and_base_pose_per_hand=True, sides={})
    for side, filename in (('left', 'd134680ab619_DH116_left.usd'), ('right', '3046ad2ed855_DH116_right.usd')):
        hand = HandPhysicalKinematics.from_usd(args.assets / filename, side)
        position_errors, normal_errors = [], []
        for active, pose in zip(states, poses):
            audit = hand.audit_jacobians(active, pose)
            position_errors.append(np.linalg.norm(audit['position_error_m_per_rad'], axis=1))
            normal_errors.append(np.linalg.norm(audit['normal_error_per_rad'], axis=1))
        result['sides'][side] = dict(source_sha256=hand.source_sha256,
                                    position_derivative_error_m_per_rad=quantiles(position_errors),
                                    normal_derivative_error_per_rad=quantiles(normal_errors),
                                    epsilon_rad=1e-6)
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
