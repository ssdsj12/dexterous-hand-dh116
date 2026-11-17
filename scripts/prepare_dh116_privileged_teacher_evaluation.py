"""Freeze independent Teacher test conditions from SHA-bound R0 artifacts."""
import argparse
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_privileged_cli import load_config
from go2_pvcnn.control.dh116.tabletop_privileged_evaluation import (
    freeze_teacher_cold_manifest, inspect_trained_teacher_checkpoint,
)
from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256, require_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'native-contract', 'primary-initialization',
                 'secondary-initialization', 'camera-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--native-contract-sha256', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--count-per-style', type=int, default=100)
    args = parser.parse_args(argv)
    if args.count_per_style < 100:
        parser.error('at least 100 cold test trials per grasp style required')
    require_sha(args.native_contract_sha256, 'reviewed native task contract SHA256')
    if file_sha256(args.native_contract) != args.native_contract_sha256:
        raise ValueError('native task contract SHA256 differs from the reviewed artifact')
    binding = json.loads(args.native_contract.read_text(encoding='utf-8'))
    config, ppo = load_config(args.config)
    inspection = inspect_trained_teacher_checkpoint(args.checkpoint, args.checkpoint_sha256,
        config, ppo, contract=binding['inherited_contract'], task_identity=binding['task_identity'])
    result = freeze_teacher_cold_manifest(args.checkpoint, binding, args.primary_initialization,
        args.secondary_initialization, args.output, seed=args.seed, count_per_style=args.count_per_style,
        camera_sha256=StereoCalibration.from_json(args.camera_config).sha256)
    print(json.dumps(dict(frozen_checkpoint=inspection, **result), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
