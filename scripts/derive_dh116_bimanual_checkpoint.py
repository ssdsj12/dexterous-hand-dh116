"""Bind a qualified physical adapter to a complete V3 tanh PPO snapshot.

This offline operation creates independent V4 .pt/.json outputs. It requires
the original source SHA-256, completed native mapping and pinch Teacher reports,
and the exact adapter contract used by that Teacher. It creates no Kit instance.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from go2_pvcnn.control.dh116.tabletop_bimanual_checkpoint_contract import derive_physical_adapter_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='complete, read-only V3 tanh_gaussian PPO checkpoint')
    parser.add_argument('--output', type=Path, required=True, help='new independent V4 .pt path; .json sidecar must also be new')
    parser.add_argument('--source-sha256', required=True, help='independently frozen SHA-256 of the source checkpoint')
    parser.add_argument('--adapter-contract', type=Path, required=True,
                        help='exact direct adapter JSON, or object containing physical_action_adapter')
    parser.add_argument('--teacher-report', type=Path, required=True,
                        help='native Teacher report with at least four qualified pinch first episodes')
    parser.add_argument('--mapping-report', type=Path, required=True,
                        help='completed native 53-case mapping summary with no failures')
    args = parser.parse_args()
    adapter = json.loads(args.adapter_contract.read_text())
    if isinstance(adapter, dict) and 'physical_action_adapter' in adapter:
        adapter = adapter['physical_action_adapter']
    report = derive_physical_adapter_checkpoint(args.input, args.output,
        expected_source_sha256=args.source_sha256, adapter_contract=adapter,
        teacher_report_path=args.teacher_report, mapping_report_path=args.mapping_report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
