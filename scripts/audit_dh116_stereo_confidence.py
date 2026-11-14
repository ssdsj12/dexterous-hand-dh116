"""Audit the confidence of existing stereo candidate pools on CPU."""
import argparse
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_stereo_confidence_audit import audit_confidence, load_replay_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='append', required=True, nargs=2, metavar=('SUMMARY', 'SHA256'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    rows, inputs = [], []
    for path, expected in args.replay:
        cohort, binding = load_replay_rows(path, expected)
        rows.extend(cohort)
        inputs.append(binding)
    report = dict(audit_confidence(rows), inputs=inputs)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
