#!/usr/bin/env python3
"""Read-only original two-hand Power mesh audit and bounded native capture plan."""
import argparse
import hashlib
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_teacher_power_surface_audit import (
    audit_asset, load_locked_pads, prepare_capture_inventory,
)
from scripts.verify_dh116_cooked_convex_hulls_native import _sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--capture-dir', type=Path)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--lock-sha256')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if any(v is not None for v in (args.capture_dir, args.lock, args.lock_sha256)):
        if not all(v is not None for v in (args.capture_dir, args.lock, args.lock_sha256)):
            parser.error('capture-dir, lock, lock-sha256 must be supplied together')
        pads = load_locked_pads(args.capture_dir, args.lock, args.lock_sha256)
    else:
        pads = {}
    sources = _sources(args.project)
    reports = [audit_asset(sources[side]['path'], side, locked_pads=pads) for side in ('left','right')]
    result = dict(schema='dh116-teacher-power-two-hand-source-audit-v1', hands=reports,
                  native_capture_input=prepare_capture_inventory(reports), native_executed=False,
                  physics_modified=False, power_qualified=False,
                  registration_scope='no runtime registration queried; historical distal SDF evidence is separate',
                  closure_evidence='actual reaction forces required; USD triangles and hulls cannot establish closure')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'counts': {r['side']: r['counts'] for r in reports}}, indent=2))


if __name__ == '__main__':
    main()
