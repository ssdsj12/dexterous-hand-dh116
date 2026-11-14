"""Read-only CPU audit of an independently SHA-pinned PhysX calibration ledger."""
import argparse
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_stereo_contact_calibration import audit_archived_contact_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--ledger-sha256', required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--object-actor', required=True)
    parser.add_argument('--partner-actor', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New report file; never overwritten')
    args = parser.parse_args(argv)
    report = audit_archived_contact_sources(args.ledger, args.inventory, args.stage,
        raw_sha256=args.ledger_sha256, inventory_sha256=args.inventory_sha256,
        object_path=args.object_actor, expected_partner=args.partner_actor)
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(payload)


if __name__ == '__main__':
    main()
