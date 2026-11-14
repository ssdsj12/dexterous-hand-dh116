"""Audit existing DH116 candidate geometry on CPU for a distinct wrap seed."""
import argparse
import hashlib
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_teacher_wrap_preflight import audit_wrap_candidates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--native-contract', type=Path, required=True)
    parser.add_argument('--native-contract-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    source, native = args.source.read_bytes(), args.native_contract.read_bytes()
    if (hashlib.sha256(source).hexdigest() != args.source_sha256
            or hashlib.sha256(native).hexdigest() != args.native_contract_sha256):
        parser.error('source or native contract SHA256 does not match')
    result = audit_wrap_candidates(json.loads(source), json.loads(native))
    result.update(source_sha256=args.source_sha256, native_contract_sha256=args.native_contract_sha256)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
