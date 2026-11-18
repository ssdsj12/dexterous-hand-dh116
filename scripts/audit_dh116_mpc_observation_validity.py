"""Generate a SHA-bound MPC simulator observation audit without Isaac or training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from go2_pvcnn.control.dh116.tabletop_mpc_observation_validity import (
    GROUPS,
    PROOF_SCOPE,
    PROOF_VERSION,
    audit_observation_mask,
)
from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import file_identity


def audit_files(archive_path, archive_sha, sidecar_path, sidecar_sha, output):
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError('output must be a new file in an existing directory')
    archive_id = file_identity(archive_path)
    sidecar_id = file_identity(sidecar_path)
    if archive_id['sha256'] != archive_sha or sidecar_id['sha256'] != sidecar_sha:
        raise ValueError('archive or native sequence sidecar SHA differs')
    archive = torch.load(archive_path, map_location='cpu', weights_only=True)
    # Existing SDK diagnostic arrays use numpy in the trusted, SHA-bound sidecar.
    globals_ = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                *[type(np.dtype(name)) for name in ('float32', 'float64', 'int32', 'int64', 'uint8', 'bool')]]
    with torch.serialization.safe_globals(globals_):
        sidecar = torch.load(sidecar_path, map_location='cpu', weights_only=True)
    mask = audit_observation_mask(archive, sidecar)
    if file_identity(archive_path) != archive_id or file_identity(sidecar_path) != sidecar_id:
        raise ValueError('source SHA changed during observation audit')
    proof = dict(version=PROOF_VERSION, scope=PROOF_SCOPE, archive=archive_id,
        sequence_sidecar=sidecar_id, run_uuid=archive['run_uuid'],
        group_sources=[dict(start=start, stop=stop, source=source)
                       for (start, stop), source in GROUPS.items()],
        qualified_for_bc=False, force_contact_qualification=False,
        counts_as_Student_success=False, privileged_valid=mask,
        checked_rows=len(archive['sample_metadata']))
    with output.open('xb') as stream:
        torch.save(proof, stream)
    return dict(output=file_identity(output), checked_rows=proof['checked_rows'],
                exact_columns_all_rows=int(mask.all(0).sum()),
                per_column_valid_counts=mask.sum(0).tolist(),
                scope=PROOF_SCOPE, qualified_for_bc=False, force_contact_qualification=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--sequence-sidecar', type=Path, required=True)
    parser.add_argument('--sidecar-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_files(args.archive, args.archive_sha256, args.sequence_sidecar,
                                args.sidecar_sha256, args.output), allow_nan=False))


if __name__ == '__main__':
    main()
