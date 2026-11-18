"""Startup evidence and cooperative diagnostic stopping, never success gates."""
import os
from pathlib import Path

from .evidence_json import dumps


def write_start_manifest(output, run_config, asset_manifest):
    payload = dumps(dict(status='initialized_not_qualified', grasp_qualified=False,
                         run_config=run_config, asset_manifest=asset_manifest), indent=2)+'\n'
    # Serialize first so invalid values do not publish a partial manifest.
    # Exclusive creation preserves any prior evidence.
    with (Path(output)/'run_manifest.json').open('x') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def check_stop_request(output):
    if (Path(output)/'STOP_REQUESTED').exists():
        raise RuntimeError('operator stop requested; diagnostic not qualified')
