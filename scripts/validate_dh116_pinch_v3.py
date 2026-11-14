"""V3 identity and import-boundary gate above the unchanged full v2 audit."""
from pathlib import Path
from scripts import validate_dh116_pinch_v2 as prior
from scripts.dh116_pinch_startup_v3 import VERSION,BOUNDARY_VERSION
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings


def audit_outcome(output,outer_exit,cleanup):
    with bindings((prior,"VERSION",VERSION)):
        result=prior.audit_outcome(output,outer_exit,cleanup)
    if not result["lifecycle_qualified"]:
        return result
    try:
        boundary=prior.read(Path(output)/"before_app_launcher.json")
        prior.require(boundary==dict(version=BOUNDARY_VERSION,
            phase="before_isaaclab_app_import",loaded_usd_modules=[],
            native_execution_performed=False),"v3 clean import boundary missing")
    except (OSError,ValueError,KeyError,TypeError) as error:
        result.update(lifecycle_qualified=False,reason=f"{type(error).__name__}: {error}")
    return result
