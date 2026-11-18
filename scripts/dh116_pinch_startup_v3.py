"""Narrow v3 startup/empty-acquisition diagnostics; v2 physics remains frozen."""
import os
from pathlib import Path
import sys

from scripts import dh116_pinch_acquisition_v2 as native
from scripts.probe_dh116_physx_lifetime import LifetimeTrace as PriorTrace,write_json
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116.tabletop_deformable_rod_bridge import NativeOwnershipTrace
from go2_pvcnn.control.dh116.tabletop_native_lifetime_audit import (
    VERSION as TRACE_VERSION,snapshot_weakrefs)

VERSION="dh116-single-left-pinch-v3"
BOUNDARY_VERSION="dh116-pinch-import-boundary-v3"


def assert_clean_import_boundary():
    loaded=sorted(name for name in sys.modules
                  if name.split(".")[0] in ("pxr","PhysxSchema"))
    if loaded:
        raise RuntimeError("USD modules loaded before AppLauncher: "+", ".join(loaded))
    return dict(version=BOUNDARY_VERSION,phase="before_isaaclab_app_import",
                loaded_usd_modules=[],native_execution_performed=False)


def snapshot_for_startup(references):
    if not references:
        return dict(version=TRACE_VERSION,resources={},native_release_verified=False,
                    observation="no_registered_owners")
    return snapshot_weakrefs(references)


class StartupTrace(PriorTrace):
    def checkpoint(self,phase):
        if self.references:
            return super().checkpoint(phase)
        # Construction can fail before even the simulation weakref is acquired.
        # Empty evidence is unavailable ownership, never successful release.
        NativeOwnershipTrace.checkpoint(self,phase)
        name=phase.replace(" ","_").replace(",","")
        write_json(self.path.parent/(name+".refs.json"),snapshot_for_startup({}))
        with (self.path.parent/(name+".maps")).open("x") as stream:
            stream.write(Path("/proc/self/maps").read_text())
            stream.flush()
            os.fsync(stream.fileno())


def run_lifecycle(*args,**kwargs):
    try:
        with bindings((native,"VERSION",VERSION),(native,"LifetimeTrace",StartupTrace),
                      (native,"snapshot_weakrefs",snapshot_for_startup)):
            return native.run_lifecycle(*args,**kwargs)
    finally:
        args=kwargs=None


def execute(args,contract,cfg,project):
    boundary=assert_clean_import_boundary()
    write_json(args.output/"before_app_launcher.json",boundary)
    # Existing execute calls its module's run_lifecycle, so bind the diagnostic
    # pieces directly rather than rebinding run_lifecycle recursively.
    with bindings((native,"VERSION",VERSION),(native,"LifetimeTrace",StartupTrace),
                  (native,"snapshot_weakrefs",snapshot_for_startup)):
        return native.execute(args,contract,cfg,project)
