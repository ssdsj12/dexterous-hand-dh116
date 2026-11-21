"""Read-only admission from a committed17-slot Reset to an exclusive exchange."""
from copy import deepcopy
from . import tabletop_supported_mesh_reset as reset_reader
from .tabletop_power_shared_rod import PowerSharedRodRuntime, PROXIMAL_POWER_MESH_NAMES
from .tabletop_deformable_r0_replay import _digest

VERSION = "dh116-fresh-supported-exchange-binding-v1"


def validate_fresh_supported_reset(sim, rod, *, reset_receipt, mesh_plan):
    """No action, Reset, settling step, owner transfer or contact claim."""
    try:
        if (type(rod) is not PowerSharedRodRuntime
                or tuple(rod.mesh_names) != PROXIMAL_POWER_MESH_NAMES):
            raise ValueError("actual seventeen-slot supported rod owner required")
        clock = deepcopy(rod.clock())
        epoch = tuple(clock[k] for k in ("run_uuid","env_id","rod_uid","episode_uid","reset_id"))
        if getattr(rod, "_action_exchange_claimed", False):
            previous = getattr(rod, "_fresh_action_exchange_epoch", None)
            if (not isinstance(previous, tuple) or len(previous) != 5
                    or previous[:3] != epoch[:3] or previous[3] == epoch[3]
                    or type(previous[4]) is not int or epoch[4] <= previous[4]):
                raise ValueError("supported rod already attached to an action exchange")
        if (clock["clock_valid"] is not True or clock["rod_tick"] != 0
                or clock["reset_id"] < 1 or rod._base.last is not None
                or clock != reset_receipt["after_clock"]):
            raise ValueError("same actual fresh supported Reset clock required")
        reset_reader._plan(mesh_plan, rod)
        recorded = reset_receipt["mesh_state_readback"]
        reset_reader.validate_supported_readback(recorded,reset_receipt=reset_receipt,mesh_plan=mesh_plan)
        if recorded["after"]["source_instance_id"] != id(rod.rope):
            raise ValueError("recorded supported Reset belongs to another native owner")
        current = reset_reader.capture_supported_mesh_state(sim,rod)
        reset_reader.validate_supported_capture(current)
        if (current["source_instance_id"] != id(rod.rope)
                or current["clock_before"] != recorded["after"]["clock_after"]
                or current["clock_after"] != current["clock_before"]
                or _digest(current["snapshot"]) != _digest(recorded["after"]["snapshot"])
                or rod.clock() != clock):
            raise ValueError("live supported mesh state/clock drifted since committed Reset")
        return dict(version=VERSION,clock=clock,reset_epoch=epoch,source_instance_id=id(rod.rope),
            mesh_plan_sha256=mesh_plan["sha256"],mesh_snapshot_sha256=_digest(current["snapshot"]),
            reaction_source="commanded_reset_zero_not_measured_contact",
            contact_measurement_available=False,new_sdk_steps=0,new_solver_steps=0,
            native_qualification=False,teacher_admission=False)
    finally:
        sim = rod = None
