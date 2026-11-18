"""Read exact-link PhysX tensor offsets at an existing diagnostic boundary."""

from pathlib import Path
import hashlib


TARGET_LINKS = (
    "/World/envs/env_0/leftHand/finger12_link",
    "/World/envs/env_0/leftHand/finger14_link",
    "/World/envs/env_0/rightHand/finger14_link",
)
API_MODULE = "omni.physics.tensors.impl.api"
API_PATH = Path(
    "/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/"
    "extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py"
)
API_SHA256 = "bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b"


def source_contract():
    path = API_PATH.resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != API_SHA256:
        raise ValueError("installed Physics Tensor API source differs")
    return dict(api_path=str(path), api_sha256=API_SHA256, target_links=list(TARGET_LINKS),
                max_rigid_body_views=3, phase="after_raw_cooked_capture_at_sdk_step_2")


def _single_link_offsets(simulation_view, path):
    import numpy as np

    view = None
    try:
        view = simulation_view.create_rigid_body_view(path)
        if view.count != 1 or list(view.prim_paths) != [path] or view.max_shapes != 1:
            raise ValueError("exact rigid body and single shape slot required")

        def one_slot(getter):
            value = getter()
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.shape != (1, 1) or array.dtype.kind != "f" or not np.isfinite(array[0, 0]):
                raise ValueError("finite single offset buffer required")
            return float(array[0, 0])

        contact = one_slot(view.get_contact_offsets)
        rest = one_slot(view.get_rest_offsets)
        if contact < 0 or contact <= rest:
            raise ValueError("invalid contact/rest offset pair")
        return contact, rest
    finally:
        view = simulation_view = None


def read_contact_offsets(simulation_view, bindings, *, meters_per_unit, read_clock, expected_clock):
    """Store scalar SDK reads and refuse ambiguous slot identity or timing."""
    rows = [dict(body_path=path, shape_slot=None, native_shape_prim_path=None,
                 contact_offset_m=None, rest_offset_m=None) for path in TARGET_LINKS]
    report = dict(schema="dh116-contact-offset-baseline-v1", links=rows, complete=False,
                  source="Physics Tensor RigidBodyView.get_contact_offsets/get_rest_offsets",
                  identity_scope="exact rigid body and single native shape slot",
                  sdk_clock_before=None, sdk_clock_after=None, error=None)
    try:
        before = read_clock()
        report["sdk_clock_before"] = before
        paths = [binding["actual_body_path"] for binding in bindings]
        if type(meters_per_unit) not in (int, float) or meters_per_unit != 1.0:
            report["error"] = "metre stage required"
            return report
        if any(paths.count(path) != 1 for path in TARGET_LINKS):
            report["error"] = "exactly one existing cooked binding per target link required"
            return report
        if expected_clock.get("sdk_step") != 2 or before != expected_clock:
            report["error"] = "SDK step differs from the frozen capture boundary"
            return report
        if simulation_view is None:
            report["error"] = "existing Physics Tensor SimulationView unavailable"
            return report
        for row in rows:
            try:
                contact, rest = _single_link_offsets(simulation_view, row["body_path"])
                row.update(shape_slot=0, contact_offset_m=contact, rest_offset_m=rest)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                error.__traceback__ = None
            after = read_clock()
            report["sdk_clock_after"] = after
            if after != before:
                for item in rows:
                    item.update(shape_slot=None, contact_offset_m=None, rest_offset_m=None)
                report["error"] = "SDK clock changed during offset readback"
                return report
        report["complete"] = all(row["shape_slot"] == 0 for row in rows)
        return report
    finally:
        simulation_view = read_clock = None
