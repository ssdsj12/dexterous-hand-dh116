"""Opt-in clock projection, retaining the frozen original clock validator.

The input is read once. Only exact builtin scalar clock fields use the fast
path; custom containers/types retain the original deepcopy semantics. This
does not cache clocks, change physics, or suppress any identity validation.
"""
from contextlib import contextmanager
import sys
import threading

from . import tabletop_bimanual_table_contacts as original

VERSION = "dh116-owned-clock-projection-v1"
_BASE = original._clock
_KEYS = original.IDENTITY
_PREFIX = "go2_pvcnn.control.dh116."
_BINDINGS = (
    (_PREFIX + "tabletop_bimanual_table_contacts", "_clock"),
    (_PREFIX + "tabletop_material_director_readback", "shared_clock"),
    (_PREFIX + "tabletop_native_parameter_endpoints", "_clock"),
    (_PREFIX + "tabletop_flexible_parameter_readback", "_clock"),
    ("scripts.dh116_pinch_margin_contact_audit", "_clock"),
    ("scripts.validate_dh116_dual_pinch_director_interval", "_clock"),
    ("scripts.validate_dh116_native_parameter_task", "_clock"),
)
_LOCK = threading.RLock()
_ACTIVE = False


def project_clock(reader):
    row = reader()
    if (type(row) is dict and all(key in row for key in _KEYS)
            and all(type(row[key]) in (str, int, float, bool, type(None)) for key in _KEYS)):
        row = {key: row[key] for key in _KEYS}
    return _BASE(lambda: row)


def _loaded_bindings():
    for name, attr in _BINDINGS:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, attr):
            yield module, attr


@contextmanager
def clock_projection_scope():
    """Serial owner scope; no eager imports of SDK-owning consumer modules."""
    global _ACTIVE
    with _LOCK:
        if _ACTIVE:
            raise RuntimeError("clock projection scope cannot nest")
        before = list(_loaded_bindings())
        if any(getattr(module, attr) is not _BASE for module, attr in before):
            raise RuntimeError("unexpected clock validator binding")
        _ACTIVE = True
        try:
            for module, attr in before:
                setattr(module, attr, project_clock)
            yield
        finally:
            # A module imported during the scope inherits the projected source.
            # Restore only this scope's function, never overwrite other edits.
            for module, attr in _loaded_bindings():
                if getattr(module, attr) is project_clock:
                    setattr(module, attr, _BASE)
            _ACTIVE = False
