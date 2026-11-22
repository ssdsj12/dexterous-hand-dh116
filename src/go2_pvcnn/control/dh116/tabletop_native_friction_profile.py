"""Independent friction-observation source identity for full Teacher integration.

This does not relax BackendProfile or grant production context/reward admission.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path

from . import tabletop_contact_friction_observation_source as source
from .tabletop_native_teacher_profile import BackendProfile, FileIdentity, ROOT, checked_file, digest


@dataclass(frozen=True, slots=True, init=False)
class FrictionBackendProfileV2:
    version: str
    parent: BackendProfile
    binary_path: str
    binary_sha256: str
    source_tree_sha256: str
    patch_sha256: str
    inventory: tuple
    inventory_sha256: str
    digest: str

    def __init__(self):
        raise TypeError("FrictionBackendProfileV2 must load actual source files")

    @classmethod
    def load(cls, project=ROOT):
        if Path(project).resolve() != ROOT:
            raise ValueError("current source-controlled project root required")
        parent = BackendProfile.load(project)
        basis = source.source_basis()
        rows = {row.path: row for row in parent.inventory}
        def add(path, expected, role):
            path = Path(path).absolute()
            checked_file(path, expected)
            key = str(path)
            if key in rows and rows[key].sha256 != expected:
                raise ValueError("conflicting friction profile source identity")
            rows[key] = FileIdentity(key, expected, role)
        for path, expected in basis["sources"].items():
            add(path, expected, "friction_source")
        for path, expected in basis["local_sources"].items():
            add(ROOT / path, expected, "friction_local_source")
        # These new, editable readers are versioned by actual byte identity.
        import hashlib
        for path in (Path(source.__file__), Path(__file__)):
            add(path, hashlib.sha256(path.read_bytes()).hexdigest(), "friction_reader")
        values = dict(version="dh116-native-friction-backend-profile-v2", parent=parent,
            binary_path=basis["backend_path"], binary_sha256=basis["backend_sha256"],
            source_tree_sha256=basis["source_tree_sha256"],
            patch_sha256=basis["patch_delta_sha256"],
            inventory=tuple(sorted(rows.values(), key=lambda row: row.path)))
        values["inventory_sha256"] = digest([asdict(row) for row in values["inventory"]])
        values["digest"] = digest({key: asdict(value) if key == "parent"
            else [asdict(row) for row in value] if key == "inventory" else value
            for key, value in values.items()})
        result = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(result, key, value)
        return result

    def verify(self, project=ROOT):
        if type(self) is not FrictionBackendProfileV2 or type(self).load(project) != self:
            raise ValueError("current friction backend source identity differs")
        return self


_owner_state = None


@contextmanager
def friction_profile_reader_scope(profile):
    """Explicit new reader entry; old exact-type gates intentionally reject it."""
    global _owner_state
    if type(profile) is not FrictionBackendProfileV2:
        raise ValueError("source-loaded FrictionBackendProfileV2 required")
    # Reentrant scopes retain this owner's existing immutable profile. The
    # source context itself rechecks current bytes before every entry.
    if source._depth:
        from . import tabletop_native_teacher_profile as legacy
        import threading
        if source._thread != threading.get_ident():
            raise ValueError("friction profile scope belongs to another thread")
        if _owner_state is None or asdict(profile) != _owner_state:
            raise ValueError("nested friction profile owner identity differs")
        if legacy._profile_reader_depth:
            raise ValueError("friction profile requires idle legacy reader")
    else:
        profile.verify()
    # Digest validation also catches a forged profile on a nested path.
    expected = digest({key: value for key, value in asdict(profile).items() if key != "digest"})
    if profile.digest != expected:
        raise ValueError("friction backend profile identity differs")
    with source.source_scope() as basis:
        if (basis["backend_sha256"] != profile.binary_sha256
                or basis["source_tree_sha256"] != profile.source_tree_sha256):
            raise ValueError("friction profile and reader identity differ")
        outer = _owner_state is None
        if outer:
            _owner_state = deepcopy(asdict(profile))
        try:
            yield basis
        finally:
            if outer:
                _owner_state = None
