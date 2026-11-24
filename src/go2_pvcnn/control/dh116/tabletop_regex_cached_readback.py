"""Owned regex snapshots through the audited installed ArticulationData cache.

Instantiate once after binding the original SDK owner. After each SDK physics
step, call the existing ``hand.update(dt)`` exactly once, then ``reader.read()``.
The SDK update already acquires joint velocity for finite differencing; this
reader reuses that same acquisition along with other official lazy getters.

After a zero-step Reset or any external state write, call ``invalidate(reason=...)``
after all writes and before fresh acquisition. Raw PhysX writes do not advance
ArticulationData timestamps automatically. There is no independent step-number
cache here, and no SDK update, physics step, force clearing, COM acquisition or
COM/world-frame conversion is performed by this module.

This source-audited adapter makes no native-equivalence or safety qualification.
The body-link getter performs the SDK's official kinematic update before reading;
its equivalence to the existing raw path must be checked in a real native run.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
from pathlib import Path


SDK_VERSION = "0.54.4"
SDK_SOURCE_SHA256 = {
    "assets/articulation/articulation_data.py": "d69495c66c409fb20823e4e467f9d90e63163b2050b77d2ac8e433221d1d208a",
    "assets/articulation/articulation.py": "fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309",
    "utils/math.py": "881ab27758816475e26d6721bab22c934fd70d0c064392266a90b06f5bbe4149",
    "utils/buffers/timestamped_buffer.py": "1020269322e63b6383c9782f8cf369cbe8b55f52e081bccd79d03ae52712bd59",
    "utils/wrench_composer.py": "095ab1fcf374e4bce96b61430f810c83f3519b7225476ce6e0a6ceb403a5f6a8",
}
_GETTERS = ("joint_pos", "joint_vel", "root_link_pose_w", "root_com_vel_w",
            "body_link_pose_w", "body_com_vel_w")
# Keep derived pose/velocity consumers in the same invalidation transaction.
# Local immutable COM and joint acceleration/history deliberately stay untouched.
_INVALIDATE = (
    "_joint_pos", "_joint_vel", "_root_link_pose_w", "_root_com_vel_w",
    "_body_link_pose_w", "_body_com_vel_w", "_root_link_vel_w", "_root_com_pose_w",
    "_body_link_vel_w", "_body_com_pose_w", "_root_state_w", "_root_link_state_w",
    "_root_com_state_w", "_body_state_w", "_body_link_state_w", "_body_com_state_w",
)


def _type_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def verify_sdk_cache_source(owner):
    """Fail closed on SDK version/source drift; report actual producer types.

    This only audits source bytes. A CPU spy executing the installed getters is
    still a CPU spy, and the metadata explicitly keeps native qualification false.
    No Isaac package is imported and no backend state is read during this gate.
    """
    version = importlib.metadata.version("isaaclab")
    if version != SDK_VERSION:
        raise ValueError(f"unsupported ArticulationData SDK version: {version!r}")
    data = owner.hand.data
    source = Path(type(data).joint_pos.fget.__code__.co_filename).resolve(strict=True)
    if source.parts[-3:] != ("assets", "articulation", "articulation_data.py"):
        raise ValueError("unsupported ArticulationData source path")
    sdk_root = source.parents[2]
    for name in _GETTERS:
        getter = getattr(type(data), name)
        if (not isinstance(getter, property)
                or Path(getter.fget.__code__.co_filename).resolve() != source):
            raise ValueError(f"unsupported ArticulationData source getter: {name}")
    if Path(type(data).update.__code__.co_filename).resolve() != source:
        raise ValueError("unsupported ArticulationData update source")
    files = {}
    for relative, expected in SDK_SOURCE_SHA256.items():
        path = sdk_root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"installed SDK source SHA differs: {relative}")
        files[relative] = {"path": str(path), "sha256": actual}
    return {
        "adapter": "official_articulation_data_owned_readback_v1",
        "sdk_version": version,
        "source_files": files,
        "actual_producer": {"hand_type": _type_name(owner.hand), "data_type": _type_name(data),
                            "view_type": _type_name(owner.hand.root_physx_view), "device": str(owner.device)},
        "native_equivalence_verified": False,
        "qualification": "installed_sdk_source_audit_only",
        "pose_frames": {"root": "root_link_pose_w", "link": "body_link_pose_w"},
        "velocity_frames": {"root": "root_com_vel_w", "link": "body_com_vel_w"},
        "quaternion_conversion": "component_permutation_only_no_normalization",
        "backend_targets": "raw_get_dof_position_targets",
        "snapshot_ownership": "index_select_owned_copy",
        "external_write_contract": "invalidate_after_writes_before_read",
    }


class CachedSideReadback:
    """Reuse the original SDK cache, preserving the raw reader's owned fields."""

    def __init__(self, owner):
        self._metadata = verify_sdk_cache_source(owner)
        self.owner = owner
        self._hand = owner.hand
        self._data = self._hand.data
        self._view = self._hand.root_physx_view

    def _check_identity(self):
        if (self.owner.hand is not self._hand or self._hand.data is not self._data
                or self._hand.root_physx_view is not self._view):
            raise ValueError("cached readback SDK owner/data/view identity changed")

    def metadata(self):
        """Return a detached startup source/producer record, never a native claim."""
        return deepcopy(self._metadata)

    def read(self, env_ids=None):
        """Own q/qd/targets/root/link snapshots in the supplied environment order.

        Poses use actor/link frames. Velocities retain the raw SDK center-of-mass
        reference point. Both quaternion layouts use pure indexing, preserving
        sign, non-unit values and original component bit patterns.
        """
        self._check_identity()
        rows = self.owner.rows(env_ids)
        data = self._data
        state = {
            "q": data.joint_pos.index_select(0, rows),
            "qd": data.joint_vel.index_select(0, rows),
            "backend_targets": self._view.get_dof_position_targets().index_select(0, rows),
            "root_pose_wxyz": data.root_link_pose_w.index_select(0, rows),
            "root_velocity": data.root_com_vel_w.index_select(0, rows),
            "link_pose_wxyz": data.body_link_pose_w.index_select(0, rows),
            "link_velocity": data.body_com_vel_w.index_select(0, rows),
        }
        xyzw = [0, 1, 2, 4, 5, 6, 3]
        state["root_pose_xyzw"] = state["root_pose_wxyz"][..., xyzw]
        state["link_pose_xyzw"] = state["link_pose_wxyz"][..., xyzw]
        return state

    def invalidate(self, *, reason):
        """Force a fresh official acquisition after Reset/external state writes.

        SDK TimestampedBuffer objects are whole-view caches, so invalidation is
        whole-view even for a selected Reset. It does not modify physical peers,
        submitted targets, composed wrenches, cache clock, or derivative history.
        Official joint-velocity writers remain responsible for derivative history.
        """
        self._check_identity()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("explicit external-write invalidation reason required")
        for name in _INVALIDATE:
            getattr(self._data, name).timestamp = float("-inf")
        for name in ("permanent_wrench_composer", "instantaneous_wrench_composer"):
            getattr(self._hand, name)._link_poses_updated = False


def read_cached_side(owner, env_ids=None, *, reader):
    """Read with a startup-verified reader bound to this exact side owner."""
    if reader.owner is not owner:
        raise ValueError("cached readback reader belongs to a different owner")
    return reader.read(env_ids)
