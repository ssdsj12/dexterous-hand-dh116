"""Immutable source identity for the NEW common66/rod41x13/context50 Teacher.

Loading proves current CPU file provenance only. No native module is imported,
no owner is created, and historical measured models retain their own identity.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
import threading

VERSION = "dh116-native-teacher-backend-profile-v1"
MODEL_SCHEMA_VERSION = "dh116-native-context-teacher-tanh-v1"
INPUT_SCHEMA_VERSION = "dh116-native-context-teacher-input-v1"
CONTEXT_SCHEMA_VERSION = "dh116-native-teacher-context50-v1"
INPUT_KEYS = ("common371", "common_valid", "rod_nodes", "rod_mask",
              "task_context", "task_context_valid")
TARGET_SCHEMA_VERSION = "fixed_material_nodes41_set_distance_v1"
ROD_SCHEMA_VERSION = "dh116-material-nodes41-features13-v1"
ROOT = Path(__file__).resolve().parents[4]
SHA = re.compile(r"[0-9a-f]{64}")
SDF_DIR = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-f4-sdf-register-rK7gw0/bake")
SDF_SUMMARY_SHA = "db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def checked_file(path, expected):
    """Reject symlinks/nonregular files and hash the bytes on every call."""
    path = Path(path)
    if (not isinstance(expected, str) or not SHA.fullmatch(expected)
            or path.is_symlink() or not stat.S_ISREG(path.stat().st_mode)):
        raise ValueError("actual regular non-symlink file and full SHA required: " + str(path))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("actual file SHA differs: " + str(path))
    return raw


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: str
    sha256: str
    role: str


@dataclass(frozen=True, slots=True, init=False)
class BackendProfile:
    version: str
    binary_path: str
    binary_sha256: str
    source_tree_sha256: str
    patch_sha256: str
    source_file_count: int
    basis_file_count: int
    local_file_count: int
    inventory: tuple
    inventory_sha256: str
    assets_sha256: str
    sdf_sha256: str
    rod_schema: str
    reset_schema: str
    target_schema: str
    model_schema: str
    input_schema: str
    context_schema: str
    digest: str

    def __init__(self):
        raise TypeError("BackendProfile must be loaded from current actual source files")

    @classmethod
    def load(cls, project=ROOT):
        from . import tabletop_mesh_observation_source as source
        from .tabletop_mesh_reset_numeric_state import VERSION as reset_schema
        project = Path(project).resolve()
        if project != ROOT:
            raise ValueError("source-controlled current project root required")
        # Existing protected source inventory seals reader code as well as
        # compiled sources. Reading it does not hash historical checkpoints.
        protected_raw = checked_file(
            Path("/home/xk/.local/share/dh116-autonomous-3c80166/protected-ubuntu.json"),
            "fb6b58cfd2fe86408eb498a4390ecda35df2221d7198ca3a01ff59f86dea05ec")
        protected = json.loads(protected_raw)["protected_source_script_config_sha256"]
        review_raw = checked_file(Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-v3-independent-reaudit-portable-20261115-a639k8su/reaudit.json"),
            "fcf8f92c0b2838e72a6fe6d026cea61c54759b30350b607c0da6b26db6db6a15")
        protected.update(json.loads(review_raw)["replay_source_sha256"])
        protected.update({
            "src/go2_pvcnn/control/dh116/tabletop_bilateral_r0_approach.py": "e6f1325cf241976f370d3f3ee311596e998572f4b09e173233db5d7f831aa779",
            "src/go2_pvcnn/control/dh116/tabletop_r0_measured_approach.py": "bfb008346a68afe810976409c322bc8f28dcdee532bd784aece70c78d27440a6",
            "src/go2_pvcnn/control/dh116/tabletop_deformable_privileged_teacher.py": "f5ee314badf045c27b7efc44cc32e03e0b82f8199b35726f0a81e5e63722c872",
            "src/go2_pvcnn/control/dh116/tabletop_flexible_measured_common_teacher.py": "724650f30d550838c23e3b7c84c423df93fcbf5a746004f2708afec24e2b28e9",
            "src/go2_pvcnn/control/dh116/tabletop_teacher_mounted_wrist_source.py": "9aa4e96f484ba4cf8f8afa2ff0cdcddb8d4909d3fb9bc52416e616f299cd8d4e"})
        checked_file(Path(source.__file__), protected["src/go2_pvcnn/control/dh116/tabletop_mesh_observation_source.py"])
        basis = source.source_basis()
        if (basis["backend_path"] != str(source.BACKEND)
                or basis["backend_sha256"] != source.BACKEND_SHA
                or basis["source_tree_sha256"] != source.TREE_SHA
                or basis["patch_delta_sha256"] != source.PATCH_SHA):
            raise ValueError("independent observation backend identity differs")
        rows = {}
        def add(path, sha, role):
            path = Path(path).absolute()
            checked_file(path, sha)
            key = str(path)
            if key in rows and rows[key].sha256 != sha:
                raise ValueError("source inventory conflicting SHA")
            rows[key] = FileIdentity(key, sha, role)
        for path, sha in basis["sources"].items():
            add(path, sha, "backend_source")
        for path, sha in basis["local_sources"].items():
            add(project / path, sha, "local_source")
        # Candidate sources are explicit provenance, not new measured material.
        cfg = json.loads(checked_file(project / source.CONFIG, source.CONFIG_SHA))
        for path, sha in cfg["sources"].items():
            add(project / path, sha, "candidate_source")
        # Actual per-side assets are locked by the existing source-controlled
        # manifest reader. It reads bytes only; does not import USD/Isaac.
        from scripts.verify_dh116_cooked_convex_hulls_native import _sources
        asset_manifest = project / "assets/fixed_dual_panda_dh116_v1/manifest.json"
        add(asset_manifest, "9698b16e526087f62c5da32f5e78ef13d5b4467e081c942db89227ba73d34048", "asset_manifest")
        assets = _sources(project)
        for row in assets.values():
            add(row["path"], row["sha256"], "hand_asset")
        summary = json.loads(checked_file(SDF_DIR / "result.json", SDF_SUMMARY_SHA))
        add(SDF_DIR / "result.json", SDF_SUMMARY_SHA, "sdf_summary")
        sdf = {}
        for name, row in summary["pads"].items():
            path = SDF_DIR / (name + ".npz")
            add(path, row["sdf_file_sha256"], "pad_sdf")
            sdf[name] = dict(sha256=row["sdf_file_sha256"], capture_sha256=row["gpu_row_sha256"])
        if len(sdf) != 10:
            raise ValueError("complete independent left/right five-pad SDF inventory required")
        # Bind the readers/schema code too; no remembered profile snapshot.
        names = ("tabletop_mesh_observation_source.py", "tabletop_mesh_reset_numeric_state.py",
                 "tabletop_mesh_reset_readback.py", "tabletop_mesh_reset_order.py",
                 "tabletop_deformable_privileged_teacher.py", "tabletop_flexible_measured_common_teacher.py",
                 "tabletop_teacher_mounted_wrist_source.py", "tabletop_bilateral_r0_approach.py",
                 "tabletop_r0_measured_approach.py", "tabletop_r0_health_separation.py")
        for name in names:
            path = Path(__file__).parent / name
            relative = str(path.relative_to(project))
            add(path, protected[relative], "reader_schema")
        # capture_reset imports check_radius_packet from this frozen replay
        # reader. Its independent backend alias is part of the new bridge too.
        radius_reader = "scripts/replay_dh116_native_parameter_r0.py"
        add(project / radius_reader, protected[radius_reader], "reader_schema")
        values = dict(version=VERSION, binary_path=str(source.BACKEND),
            binary_sha256=source.BACKEND_SHA, source_tree_sha256=source.TREE_SHA,
            patch_sha256=source.PATCH_SHA, source_file_count=basis["source_file_count"],
            basis_file_count=len(basis["sources"]), local_file_count=len(basis["local_sources"]),
            inventory=tuple(sorted(rows.values(), key=lambda x: x.path)),
            assets_sha256=digest(assets), sdf_sha256=digest(sdf), rod_schema=ROD_SCHEMA_VERSION,
            reset_schema=reset_schema, target_schema=TARGET_SCHEMA_VERSION,
            model_schema=MODEL_SCHEMA_VERSION, input_schema=INPUT_SCHEMA_VERSION,
            context_schema=CONTEXT_SCHEMA_VERSION)
        identity = {k: ([dict(path=r.path, sha256=r.sha256, role=r.role) for r in v]
                        if k == "inventory" else v) for k, v in values.items()}
        values["inventory_sha256"] = digest(identity["inventory"])
        identity["inventory_sha256"] = values["inventory_sha256"]
        values["digest"] = digest(identity)
        result = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(result, key, value)
        return result

    @classmethod
    def from_sources(cls, *, binary_sha256, source_inventory, project=ROOT):
        from . import tabletop_mesh_observation_source as source
        if binary_sha256 != source.BACKEND_SHA:
            raise ValueError("unknown or historical backend cannot be relabelled")
        result = cls.load(project)
        if source_inventory != {row.path: row.sha256 for row in result.inventory}:
            raise ValueError("complete actual source inventory differs")
        return result

    def verify(self, project=ROOT):
        for row in self.inventory:
            checked_file(row.path, row.sha256)
        current = type(self).load(project)
        if current != self:
            raise ValueError("current BackendProfile source identity differs")
        return self


_profile_reader_depth = 0
_profile_reader_thread = None


@contextmanager
def profile_reader_scope(backend_profile):
    """Bind the complete legacy Reset-reader bridge to a verified NEW profile.

    The old radius replay default remains historical outside this single-owner
    scope. Each entry reopens actual source bytes; caller-provided SHAs cannot
    authorize a binding. Same-thread nesting and existing source scopes compose.
    """
    global _profile_reader_depth, _profile_reader_thread
    if type(backend_profile) is not BackendProfile:
        raise ValueError("actual source-loaded BackendProfile required")
    backend_profile.verify()
    from . import tabletop_mesh_observation_source as source
    from scripts import replay_dh116_native_parameter_r0 as radius
    owner = threading.get_ident()
    if _profile_reader_depth and _profile_reader_thread != owner:
        raise ValueError("profile reader scope belongs to another thread")
    expected = backend_profile.binary_sha256 if _profile_reader_depth else source.OLD_BACKEND_SHA
    if radius.BACKEND_SHA != expected:
        raise ValueError("unexpected profile radius reader binding")
    prior = radius.BACKEND_SHA
    with source.source_scope():
        if not _profile_reader_depth:
            radius.BACKEND_SHA = backend_profile.binary_sha256
            _profile_reader_thread = owner
        _profile_reader_depth += 1
        try:
            yield backend_profile
        finally:
            _profile_reader_depth -= 1
            if not _profile_reader_depth:
                radius.BACKEND_SHA = prior
                _profile_reader_thread = None
