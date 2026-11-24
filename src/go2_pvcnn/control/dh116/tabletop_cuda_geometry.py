"""Sealed opt-in CUDA measurements; all capture decisions remain inherited.

No torch.compile is used by this path. The PPO optimizer may independently
import Dynamo; this module makes no claim about imports elsewhere in training.
The private candidate's proofs cover replay measurements, not native physics.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .tabletop_measured_closure_geometry import NUMERIC_GUARD
from .tabletop_tensor_geometry import TensorGeometry

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = ROOT / "docs/contracts/dh116_cuda_geometry_candidate_v1.json"
MODE = "cuda_cabi"
CANDIDATE_VERSION = "dh116-full-vertex-fp64-cuda-measurement-v6-full-cabi-pad-nan-routing-guard"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked_file(path, expected, label):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(label + " SHA256 required")
    path = Path(path).resolve(strict=True)
    if _sha(path) != expected:
        raise ValueError(label + " SHA256 differs: " + str(path))
    return path


@dataclass(frozen=True, slots=True)
class CudaGeometryProfile:
    """A verified manifest identity; never a native admission receipt."""
    manifest_path: Path
    manifest_sha256: str
    binary_path: Path
    binary_sha256: str
    source_sha256: tuple
    native_ready: bool = False
    teacher_admission: bool = False

    @classmethod
    def load(cls, path=DEFAULT_MANIFEST):
        path = Path(path).resolve(strict=True)
        manifest_hash = _sha(path)
        m = json.loads(path.read_text())
        required = dict(schema="dh116-cuda-geometry-candidate-v1", mode=MODE,
            version=CANDIDATE_VERSION, sm=89, precision="FP64", full_vertices=True,
            support_normals=64, fast_math=False, fmad=False, native_ready=False,
            teacher_admission=False, runtime_graph_compilation=False)
        if any(type(m.get(k)) is not type(v) or m.get(k) != v for k, v in required.items()):
            raise ValueError("invalid CUDA geometry manifest version, precision, SM89 or admission flags")
        sources = m.get("repo_source_sha256", {})
        math_names = ("hull_support.py", "pad_support.py", "hull_support.cu", "pad_support.cu")
        prefix = "src/go2_pvcnn/control/dh116/"
        expected_sources = {prefix+"tabletop_cuda_geometry.py", prefix+"tabletop_tensor_geometry.py",
            prefix+"tabletop_runtime_source_audit.py", prefix+"cuda_measurements/__init__.py",
            "scripts/build_dh116_cuda_geometry_candidate.py"}
        expected_sources.update(prefix+"cuda_measurements/"+n for n in math_names)
        if set(sources) != expected_sources:
            raise ValueError("complete CUDA geometry repo source SHA inventory required")
        for relative, digest in sources.items():
            _checked_file(ROOT / relative, digest, "repo source")
        proof = m["private_candidate"]
        contract_path = _checked_file(proof["contract_path"], proof["contract_sha256"], "private contract")
        original = json.loads(contract_path.read_text())
        if (original["version"] != CANDIDATE_VERSION or original["sm"] != 89
                or original["nvcc_version"] != m["nvcc_version"]
                or original["binary_sha256"] != m["binary_sha256"]
                or original["source_sha256"] != proof["source_sha256"]
                or original["evidence_sha256"] != proof["evidence_sha256"]
                or original["oracle_source_sha256"] != proof["oracle_source_sha256"]):
            raise ValueError("private candidate proof identity differs")
        for name, digest in proof["source_sha256"].items():
            _checked_file(contract_path.parent / name, digest, "private source")
        for name, digest in proof["evidence_sha256"].items():
            _checked_file(contract_path.parent / name, digest, "private evidence")
        for name in math_names:
            if sources[prefix+"cuda_measurements/"+name] != proof["source_sha256"][name]:
                raise ValueError("repo math source SHA differs from validated private candidate")
        # The private oracle is historical: production removed its rejected
        # graph-compilation branch. The current full source is sealed above;
        # unchanged capture/eager/fallback methods are independently verified.
        preservation = m["source_preservation_evidence"]
        guard_proof = json.loads(_checked_file(preservation["path"], preservation["sha256"],
            "guard preservation").read_text())
        archive = m["rejected_compilation_archive"]
        archived = json.loads(_checked_file(archive["path"], archive["sha256"],
            "rejected compilation archive").read_text())
        if (guard_proof.get("before_source_sha256") != proof["oracle_source_sha256"]
                or archived.get("original_guard_source_sha256") != proof["oracle_source_sha256"]
                or guard_proof.get("after_source_sha256") != sources[prefix+"tabletop_tensor_geometry.py"]):
            raise ValueError("inherited guard preservation identity differs")
        binary = _checked_file(m["binary_path"], m["binary_sha256"], "CUDA binary")
        if _sha(path) != manifest_hash:
            raise ValueError("CUDA geometry manifest changed during verification")
        return cls(path, manifest_hash, binary, m["binary_sha256"], tuple(sorted(sources.items())))

    def verify(self):
        if type(self.native_ready) is not bool or self.native_ready or type(self.teacher_admission) is not bool or self.teacher_admission:
            raise ValueError("CUDA measurements have no native admission")
        current = type(self).load(self.manifest_path)
        if current != self:
            raise ValueError("CUDA geometry profile identity changed")
        return self

    def identity(self):
        return dict(actual_measurement_backend=MODE, binary_sha256=self.binary_sha256,
            manifest_sha256=self.manifest_sha256, source_sha256=dict(self.source_sha256),
            binary_path=str(self.binary_path), native_ready=False, teacher_admission=False)


class CudaMeasurementGeometry(TensorGeometry):
    """Only hull/pad measurements change; capture and every guard stay eager."""
    measurement_backend = MODE

    def __init__(self, shapes, pads=(), *, profile=None, **kwargs):
        backend = kwargs.pop("backend", None)
        if backend not in (None, "eager"):
            raise ValueError("CUDA measurements require eager base guards")
        if torch.device(kwargs["device"]).type != "cuda":
            raise ValueError("CUDA measurements require a CUDA device")
        profile = CudaGeometryProfile.load() if profile is None else profile.verify()
        if torch.cuda.get_device_capability(kwargs["device"]) != (8, 9):
            raise ValueError("validated CUDA measurements require SM89")
        super().__init__(shapes, pads, backend="eager", **kwargs)
        from .cuda_measurements.hull_support import HullSupport
        from .cuda_measurements.pad_support import PadSupport
        self.cuda_profile = profile
        self.cuda_hulls = HullSupport([(s, l, k, v) for s, l, k, v, _ in self._shapes],
            self._normals, device=self.device, library=profile.binary_path)
        self.cuda_pads = (PadSupport(self._pads, self._pad_roundoff, device=self.device,
            library=profile.binary_path) if self._pads else None)

    def _metrics(self, p, nodes, radii, rotations):
        low, high, projected = self.cuda_hulls.extrema(rotations.contiguous(), p[..., :3].contiguous())
        rod_high = (nodes @ self._normals.T).amax(1)
        lower = (projected-rod_high[:, None]).amax(-1).clamp_min(0)-radii[:, None]-NUMERIC_GUARD
        return low, high, lower

    def _pad_metrics(self, p, nodes, radii, rotations, thresholds):
        if self.cuda_pads is None:
            return self._pad_metrics_eager(p, nodes, radii, rotations, thresholds)
        # Original capture accepts strided tensors; C ABI requires contiguous
        # buffers. These same-device copies preserve every input value.
        return self.cuda_pads.metrics(rotations.contiguous(), p[..., :3].contiguous(),
            nodes.contiguous(), radii.contiguous(), thresholds)


def make_tensor_geometry(owner, body_names, *, device, backend=None):
    """Shared measurement factory for source owners, independent of any model."""
    mode = os.environ.get("DH116_TENSOR_GEOMETRY_BACKEND", "eager") if backend is None else backend
    if mode == MODE:
        return CudaMeasurementGeometry.from_policy_geometry(owner, body_names, device=device, backend="eager")
    if mode == "compiled":
        raise ValueError("compiled tensor geometry backend is disabled; use eager or cuda_cabi")
    if mode != "eager":
        raise ValueError("tensor geometry backend must be eager or cuda_cabi")
    return TensorGeometry.from_policy_geometry(owner, body_names, device=device, backend=mode)


def geometry_measurement_identity(geometry):
    """Report the real measurement backend; Power has no SDF geometry here."""
    if isinstance(geometry, CudaMeasurementGeometry):
        return geometry.cuda_profile.identity()
    return dict(actual_measurement_backend=getattr(geometry, "backend", None),
        binary_sha256=None, manifest_sha256=None, source_sha256={}, native_ready=False, teacher_admission=False)


def verify_measurement_native_receipts(receipts, *, style):
    """Authenticate N2 contact/Reset and N128 timing before bounded PPO work.

    Each item is (outer.json path, externally supplied SHA256). Only fixed
    native result/AppClose/source-freeze/Timeline artifacts, native.log and
    parent attempt.json siblings are read. Evidence authorizes this scope and never
    changes teacher admission, physical qualification, or profile readiness.
    """
    if style != "pinch":
        raise ValueError("CUDA geometry receipts currently cover Pinch only")
    if len(receipts) != 2:
        raise ValueError("distinct N2 contact and N128 throughput receipts required")
    identity = CudaGeometryProfile.load().verify().identity()
    from .tabletop_runtime_source_audit import SOURCE_NAMES, VERSION as SOURCE_AUDIT_VERSION
    checked, counts, paths = [], set(), set()
    readback_mode = os.environ.get("DH116_NATIVE_READBACK_BACKEND", "raw")
    source_cache_metadata = None
    for path, expected_sha in receipts:
        path = _checked_file(path, expected_sha, "measurement native receipt")
        if path in paths:
            raise ValueError("distinct N2 contact and N128 throughput receipts required")
        paths.add(path)
        outer = json.loads(path.read_text())
        status = outer.get("inferior_status", {})
        if (type(outer.get("gdb_exit_code")) is not int or outer["gdb_exit_code"] != 0
                or type(status.get("exit_code")) is not int or status["exit_code"] != 0
                or "signal" not in status or status["signal"] is not None
                or outer.get("forced_exit") is not False):
            raise ValueError("actual natural inferior exit zero without signal or force required")
        native = path.parent / "native"
        if native.is_symlink() or not native.is_dir():
            raise ValueError("fixed sibling native artifact directory required")
        _checked_file(path.parent/"native.log", outer.get("log_sha256"), "native log")
        artifacts = outer.get("artifacts", {})
        loaded = {}
        for name in ("result.json", "app_close_returned.json", "source_freeze_audit.json", "timeline_release.json"):
            artifact = native / name
            if artifact.is_symlink():
                raise ValueError("fixed native artifacts cannot be symlinks")
            loaded[name] = json.loads(_checked_file(artifact, artifacts.get(name), "native artifact").read_text())
        attempt_path = path.parent / "attempt.json"
        if attempt_path.is_symlink():
            raise ValueError("fixed parent attempt cannot be a symlink")
        attempt = json.loads(_checked_file(attempt_path, artifacts.get("attempt.json"), "native attempt").read_text())
        freeze = loaded["source_freeze_audit.json"]
        if (freeze.get("version") != SOURCE_AUDIT_VERSION or freeze.get("admissible") is not True
                or freeze.get("startup_checked") is not True
                or type(freeze.get("drift")) is not list or freeze["drift"]
                or freeze.get("attempt_sha256") != artifacts["attempt.json"]
                or freeze.get("process_exit_verified") is not False or freeze.get("qualification_granted") is not False):
            raise ValueError("admissible startup/pre-close source freeze without drift required")
        frozen_sources, audited_sources = attempt.get("source_sha256"), freeze.get("source_sha256")
        if not isinstance(frozen_sources, dict) or not isinstance(audited_sources, dict):
            raise ValueError("complete shared runtime source freeze inventory required")
        for name in SOURCE_NAMES:
            expected = frozen_sources.get(name)
            if expected is None or audited_sources.get(name) != expected or (ROOT/name).is_symlink():
                raise ValueError("shared runtime source freeze differs: " + name)
            _checked_file(ROOT/name, expected, "current shared runtime source")
        timeline = loaded["timeline_release.json"]
        timeline_required = dict(version="dh116-local-timeline-release-v1", stop_commit_returned=True,
            local_timeline_weakref_supported=True, local_timeline_weakref_alive=False,
            local_timeline_release_verified=True, native_release_verified=False,
            process_exit_verified=False, qualification_granted=False)
        if any(type(timeline.get(k)) is not type(v) or timeline.get(k) != v for k, v in timeline_required.items()):
            raise ValueError("verified local Timeline release without native/exit qualification required")
        result, close = loaded["result.json"], loaded["app_close_returned.json"]
        a = result.get("acquisition", {})
        metadata = a.get("source_cache_metadata")
        if (a.get("native_readback_mode") != readback_mode
                or (readback_mode != "raw" and not metadata)
                or (checked and metadata != source_cache_metadata)):
            raise ValueError("native readback mode or source cache metadata differs")
        source_cache_metadata = metadata
        if (result.get("failure", "missing") is not None or result.get("close_mode") != "normal"
                or result.get("lifetime_released_before_close") is not True
                or close.get("app_close_returned") is not True or close.get("close_mode") != "normal"
                or a.get("style") != style or a.get("measurement_scope") != "bounded_diagnostic"
                or a.get("actual_measurement_backend") != MODE or a.get("geometry_measurement") != identity
                or type(a.get("updates")) is not int or a["updates"] != 0
                or a.get("teacher_admission") is not False or a.get("grasp_qualified") is not False
                or result.get("teacher_admission") is not False):
            raise ValueError("matching style/profile and bounded diagnostic with natural AppClose required")
        n = a.get("environments")
        if type(n) is not int or n not in (2, 128) or n in counts:
            raise ValueError("distinct N2 contact and N128 throughput receipts required")
        counts.add(n)
        expected_steps, expected_windows = (218, 27) if n == 2 else (18, 2)
        times = a.get("window_seconds", [])
        if (a.get("sdk_step") != expected_steps or result.get("physics_steps") != expected_steps
                or a.get("controls") != expected_windows or len(times) != expected_windows
                or not all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in times)):
            raise ValueError("complete actual bounded SDK windows and timings required")
        if n == 128:
            if a.get("throughput_probe_only") is not True or a.get("reset_ids") != [1]*128:
                raise ValueError("N128 bounded throughput receipt required")
        else:
            if readback_mode == "official_cache":
                audit = a.get("readback_audit", {})
                if (audit.get("enabled") is not True or audit.get("diagnostic_only") is not True
                        or type(audit.get("side_snapshots")) is not int or audit["side_snapshots"] <= 0
                        or type(audit.get("mismatches")) is not list or audit["mismatches"]
                        or audit.get("extra_raw_reads_per_side_snapshot") != 2):
                    raise ValueError("N2 official cache requires a successful dual raw/cache oracle audit")
            coverage = a.get("contact_coverage", {})
            for key in ("hand_pair_samples", "nonzero_captured_hand_steps", "nonzero_applied_hand_steps"):
                values = coverage.get(key)
                if (not isinstance(values, list) or len(values) != 2
                        or any(type(v) is not int or v < 0 for v in values) or values[0] <= 0 or values[1] != 0):
                    raise ValueError("real active-lane hand contact and next-SDK reaction required")
            checks = a.get("reset_checks", [])
            continuation = a.get("post_reset_next_sdk") or {}
            if (a.get("reset_ids") != [2, 1] or a.get("ppo_rollout_rows") != 0
                    or a.get("ppo_rollout_admission") is not False or len(checks) != 1
                    or checks[0].get("selected") != [0] or checks[0].get("peers") != [1]
                    or checks[0].get("native_mesh_verified") is not True
                    or checks[0].get("sdk_step") != 210 or checks[0].get("reset_ids") != [2, 1]
                    or continuation.get("sdk_step") != 211 or continuation.get("source_sdk_step") != 210
                    or continuation.get("reset_ids") != [2, 1] or continuation.get("rod_ticks_pre") != [0, 2496]
                    or continuation.get("applied", {}).get("nonzero_hand_force", [None])[0] is not False):
                raise ValueError("isolated Reset and verified next-SDK zero-reaction continuation required")
        checked.append(dict(path=str(path), sha256=expected_sha, environments=n,
            result_sha256=artifacts["result.json"], app_close_sha256=artifacts["app_close_returned.json"],
            attempt_sha256=artifacts["attempt.json"], source_freeze_audit_sha256=artifacts["source_freeze_audit.json"],
            timeline_release_sha256=artifacts["timeline_release.json"]))
    if counts != {2, 128}:
        raise ValueError("distinct N2 contact and N128 throughput receipts required")
    return dict(scope="bounded_diagnostic", style=style, receipts=checked,
        geometry_measurement=identity, native_readback_mode=readback_mode,
        source_cache_metadata=source_cache_metadata, native_ready=False, teacher_admission=False)
