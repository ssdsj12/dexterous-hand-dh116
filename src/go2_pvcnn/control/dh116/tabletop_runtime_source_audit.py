"""Bind shared runtime source bytes at startup and before App.close.

This is a two-boundary CPU file audit; it never runs per physics substep,
changes a model/guard, or certifies native exit.
"""
import hashlib
import json
from pathlib import Path

VERSION = "dh116-attempt-source-freeze-audit-v1"
SOURCE_NAMES = ('src/go2_pvcnn/control/dh116/tabletop_runtime_source_audit.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_physics_runtime.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_collector.py', 'src/go2_pvcnn/control/dh116/tabletop_runtime_execution.py', 'src/go2_pvcnn/control/dh116/tabletop_runtime_lifecycle.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_control.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_wrist_projection.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_mesh_bridge.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_pair_capture.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_rod_owner.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_source_owner.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_training_reward.py', 'src/go2_pvcnn/control/dh116/tabletop_tensor_geometry.py', 'src/go2_pvcnn/control/dh116/tabletop_cuda_geometry.py', 'src/go2_pvcnn/control/dh116/tabletop_regex_articulation.py', 'src/go2_pvcnn/control/dh116/tabletop_regex_cached_readback.py', 'src/go2_pvcnn/control/dh116/tabletop_regex_task_setup.py', 'src/go2_pvcnn/control/dh116/tabletop_training_reward.py', 'scripts/audit_dh116_isaac_batch_scene.py', 'scripts/dh116_fault_safe_lifecycle_v3.py', 'scripts/smoke_dh116_tensor_contact_replay.py', 'scripts/train_dh116_tensor_native_teacher.py')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceFreezeAudit:
    def __init__(self, *, project, attempt, output):
        self.project, self.attempt, self.output = Path(project), Path(attempt), Path(output)
        if self.attempt.is_symlink():
            raise ValueError("source attempt must be an owned regular file")
        self.attempt_sha256 = _sha(self.attempt)
        inventory = json.loads(self.attempt.read_text()).get("source_sha256")
        if not isinstance(inventory, dict) or any(name not in inventory for name in SOURCE_NAMES):
            raise ValueError("complete shared runtime source inventory required")
        self.expected = {name: inventory[name] for name in SOURCE_NAMES}
        self._started = self._finished = False

    def _inspect(self):
        drift, actual = [], {}
        for name, expected in self.expected.items():
            path = self.project / name
            value = _sha(path) if path.is_file() and not path.is_symlink() else None
            actual[name] = value
            if value != expected:
                drift.append(dict(path=name, before=expected, after=value))
        value = _sha(self.attempt)
        if value != self.attempt_sha256:
            drift.append(dict(path="attempt.json", before=self.attempt_sha256, after=value))
        return actual, drift

    def _write(self, actual, drift, *, admissible):
        from scripts.diagnose_dh116_teacher_initialization import write_new
        write_new(self.output / "source_freeze_audit.json", dict(
            version=VERSION, attempt_sha256=self.attempt_sha256, source_sha256=actual,
            drift=drift, admissible=admissible, startup_checked=self._started,
            process_exit_verified=False, qualification_granted=False))

    def check_startup(self):
        if self._started:
            raise ValueError("source freeze startup audit is once only")
        actual, drift = self._inspect()
        if drift:
            self._write(actual, drift, admissible=False)
            raise ValueError("source freeze changed before native acquisition")
        self._started = True

    def before_close(self):
        if self._finished:
            raise ValueError("source freeze pre-close audit is once only")
        self._finished = True
        actual, drift = self._inspect()
        admissible = self._started and not drift
        self._write(actual, drift, admissible=admissible)
        if not admissible:
            raise ValueError("source freeze changed or startup audit missing")
        return True
