"""Explicit unqualified profile for the independently built flat CUDA candidate."""
from dataclasses import dataclass
from pathlib import Path
import json
from .tabletop_native_teacher_profile import checked_file,digest,ROOT

LOCK=ROOT/"docs/contracts/dh116_flat_backend_candidate_v1.json"

@dataclass(frozen=True,slots=True)
class TensorBackendProfile:
    binary_path:str
    binary_sha256:str
    source_tree_sha256:str
    patch_sha256:str
    digest:str
    manifest_path:str

    @classmethod
    def load(cls,path=LOCK):
        path=Path(path)
        lock=json.loads(path.read_text())
        if (lock["schema"]!="dh116-flat-backend-candidate-v1"
                or lock["teacher_admission"] is not False or lock["native_mechanical_receipt"] is not None
                or lock["upstream_commit"]!="96b8514c7c55c724f1abba1fbf6ed6642b7db72b"
                or lock["source_file_count"]!=72 or lock["physics_kernel_changes"] is not False
                or lock["cuda_architectures"]!=[89] or lock["flat_abi"]!="contact_pair_samples_flat_cuda_v1"):
            raise ValueError("explicit separately source-bound unqualified flat backend required")
        for p,h in lock["source_inventory"].items():checked_file(p,h)
        checked_file(lock["binary_path"],lock["binary_sha256"])
        proof=lock["parity_evidence"]
        parity=json.loads(checked_file(proof["path"],proof["sha256"]))
        if (parity["binary_sha256"]!=lock["binary_sha256"] or parity["raw_indices_vectors_exact"] is not True
                or parity["reductions_max_abs_error"]>1e-12
                or not all(parity[k] for k in ("all_empty_batch","overflow_rejected","owned_outputs_stable",
                    "read_once","reset_rejected","tensor_owner_reset_continuation","gpu_all_mesh_submission"))
                or parity["isaac_steps"]!=0 or parity["ppo_updates"]!=0):
            raise ValueError("actual flat native parity proof required; not Isaac qualification")
        patch_path=str(ROOT/"scripts/prepare_dh116_tensor_pair_backend.py")
        return cls(lock["binary_path"],lock["binary_sha256"],digest(lock["source_inventory"]),
                   lock["source_inventory"][patch_path],digest(lock),str(path.resolve()))

    def verify(self):
        if self!=type(self).load(self.manifest_path):
            raise ValueError("flat backend candidate source identity changed")
        return self
