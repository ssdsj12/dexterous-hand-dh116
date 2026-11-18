"""Exact row replay from records sealed by the actual native process observer.

This verifier is for completed disk sources, not active online DAgger sessions.
It does not grant Teacher success, force-source or native-task qualification.
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import torch

from . import tabletop_stereo_promotion as promotion


def _same(a, b):
    """Compare tensor storage bytes, including NaN payloads, and typed trees."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        left = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        right = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return torch.equal(left, right)
    if isinstance(a, dict) or isinstance(b, dict):
        return (type(a) is type(b) and a.keys() == b.keys()
                and all(_same(a[key], b[key]) for key in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    return type(a) is type(b) and a == b


def _validate_bindings(bindings):
    """Validate only reference syntax; references never create qualification."""
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("formal data requires trusted source bindings")
    for binding in bindings:
        if not isinstance(binding, dict) or binding.keys() != {
            "proof_path", "proof_sha256", "row_indices",
        }:
            raise ValueError("exact trusted source bindings required")
        if (not isinstance(binding["proof_path"], str)
                or not Path(binding["proof_path"]).is_absolute()):
            raise ValueError("absolute native source proof required")
        promotion.require_sha(binding["proof_sha256"])
        indices = binding["row_indices"]
        if (not isinstance(indices, list) or not indices
                or any(type(index) is not int or index < 0 for index in indices)
                or indices != sorted(set(indices))):
            raise ValueError("ordered unique raw record row selection required")
    return bindings


def verify_stereo_dataset_sources(data):
    """Replay exact formal rows from outputs owned by one actual completion."""
    if not isinstance(data, dict):
        raise ValueError("versioned stereo dataset required")
    if data.get("schema_only") is True:
        return
    bindings = _validate_bindings(data.get("source_bindings"))
    from .tabletop_stereo_supervision_dataset import rows_from_records

    expected_rows = []
    for binding in bindings:
        try:
            proof_path = Path(binding["proof_path"])
            if not proof_path.is_file():
                raise ValueError("regular native source proof required")
            proof_bytes = proof_path.read_bytes()
            if hashlib.sha256(proof_bytes).hexdigest() != binding["proof_sha256"]:
                raise ValueError("native source proof SHA mismatch")
            proof = json.loads(proof_bytes)
            if not isinstance(proof, dict):
                raise ValueError("native source proof object required")
        except (OSError, TypeError, ValueError) as error:
            raise ValueError("native source proof unavailable or changed") from error

        base = proof_path.parent
        promotion._native_run(proof, base)
        # Require raw, manifest and records to be outputs of this same observer.
        promotion._process_output(proof, base, proof.get("raw_artifact"))
        manifest_bytes, manifest_path = promotion._process_output(
            proof, base, proof.get("stereo_source_manifest_artifact"))
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeError, ValueError) as error:
            raise ValueError("native stereo source manifest identity differs") from error
        if (
            not isinstance(manifest, dict)
            or manifest.keys() != {
                "version", "teacher_checkpoint_sha256",
                "native_raw_artifact", "records_artifact",
            }
            or manifest["version"] != "dh116-stereo-source-manifest-v1"
            or manifest["teacher_checkpoint_sha256"] != data.get("source_sha256")
            or manifest["native_raw_artifact"] != proof.get("raw_artifact")
        ):
            raise ValueError("native stereo source manifest identity differs")
        records_bytes, _ = promotion._process_output(
            proof, base, manifest["records_artifact"], binding_base=manifest_path.parent)
        packet = torch.load(io.BytesIO(records_bytes), map_location="cpu", weights_only=True)
        if (
            not isinstance(packet, dict)
            or packet.get("version") != "dh116-stereo-teacher-candidate-records-v1"
            or packet.get("teacher_checkpoint_sha256") != data.get("source_sha256")
            or not isinstance(packet.get("records"), list)
        ):
            raise ValueError("actual sealed stereo records required")
        records = deepcopy(packet["records"])
        if data.get("purpose") == "successful_bc":
            for record in records:
                record["mode"] = "teacher_reference"
                record["teacher_qualification"] = deepcopy(data.get("qualification"))
                record["candidate_only"] = False
        elif data.get("purpose") != "dagger_visited_states":
            raise ValueError("explicit formal source purpose required")
        rows = rows_from_records(records)
        indices = binding["row_indices"]
        if any(index >= len(rows) for index in indices):
            raise ValueError("ordered unique raw record row selection required")
        expected_rows.extend(rows[index] for index in indices)
    if not _same(data.get("rows"), expected_rows):
        raise ValueError("dataset rows differ from sealed native records")
