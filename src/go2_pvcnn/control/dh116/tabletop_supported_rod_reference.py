"""Read a sealed parked-hand rod observation for geometry only, not Reset."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .tabletop_deformable_r0_replay import decode_owned

IDENTITY = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")
SUPPORTS = ("support_0", "support_1")


@dataclass(frozen=True)
class SupportedRodReference:
    physics_step: int
    rod_tick: int
    node_positions: np.ndarray
    node_velocities: np.ndarray
    rest_lengths_m: np.ndarray
    support_boxes: tuple
    receipt_sha256: str
    sample_sha256: str


def _read_artifact(output, hashes, name):
    path = (output/name).resolve(strict=True)
    if path.parent != output or not path.is_file():
        raise ValueError(f"invalid recorded artifact path: {name}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != hashes.get(name):
        raise ValueError(f"changed recorded artifact: {name}")
    return json.loads(data), digest


def _readonly(array, shape):
    if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError("finite original floating-point material arrays required")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)


def load_parked_reference(receipt, expected_sha256, sdk_step, *, task="power"):
    """Bind native clock, both support contacts and all material geometry at this tick."""
    if task not in ("power", "tripod", "pinch"):
        raise ValueError("explicit original supported task identity required")
    receipt = Path(receipt).resolve(strict=True)
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("parked native receipt SHA-256 differs")
    if type(sdk_step) is not int or sdk_step < 3:
        raise ValueError("an explicit measured SDK step is required")
    outer = json.loads(raw)
    if (outer.get("exit_code") != 0 or outer.get("debugger_inferior_exit_code") != 0
            or outer.get("debugger_inferior_signal") is not None
            or outer.get("debugger_status_verified") is not True
            or outer.get("forced_exit") is not False or outer.get("exceeded_budget") is not False):
        raise ValueError("parked reference requires verified natural exit within budget")
    output = (receipt.parent/"native").resolve(strict=True)
    if output.parent != receipt.parent:
        raise ValueError("recorded native output directory is outside the receipt scope")
    manifest = outer["artifact_sha256"]
    scene, _ = _read_artifact(output, manifest, f"{task}_support_scene.json")
    result, _ = _read_artifact(output, manifest, f"{task}_shared_result.json")
    lifecycle, _ = _read_artifact(output, manifest, "result.json")
    end = result["sdk_steps"]
    if (result["scope"] != f"measured_{task}_support_settling_without_hand"
            or type(end) is not int or end < sdk_step or result["measured_exchanges"] != end-2
            or result["solver_substeps"] != (end-2)*12 or lifecycle["physics_steps"] != end
            or lifecycle["failure"] is not None or lifecycle["lifetime_released_before_close"] is not True):
        raise ValueError("complete isolated support-settling acquisition required")
    sample, sample_sha = _read_artifact(output, manifest, f"{task}_exchange_{sdk_step}.json")
    rod = decode_owned(sample)["rod"]
    pairs = rod["rod_mesh_contact_pairs"]
    tick = (sdk_step-2)*12
    if (any(rod[key] != scene["reset_clock"][key] for key in IDENTITY)
            or rod["clock_valid"] is not True or rod["pair_capture_status"] != "validated_solver_samples"
            or (rod["physics_step"], rod["rod_tick"]) != (sdk_step,tick)
            or (pairs["physics_step"],pairs["rod_tick"]) != (sdk_step,tick)
            or not math.isclose(rod["timestamp_s"],sdk_step/480,rel_tol=0,abs_tol=1e-7)):
        raise ValueError("parked observation identity or clock differs from its measured SDK boundary")
    loaded = set()
    for pair in pairs["pairs"]:
        substep, name = pair["substep"], pair["mesh_name"]
        if type(substep) is not int or not 0 <= substep < 12 or name not in SUPPORTS:
            raise ValueError("isolated support pairs must exclude the table and both hands")
        normal = pair["normal_force_on_rod_n"]
        if not math.isfinite(normal) or normal < 0.:
            raise ValueError("invalid normal load in isolated support pair")
        if normal > 0.:
            loaded.add((substep,name))
    if loaded != {(substep,name) for substep in range(12) for name in SUPPORTS}:
        raise ValueError("isolated support pairs must carry both supports in every solver substep")
    boxes = tuple(scene["supports"])
    if (tuple(box["name"] for box in boxes) != ("negative_y","positive_y")
            or any(len(box["center_m"]) != 3 or len(box["size_m"]) != 3
                   or any(not math.isfinite(n) for n in (*box["center_m"], *box["size_m"]))
                   or min(box["size_m"]) <= 0 for box in boxes)):
        raise ValueError("two original finite support boxes required")
    positions = _readonly(rod["node_positions"], (1,41,3))[0]
    velocities = _readonly(rod["velocities"], (1,41,3))[0]
    rest = _readonly(rod["rest_lengths_m"], (40,))
    if np.any(rest <= 0):
        raise ValueError("original positive rest lengths required")
    return SupportedRodReference(sdk_step,tick,positions,velocities,rest,boxes,expected_sha256,sample_sha)
