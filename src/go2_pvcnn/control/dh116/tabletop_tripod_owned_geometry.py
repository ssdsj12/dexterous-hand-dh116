"""Independent measured left Tripod scope; preserves original collision guards.

Only the frozen moving-wrist target and left middle-pad classification change.
This is an owned, read-only diagnostic: no SDK stepping or native qualification.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from scripts import prepare_dh116_tripod_owned_approach as plan
from scripts.prepare_dh116_pinch_margin_contact import required_sdf_valid
from .tabletop_bilateral_owned_geometry import _equal, separation_lower
from .tabletop_segment_closure_geometry import SegmentClosureGeometry
from .tabletop_measured_closure_geometry import _bytes
from .tabletop_deformable_r0_replay import encode_owned, decode_owned
from .tabletop_deformable_wrist_projection import SOURCE

VERSION = "dh116-left-tripod-owned-geometry-v1"

def validate_projection(pair, exchange, *, schedule=None):
    action = exchange["action_readback"]
    ids = action["post"]["identity"]
    if len(ids) != 1:
        raise ValueError("one same-clock Tripod physical identity required")
    clock = ids[0]
    step = clock["physics_step"]
    if type(step) is not int:
        raise ValueError("integer SDK step required")
    ordinal = step - plan.SETUP_STEPS
    if (not 1 <= ordinal <= plan.MEASURED_STEPS
            or clock["policy_step"] != (ordinal - 1) // 8
            or clock["substep"] != (ordinal - 1) % 8):
        raise ValueError("same frozen policy/physics substep required")
    expected = plan.expected_boundary(pair, ordinal)
    _equal(action["request"]["action24"],
           (plan.action_schedule(pair).numpy() if schedule is None else schedule)[(ordinal - 1) // 8][None],
           "24D action", exact=True)
    projection = exchange["wrist_projection"]
    submission = action["submission"]
    for value, label in (
            (projection["palm_offsets_m"], "projected offsets"),
            (action["request"]["requested_palm_offsets_m"], "requested offsets")):
        _equal(value, expected["palm_offsets_m"][None], label)
    for value, label in (
            (projection["projected_angles"], "projected angles"),
            (action["request"]["requested_angles_rad"], "requested angles"),
            (submission["projected_angles"], "submitted angles")):
        _equal(value, expected["intrinsic_angles_rad"][None], label)
    for value, label in (
            (projection["projected_root_poses"], "projected roots"),
            (submission["projected_root_poses"], "submitted roots")):
        _equal(value, expected["root_poses_w"][None], label)
    _equal(projection["safe_z_clipped"], np.zeros((1, 2), bool), "safe-z", exact=True)
    _equal(projection["projection_reasons"], np.zeros((1, 2), int), "projection reasons", exact=True)
    if projection["projection_source"] != SOURCE or submission["projection_source"] != SOURCE:
        raise ValueError("original per-hand projection source required")
    return dict(ordinal=ordinal, version=VERSION, phase=expected["phase"],
                requested_projected_submitted_targets_agree=True, native_qualified=False)

def reclassify_left_middle(packet):
    """Internal helper, only after validate_projection; all other guards survive."""
    result = deepcopy(packet)
    superseded = {"unexpected_wrist_projection",
                  "left:non_target_not_proven_clear:source:finger33_link",
                  "left:non_target_not_proven_clear:cooked:finger33_link"}
    result["failure_reasons"] = [r for r in result["failure_reasons"] if r not in superseded]
    left = result["sides"]["left"]
    gaps = left["non_target_surface_gap_lower_m"]
    for key in ("source:finger33_link", "cooked:finger33_link"):
        gaps.pop(key, None)
    left["minimum_non_target_gap_lower_m"] = min(gaps.values())
    result["passed"] = not result["failure_reasons"]
    result["terminal_reason"] = None if result["passed"] else "closure_geometry_guard"
    return result

class TripodOwnedGeometry(SegmentClosureGeometry):
    def __init__(self, project):
        super().__init__(project)
        self._pair = plan.load_pair(project)
        self._schedule = plan.action_schedule(self._pair).numpy()
        paths = (plan.CONFIG, "scripts/prepare_dh116_tripod_owned_approach.py",
                 "src/go2_pvcnn/control/dh116/tabletop_tripod_owned_geometry.py",
                 "src/go2_pvcnn/control/dh116/tabletop_bilateral_owned_geometry.py",
                 "scripts/prepare_dh116_pinch_margin_contact.py")
        self.sources["tripod_owned"] = {
            p: hashlib.sha256((Path(project) / p).read_bytes()).hexdigest() for p in paths}
        self.sources["tripod_owned"]["original_24d_path_proof_sha256"] = plan.PROOF_SHA

    def _evaluate(self, raw, exchange):
        checked = validate_projection(self._pair, exchange, schedule=self._schedule)
        packet = reclassify_left_middle(super()._evaluate(raw, exchange))
        whole = {}
        matrices = {}
        for side in ("left", "right"):
            data = raw["hands"][side]
            poses = data["link_poses_xyzw"][0]
            matrices[side] = {
                name: self._pose_matrix(np.r_[pose[:3], pose[6], pose[3:6]])
                for name, pose in zip(data["body_names"], poses)}
            whole[side] = np.concatenate([
                self._transform(vertices, matrices[side][name])
                for shapes in self._geometry[side].values()
                for name, vertices in shapes.items()])
        # The same original source/SDF identity and actual rod nodes are already
        # validated by the parent capture/replay. Geometry is never contact.
        middle = self._sdf(self._grids["left"]["finger33_link"],
                           matrices["left"]["finger33_link"], raw["rod_nodes_m"])
        packet["sides"]["left"]["required_pad_sdf"]["finger33_link"] = middle
        for side, names in self._pair["required_pads"].items():
            for name in names:
                if not required_sdf_valid(packet["sides"][side]["required_pad_sdf"][name]):
                    reason = side + ":sdf_coverage:" + name
                    if reason not in packet["failure_reasons"]:
                        packet["failure_reasons"].append(reason)
        gap = separation_lower(whole["left"], whole["right"])
        packet["inter_hand_source_cooked_separation_lower_m"] = gap
        if gap < 0:
            packet["failure_reasons"].append("inter_hand_separation_not_proven")
        packet.update(version=VERSION, scoped_target_validation=checked,
                      required_pads=deepcopy(self._pair["required_pads"]),
                      original_zero_wrist_scope_replaced_only=True,
                      left_middle_reclassified_as_required=True,
                      passed=not packet["failure_reasons"])
        packet["terminal_reason"] = None if packet["passed"] else "closure_geometry_guard"
        return packet

    def validate(self, packet, exchange):
        if packet.get("version") != VERSION or packet.get("sources") != self.sources:
            raise ValueError("Tripod measured geometry source/version differs")
        rebuilt = self._evaluate(decode_owned(encode_owned(packet["raw"])), exchange)
        if _bytes(rebuilt) != _bytes(packet):
            raise ValueError("Tripod raw geometry recomputation differs")
        return rebuilt
