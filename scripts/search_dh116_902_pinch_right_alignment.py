"""Screen alternate right 902 Pinch wrist/thumb alignments on archived material."""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.fit_dh116_902_pinch_alignment import MOVE_SCALE, _boundary_context, _side_fit
from scripts.verify_dh116_902_flexible_reset import sha


BASELINE = dict(pad_gaps_m=[.000644681245256404, .000024220019665641655],
                pad_stations=[4, 4], nonpad_hull="source_colliders:finger13_link",
                nonpad_clearance_m=.000275597272321515,
                result_sha256="2328a0643f260c54517c76540d196e3f2d5d616162c3454cc1c4f40eeaf58949")
# Millimetres/radians local to each measured material translation. The inherited
# fitter intersects the translation and joint spans with the original limits.
SEARCH_SCALE = np.array([.004, .005, .004, .32, .32, .32, .25, .2, .22])


class StationSeed(NamedTuple):
    station: int
    approach: str
    delta: np.ndarray

    @property
    def translation(self):
        return self.delta[:3]


def station_seeds(nodes, original, rotation, *, radius, initial_delta=None):
    """Move with archived material and roll both ways around the local rod axis."""
    nodes = np.asarray(nodes, dtype=float)
    if (nodes.shape != (41, 3) or not np.isfinite(nodes).all()
            or type(original) is not int or not 0 <= original < 20
            or type(radius) is not int or not 0 <= radius <= 2):
        raise ValueError("bounded right-hand stations on the original measured wire required")
    if initial_delta is None:
        start = np.zeros(9)
        approaches = (("material_translation", 0., [0., 0., 0.]),
                      ("positive_tangent_roll", .22, [.06, .075, 0.]),
                      ("negative_tangent_roll", -.22, [-.06, -.04, 0.]))
    else:
        start = np.asarray(initial_delta, dtype=float)
        if start.shape != (9,) or not np.isfinite(start).all():
            raise ValueError("finite warm start in the original measured frame required")
        approaches = (("warm", 0., [0., 0., 0.]),
                      ("warm_plus", .06, [0., .015, 0.]),
                      ("warm_minus", -.06, [0., -.015, 0.]))
    choices = []
    for station in range(max(0, original - radius), min(20, original + radius + 1)):
        translation = nodes[station] - nodes[original]
        tangent = nodes[station + 1] - nodes[max(0, station - 1)]
        length = np.linalg.norm(tangent)
        if length <= 0.:
            raise ValueError("measured station requires a nonzero tangent")
        axis = rotation.inv().apply(tangent / length)
        for approach, angle, thumb in approaches:
            choices.append(StationSeed(station, approach, start + np.r_[translation, angle * axis, thumb]))
    return choices


def _baseline_delta(path, source):
    if sha(path) != BASELINE["result_sha256"]:
        raise ValueError("sealed earlier Pinch result required for warm continuation")
    report = json.loads(path.read_bytes())
    if any(report[key] != source[key] for key in ("outer_sha256", "last_row_sha256", "archive_sha256")):
        raise ValueError("warm continuation must use the same measured approach")
    measured = report["hands"]["right"]["measured"]
    candidate = report["hands"]["right"]["best"]
    position = np.asarray(candidate["pose_wxyz"][:3]) - measured["pose_wxyz"][:3]
    rotations = np.asarray([measured["pose_wxyz"], candidate["pose_wxyz"]])[:, [4, 5, 6, 3]]
    start, end = Rotation.from_quat(rotations)
    joints = np.asarray(candidate["active_q_rad"][:3]) - measured["active_q_rad"][:3]
    return np.r_[position, (start.inv() * end).as_rotvec(), joints]


def search(receipt, *, radius=1, max_evaluations=48, baseline_result=None):
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 96:
        raise ValueError("explicit bounded CPU iteration count required")
    if type(radius) is not int or not 0 <= radius <= 2:
        raise ValueError("bounded right-hand station radius required")
    context, models, source = _boundary_context(receipt, sides=("right",))
    initial_delta = None if baseline_result is None else _baseline_delta(baseline_result, source)

    def choices(station, rotation):
        if station != BASELINE["pad_stations"][0]:
            raise ValueError("archived right thumb station differs from the established baseline")
        return station_seeds(context["nodes"][-1], station, rotation, radius=radius,
                             initial_delta=initial_delta)

    result = _side_fit("right", **context, model=models["right"], max_evaluations=max_evaluations,
                       starts=1, alternate=choices, move_scale=SEARCH_SCALE if initial_delta is None else MOVE_SCALE)
    return dict(version="dh116-902-pinch-right-alternate-alignment-v1", **source, **result,
                search_source_sha256=sha(Path(__file__)), baseline=BASELINE,
                budget=dict(station_radius=radius, max_evaluations_per_seed=max_evaluations),
                scope="cpu_geometric_right_endpoint_only", warm_start=initial_delta is not None,
                command_path_qualified=False, native_contact_verified=False,
                teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, help="sealed prior fit to use as a warm start")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--max-evaluations", type=int, default=48)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh alternate alignment output required")
    result = search(args.receipt, radius=args.radius, max_evaluations=args.max_evaluations,
                    baseline_result=args.baseline_result)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if result["best"]["endpoint_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
