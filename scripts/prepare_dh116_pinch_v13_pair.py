"""Bind the validated earliest-approach right Pinch Reset to the original left and rod."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import load_bound_pair_proposal
from scripts.audit_dh116_pinch_initial_pose import checked_json, digest
from scripts.prepare_dh116_paired_pinch_reset import independent_sweep_gap, make_joint_seed
from scripts.validate_dh116_right_pinch_runtime_clearance import minimum_clearance, qualifies


PARENT_SHA = "0e49af8b5bb205c12c1ee89883ed70e42649166c7a2188c7eac463533153457c"
STATIC_SHA = "8f35a28bb56bdbb18a8ee4a96c90f8b2416dd986f59c60c3f1c96601e5bd48e3"
REPLAY_SHA = "f0edbf22bc423bb306fc235d721450b35a65159498935dd86450983ef09e4981"
SELECTION = "pinch-first-approach-reserve-2"


def prepare_pair(parent_path, static_path, replay_path):
    parent = checked_json(Path(parent_path), PARENT_SHA)
    static = checked_json(Path(static_path), STATIC_SHA)
    replay = checked_json(Path(replay_path), REPLAY_SHA)
    analytic, = [row for row in replay["candidates"] if row["candidate_id"] == SELECTION]
    static_row, = [row for row in static["rows"] if row["static_candidate"]["candidate_id"] == SELECTION]
    if (analytic["conditional_pass"] is not True or replay["qualification"] is not False
            or analytic["minimum_full_measured_rotation_replay_m"] < .00220
            or analytic["v12_terminal_distal_gap_m"]["finger23_link"] > .00205
            or static_row["qualifies"] is not True or static_row["change"][-1] != .0055):
        raise ValueError("original offline Pinch clearance and early approach evidence required")

    candidate = deepcopy(static_row["static_candidate"])
    path = deepcopy(static_row["complete_path"])
    if not qualifies(candidate, path):
        raise ValueError("right Pinch candidate failed the original closure and table checks")
    candidate["candidate_id"] = path["candidate_id"] = "pinch-v13-right-positive-approach-window"
    candidate["parent_candidate_id"] = parent["right"]["candidate"]["candidate_id"]
    candidate["local_change"] = dict(root_x_m=-.0004, root_z_m=.000145,
                                    thumb_seed_rad=.002, index_seed_rad=.0055)

    pair = deepcopy(parent)
    pair["right"] = dict(candidate=candidate, path=path, source_sha256=parent["right"]["source_sha256"])
    pair["right_continuous_clearance_m"] = minimum_clearance(path)
    pair["reset_seed"] = make_joint_seed(pair["left"]["candidate"], candidate)
    pair["independent_hand_y_gap_m"] = independent_sweep_gap(
        pair["left"]["path"]["continuous_hand_y_bounds_m"], path["continuous_hand_y_bounds_m"])
    return pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "static", "replay", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv)
    pair = prepare_pair(args.parent, args.static, args.replay)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(pair, stream, indent=2, allow_nan=False)
        stream.write("\n")
    bound = dict(path=str(args.output.resolve()), sha256=digest(args.output))
    load_bound_pair_proposal(bound)
    print(json.dumps(bound))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
