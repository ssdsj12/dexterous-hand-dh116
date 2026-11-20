#!/usr/bin/env python3
"""Recompute the initial-policy decisions of an archived, seeded Pinch run."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, _digest
from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_factories import (
    FRESH_POLICY_EXPLORATION_SEED,
    FRESH_POLICY_INITIALIZATION_SEED,
)
from go2_pvcnn.control.dh116.tabletop_native_teacher_model import NativeContextTeacher
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, checked_file


MAX_ACTION_ERROR = 1e-7


def audit(journal, journal_sha, training, training_sha, *, project):
    rows = [decode_owned(json.loads(line)) for line in checked_file(journal, journal_sha).splitlines()]
    if [row["sequence"] for row in rows] != list(range(len(rows))):
        raise ValueError("complete original journal sequence required")
    config = json.loads(checked_file(training, training_sha))

    def operations(name):
        return [row for row in rows if row["operation"] == name]

    origin, = operations("collector_origin")
    seed_row, = operations("fresh_policy_rng")
    decisions = operations("decision")
    if not decisions or not origin["sequence"] < seed_row["sequence"] < decisions[0]["sequence"]:
        raise ValueError("initial policy and seed must precede the first decision")
    seed = seed_row["payload"]
    expected = dict(version="dh116-fresh-policy-cpu-rng-v1", device_scope="cpu",
                    initialization_seed=FRESH_POLICY_INITIALIZATION_SEED,
                    exploration_seed=FRESH_POLICY_EXPLORATION_SEED, native_qualification=False,
                    teacher_admission=False, deterministic_physics_claimed=False)
    if any(seed.get(name) != value or type(seed[name]) is not type(value) for name, value in expected.items()):
        raise ValueError("explicit fresh Worker CPU seed evidence required")

    verified = []
    profile = BackendProfile.load(project)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"), torch.inference_mode():
        torch.default_generator.manual_seed(seed["initialization_seed"])
        model = NativeContextTeacher(config["model_config"], backend_profile=profile)
        weights = _digest(model.state_dict())
        if weights != seed["initial_weights_sha256"] or weights != origin["payload"]["policy"]["weights"]:
            raise ValueError("reconstructed initial policy weights differ")
        torch.default_generator.manual_seed(seed["exploration_seed"])
        state_sha = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        if state_sha != seed["exploration_state_sha256"]:
            raise ValueError("recorded CPU exploration RNG state differs")

        initial = origin["payload"]["policy"]
        for row in decisions:
            packet = row["payload"]
            if packet["policy"] != initial:
                break
            source_index = packet["source_pack_sequence"]
            if type(source_index) is not int or not 0 <= source_index < row["sequence"]:
                raise ValueError("decision requires an earlier observation source")
            source = rows[source_index]
            if (source["operation"] != "teacher_input_source" or source["payload"]["purpose"] != "decision"
                    or _digest(packet["observation"]) != source["payload"]["sha256"]["observation"]):
                raise ValueError("decision and recorded observation source differ")
            obs = {name: torch.from_numpy(values.copy()) for name, values in packet["observation"].items()}
            mean = model.latent_mean(obs)
            std = model.log_std.exp().expand_as(mean)
            predicted = (mean + std * torch.randn_like(mean)).tanh()
            actual = torch.from_numpy(packet["sampled_action24"])
            if predicted.shape != actual.shape:
                raise ValueError("recorded canonical sampled action shape differs")
            error = float((predicted - actual).abs().max())
            if not error <= MAX_ACTION_ERROR:
                raise ValueError(f"sampled action replay differs at sequence {row['sequence']}: {error}")
            verified.append(dict(sequence=row["sequence"], max_action_abs_error=error))

    if not verified:
        raise ValueError("no decision matched the initial policy")
    return dict(journal_sha256=journal_sha, weights_sha256=weights, actions=verified,
                decisions_after_initial_policy=len(decisions) - len(verified),
                replay_max_abs=max(row["max_action_abs_error"] for row in verified),
                scope="CPU replay of archived pre-action inputs", native_qualification=False,
                teacher_admission=False, actual_isaac_steps=0, ppo_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.journal, args.journal_sha256, args.training, args.training_sha256,
                   project=Path(__file__).resolve().parents[1])
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
