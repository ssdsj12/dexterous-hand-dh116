"""Join original policy/physics evidence to native M2 replays without starting Isaac."""
import argparse
import hashlib
import json
from pathlib import Path


def records(path):
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError(f"incomplete JSONL record in {path}")
            yield json.loads(line)


def key(row):
    return row["episode_uid"], row["physics_step"]


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(source, recorded, other_arms):
    policy_path = source / "policy_ppo_diagnostics.jsonl"
    source_hash = file_digest(policy_path)
    policy = {}
    for row in records(policy_path):
        k = row["env_id"], row["episode_id"], row["policy_step"]
        if k in policy:
            raise ValueError(f"duplicate policy record {k}")
        policy[k] = row
    terminal_envs = {row["env_id"] for row in policy.values() if row["done"]}
    original = {}
    for row in records(source / "physics_action_diagnostics.jsonl"):
        if row["env_id"] in terminal_envs:
            k = key(row)
            if k in original:
                raise ValueError(f"duplicate physics record {k}")
            original[k] = row
    original_events = {k for k, row in original.items() if row["terminal_sample"]}
    if not original_events:
        raise ValueError("no original terminal to audit")
    results = {}
    initial_hash = None
    for path in [recorded, *other_arms]:
        summary = json.loads((path / "joint_replay_summary.json").read_text())
        arm = summary["arm"]
        if arm in results or (path == recorded and arm != "recorded"):
            raise ValueError("expected one recorded baseline and distinct replay arms")
        if Path(summary["source"]).name != source.name:
            raise ValueError("replay source directory differs from historical run")
        if summary.get("source_policy_sha256", source_hash) != source_hash:
            raise ValueError("source policy digest differs from replay")
        if initial_hash is None:
            initial_hash = summary["initial_state_sha256"]
        if summary["initial_state_sha256"] != initial_hash:
            raise ValueError("physical reset fingerprint differs across arms")
        seen = set()
        events = []
        at_original_event = []
        for replay in records(path / "joint_replay.jsonl"):
            k = key(replay)
            if k in seen:
                raise ValueError(f"duplicate replay physics record {k}")
            seen.add(k)
            if arm == "recorded":
                expected = original[k]
                p = policy[expected["env_id"], expected["episode_id"], expected["policy_step"]]
                pairs = (("env_id", "env_id"), ("policy_step", "policy_step"),
                         ("timestamp_s", "timestamp_s"), ("physical_action24", "action24"),
                         ("live", "live_before_substep"), ("terminal", "terminal_sample"),
                         ("contact_loads_n", "contact_loads_n"))
                if any(replay[a] != expected[b] for a, b in pairs) or replay["actor_raw24"] != p["action24"]:
                    raise ValueError(f"recorded replay diverges at {k}")
            if k in original_events:
                at_original_event.append(replay)
            if replay["terminal"]:
                violations = []
                for hand, names in enumerate(summary["joint_names"]):
                    for joint, name in enumerate(names):
                        value = replay["q"][hand][joint]
                        lower, upper = replay["limits"][hand][joint]
                        if value < lower - .02 or value > upper + .02:
                            violations.append({"hand": hand, "joint": name, "q": value,
                                               "bounds": [lower, upper]})
                events.append({"sample": replay, "out_of_bounds": violations})
        if arm == "recorded" and seen != original.keys():
            raise ValueError("missing original substeps")
        if {key(row) for row in at_original_event} != original_events:
            raise ValueError("missing at least one original episode/substep in replay")
        if {key(e["sample"]) for e in events} != {key(e) for e in summary["terminal_events"]}:
            raise ValueError("replay summary disagrees with terminal samples")
        results[arm] = {"directory": str(path), "events": events,
                        "at_original_event": at_original_event}
    return {"source_policy_sha256": source_hash, "initial_state_sha256": initial_hash,
            "matched_physics_samples": len(original), "episodes": sorted(terminal_envs), "arms": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("recorded", type=Path)
    parser.add_argument("other_arms", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = audit(args.source, args.recorded, args.other_arms)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
