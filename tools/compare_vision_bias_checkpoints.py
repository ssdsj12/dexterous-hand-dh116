"""Compare legacy M0 actor means on one recorded set of raw RGB-D actor inputs."""
import argparse
import json
from pathlib import Path

import torch

from go2_pvcnn.control.dh116.tabletop_vision_action_contract import LEGACY_VERSION
from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import evaluation_contract, file_sha256
from go2_pvcnn.control.dh116.tabletop_vision_inference import load_vision_inference
from go2_pvcnn.control.dh116.tabletop_vision_student import POLICY_DIM


def compare(observations, checkpoints):
    batch = torch.load(observations, map_location="cpu", weights_only=True)
    sensor_contract = batch["contract"]
    observations_tensor = batch["policy"]
    groups = sensor_contract["groups"]
    if sensor_contract["version"] != LEGACY_VERSION:
        raise ValueError("this comparison requires the legacy bounded visual actor")
    for run in [observations.parent, *(p.parent for p in checkpoints)]:
        mirror_path = run / "vision_mirror_preprocessing.json"
        if mirror_path.exists() and json.loads(mirror_path.read_text())["mode"] != "m0_raw":
            raise ValueError(f"expected raw M0 data and checkpoints: {run}")
    if observations_tensor.shape != (sum(g["num_envs"] for g in groups), POLICY_DIM):
        raise ValueError("actor inputs do not match saved group counts")
    if not observations_tensor.isfinite().all():
        raise ValueError("non-finite sensor inputs")
    result = dict(
        observations=str(observations), observations_sha256=file_sha256(observations),
        dimensions=list(observations_tensor.shape),
        statistic="M0 deterministic actor mean; population SD across saved reset rows",
        palm_z_action_indices={"left": 14, "right": 17}, checkpoints=[],
    )
    actor_contract = evaluation_contract(sensor_contract, len(observations_tensor))
    for path in checkpoints:
        policy = load_vision_inference(path, actor_contract)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if policy.checkpoint_sha256 != file_sha256(path):
            raise RuntimeError(f"checkpoint changed while reading its log_std: {path}")
        actions = torch.cat([policy.act(chunk) for chunk in observations_tensor.split(32)])
        stage = dict(path=str(path), sha256=policy.checkpoint_sha256, iteration=checkpoint["iteration"],
                     log_std24=checkpoint["policy"]["log_std"].tolist(), by_style={})
        for group, block in zip(groups, actions.split([g["num_envs"] for g in groups])):
            stage["by_style"][group["label"]] = dict(
                env_count=len(block), action24_mean=block.double().mean(0).tolist(),
                action24_std=block.double().std(0, unbiased=False).tolist(),
            )
        result["checkpoints"].append(stage)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.observations, args.checkpoints)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
