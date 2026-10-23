"""Promote the reviewed forward-feasibility candidate into a pinned pose artifact."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "runs/forward_feasibility3/summary.json"
    output = root / "runs/forward_feasibility3/pose.json"
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    data = json.loads(source.read_text())
    asset_sha = data["layers"][next(p for p in data["layers"] if p.endswith("fixed_dual_panda_dh116.usd"))]
    pose = {
        "schema": "dh116-static-pose-v1",
        "asset_sha256": asset_sha,
        "layers": data["layers"],
        "states": data["states"],
        "grasp_qualified": False,
        "source": {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "operation": "promoted reviewed forward_feasibility3 palm-down candidate 3; offline geometry only",
        },
        "offline_screen": data.get("source_screen"),
        "scope": "offline forward/palm-down candidate; no physics or grasp credit",
    }
    output.write_text(json.dumps(pose, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "asset_sha256": asset_sha}, indent=2))


if __name__ == "__main__":
    main()
