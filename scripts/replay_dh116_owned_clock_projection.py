"""Read-only A/B comparison of the original complete physical-ledger audit."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.prepare_dh116_effective_right_native import inspect_interval
from go2_pvcnn.control.dh116.tabletop_owned_clock_projection import clock_projection_scope


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = sha(source)
    rows = []
    # Reverse the second pair so filesystem warmup is not credited as a fix.
    for index, mode in enumerate(("original", "projection", "projection", "original")):
        context = nullcontext() if mode == "original" else clock_projection_scope()
        start = time.perf_counter()
        with context:
            result = inspect_interval(source)
        elapsed = time.perf_counter() - start
        payload = json.dumps(result, sort_keys=True, allow_nan=False).encode()
        target = output / f"{index:02d}-{mode}.json"
        with target.open("xb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        rows.append(dict(mode=mode, elapsed_s=elapsed, result_sha256=sha(target),
                         physical_rows=result["measured_physics_rows"]))
    same = len({row["result_sha256"] for row in rows}) == 1
    after = sha(source)
    report = dict(kind="read_only_complete_ledger_inspection_ABBA", source=str(source),
                  source_sha256=before, source_unchanged=after == before,
                  exact_result_match=same, timings=rows, native_execution_performed=False,
                  native_qualification=False, teacher_admission=False, new_physics_steps=0)
    (output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    if not same or before != after:
        raise ValueError("projection changed audit or original evidence")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.physical, args.output), indent=2))
