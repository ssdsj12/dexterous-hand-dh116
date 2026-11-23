"""Current-source CPU endpoint probe, separate from the historical clock gate."""
from hashlib import sha256
from pathlib import Path

from . import tabletop_native_clock_consumers_v2 as legacy


VERSION = "dh116-current-clock-consumers-v3"
SOURCE_SHA256 = {
    "tabletop_deformable_bimanual_step.py":
        "e144640a0310ab4824e482a2d643165239a74bbfdee66cae42289ff8cb3ee7f2",
    "tabletop_deformable_live_geometry.py":
        "19e2535dc9ba86556d41284f15fc65bbf30ab7f261653488f4f03e64fe8a4b17",
    "tabletop_deformable_dual_evidence.py":
        "df54cff14e78ab261b8b76e5e488d7423a21dfb91fc5d13e397a1c35c07a1150",
    "tabletop_deformable_dual_hand_episode.py":
        "95a8a1f71f1ec4c75d7335c99fcf6c764f4a36f53e33bf7113eef173738934f6",
    "tabletop_dual_hand_rod_runtime.py":
        "2526072d03efbd2d4c37818b4aaca8f79ca2fc51ed55e9c8b6bdb7a8c1eb94ed",
    "tabletop_native_clock_consumers_v2.py":
        "0997ddefd0cfec777830e83ff78e92ef07513e9acbc4f92b4605d6fe4ae11fe8",
    "tabletop_native_training_clock.py":
        "90bc0e7bab5117d5724d35ea5faaee25cb378981b1f3d58815d295c34439a32b",
}


def _verified_sources():
    root = Path(__file__).parent
    verified = {}
    for name, expected in SOURCE_SHA256.items():
        digest = sha256((root / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"current clock consumer source SHA differs: {name}")
        verified[name] = digest
    return verified


def probe_current_clock_consumers():
    """Run the existing endpoint hypotheses on the independently pinned current readers."""
    sources = _verified_sources()
    rows = [
        row for mode in legacy.MODES for step in legacy.BOUNDARIES
        for row in legacy._probe_endpoint(step, mode)
    ]
    if _verified_sources() != sources:
        raise ValueError("clock consumer sources changed during the probe")
    table = dict(
        schema=VERSION, source_sha256=sources, rows=rows, native_steps=0,
        scope="cpu_endpoint_reader_probe", source_fields_distinct=True,
        actual_sdk_precision_qualified=False, native_ready=False,
    )
    table["table_sha256"] = legacy.clock_table_sha256_v2(table)
    return table
