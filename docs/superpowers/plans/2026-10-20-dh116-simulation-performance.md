# DH116 Simulation Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Reduce repeated host, USD, state-copy, and evidence-flush work in the fixed dual-Panda + DH116 DeformX simulation while preserving contact physics, collision protection, single-core execution, and grasp evidence.

**Architecture:** Add an isolated wall-clock `PerfMeter` and runtime manifest first. Then apply three independently reversible changes to the verifier: wall-clock wire publication, immutable pre/post snapshots with reusable reaction buffers, and buffered evidence flushing. Each change keeps the audit data schema and physical step order unchanged. A separate comparison runner executes matched phases three times and emits machine-readable summaries; no default physical parameter changes are introduced.

**Tech Stack:** Python 3.11, pytest, Isaac Lab/Isaac Sim, PyTorch, NumPy, DeformX/PyElastica, JSONL.

## Global Constraints

- Preserve outer `physics_dt=1e-4 s` and wire `py_dt=1e-5 s`.
- Preserve 16-point spatial contact sampling, reaction-force submission, gravity hold, source-clearance checks, and collision protection.
- Submit one reaction force/torque set per physics step.
- Keep `taskset -c 0`, OMP/OpenBLAS/MKL/Numba single-thread settings, and one Isaac instance.
- Keep the original JSONL evidence row count, fields, and phase ordering.
- Do not change material, friction, target loads, solver iterations, contact sampling, or grasp policy in this plan.
- Do not include `runs/`, `archive/`, caches, logs, secrets, or the local progress fragment `1332` in commits.
- Every performance claim must compare the same phase and physical duration with at least three serial repetitions.

---

### Task 1: Add phase-aware wall-clock instrumentation

**Files:**
- Create: `src/go2_pvcnn/control/dh116/perf_meter.py`
- Create: `tests/test_dh116_perf_meter.py`
- Modify: `scripts/verify_fixed_dual_panda_dh116.py:424-790`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- `PerfMeter(path: Path, physics_dt: float, flush_every_steps: int = 1000)` owns per-step records and aggregate summaries.
- `PerfMeter.begin_step(step: int, stage: str, sim_time: float) -> None` starts a step.
- `PerfMeter.section(name: str) -> context manager` records a non-nested section.
- `PerfMeter.end_step(completed: bool) -> None` writes one `perf.jsonl` row and updates stage aggregates.
- `PerfMeter.close() -> dict[str, object]` flushes and writes `perf_summary.json`.
- `PerfMeter.summary() -> dict[str, object]` returns SPS, RTF, count, mean, P50, P95, and total wall time per stage.

- [ ] **Step 1: Write failing tests for step records and aggregates**

```python
def test_perf_meter_records_sections_and_stage_summary(tmp_path):
    meter = PerfMeter(tmp_path / "perf.jsonl", physics_dt=1e-4)
    meter.begin_step(0, "approach", 0.0)
    with meter.section("control"):
        pass
    with meter.section("physx_step"):
        pass
    meter.end_step(completed=True)
    summary = meter.close()
    row = json.loads((tmp_path / "perf.jsonl").read_text().splitlines()[0])
    assert row["completed"] is True
    assert set(row["sections"]) == {"control", "physx_step"}
    assert summary["stages"]["approach"]["steps"] == 1
    assert summary["stages"]["approach"]["rtf"] >= 0.0

def test_perf_meter_marks_incomplete_step(tmp_path):
    meter = PerfMeter(tmp_path / "perf.jsonl", physics_dt=1e-4)
    meter.begin_step(7, "close", 0.0007)
    meter.end_step(completed=False)
    meter.close()
    row = json.loads((tmp_path / "perf.jsonl").read_text().splitlines()[0])
    assert row["completed"] is False
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `bash scripts/python.sh -m pytest tests/test_dh116_perf_meter.py -q`

Expected: collection or import failure because `perf_meter.py` does not exist.

- [ ] **Step 3: Implement the meter without CUDA synchronization**

Use `time.perf_counter_ns()` for section timing. Reject nested sections with `RuntimeError`, reject duplicate section names in one step, serialize finite floats only, and write one JSON object per completed or failed step. Calculate percentiles from sorted durations using linear interpolation. `close()` must flush, close the file, and write the summary atomically with `Path.with_suffix(".tmp")` followed by `replace()`.

- [ ] **Step 4: Integrate the meter around the existing loop**

Instantiate it only when `--profile` is set. Wrap existing operations in the fixed order `control`, `rod_and_contact`, `source_guard_pre`, `reaction_submit`, `physx_step`, `state_and_sensors`, `source_guard_post`, `wire_visual`, and `evidence`. Put `end_step(completed=False)` in the exception path and close the meter in the outer `finally`. Do not move safety checks or alter the evidence row.

- [ ] **Step 5: Add CLI and manifest identity**

Add `--profile` and record `source_commit`, `device`, `physics_dt`, `wire_py_dt`, `contact_subsamples`, `cpu_affinity`, numeric thread environment, and software versions in `run_manifest.json`. Keep profiling off by default.

- [ ] **Step 6: Run focused and full tests**

Run:

```bash
bash scripts/python.sh -m pytest tests/test_dh116_perf_meter.py -q
bash scripts/python.sh -m pytest tests -q
git diff --check
```

Expected: focused tests pass and the full suite reports zero failures.

- [ ] **Step 7: Commit**

```bash
git add src/go2_pvcnn/control/dh116/perf_meter.py tests/test_dh116_perf_meter.py \
  scripts/verify_fixed_dual_panda_dh116.py README.md CHANGELOG.md
git commit -m "perf: add phase-aware DH116 runtime profiling"
```

### Task 2: Separate wire display publication from physics

**Files:**
- Create: `tests/test_dh116_visual_schedule.py`
- Modify: `src/go2_pvcnn/control/dh116/wire_mesh_publisher.py`
- Modify: `scripts/verify_fixed_dual_panda_dh116.py:620-630`
- Modify: `tests/test_dh116_gui_render.py`

**Interfaces:**
- `should_publish_wire(*, headless: bool, now: float, next_publish: float, ui_hz: float) -> bool` is a pure scheduling function.
- `WireMeshPublisher.publish(snapshot)` updates cached topology once and only mutable point/extent attributes thereafter.

- [ ] **Step 1: Add failing schedule and topology tests**

```python
def test_headless_never_publishes():
    assert should_publish_wire(headless=True, now=10.0, next_publish=0.0, ui_hz=15.0) is False

def test_wall_clock_schedule_publishes_when_due():
    assert should_publish_wire(headless=False, now=2.0, next_publish=1.9, ui_hz=15.0) is True
    assert should_publish_wire(headless=False, now=2.0, next_publish=2.1, ui_hz=15.0) is False

def test_publisher_keeps_topology_after_first_publish(fake_stage):
    publisher = WireMeshPublisher(fake_stage, radius=0.008)
    publisher.publish(snapshot_a)
    topology = publisher.topology_arrays()
    publisher.publish(snapshot_b)
    assert publisher.topology_arrays() == topology
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `bash scripts/python.sh -m pytest tests/test_dh116_visual_schedule.py tests/test_dh116_gui_render.py -q`

Expected: missing scheduler or topology-cache behavior.

- [ ] **Step 3: Implement wall-clock scheduling**

Add `--ui-hz` with default `15.0`; set the next publication deadline from `time.perf_counter()`. Keep `sim.step(render=False)` in every physics step. On a due GUI frame, publish the current post snapshot once and call `sim.render()` once. Never call a second `sim.step`, and never publish in headless mode.

- [ ] **Step 4: Cache fixed mesh topology**

Make `WireMeshPublisher` build face counts and indices during initialization. Subsequent calls update only points and extent. If the node count or radial resolution changes, raise a clear `ValueError` and require a new publisher rather than silently reusing incompatible topology.

- [ ] **Step 5: Run focused/full tests and commit**

Run:

```bash
bash scripts/python.sh -m pytest tests/test_dh116_visual_schedule.py tests/test_dh116_gui_render.py -q
bash scripts/python.sh -m pytest tests -q
```

Then commit:

```bash
git add src/go2_pvcnn/control/dh116/wire_mesh_publisher.py \
  scripts/verify_fixed_dual_panda_dh116.py tests/test_dh116_visual_schedule.py \
  tests/test_dh116_gui_render.py
git commit -m "perf: decouple DH116 wire display from physics steps"
```

### Task 3: Reuse immutable pre/post snapshots and reaction buffers

**Files:**
- Create: `src/go2_pvcnn/control/dh116/runtime_snapshot.py`
- Create: `tests/test_dh116_runtime_snapshot.py`
- Modify: `scripts/verify_fixed_dual_panda_dh116.py:430-700`
- Modify: `src/go2_pvcnn/control/dh116/grasp_evidence.py`

**Interfaces:**
- `RuntimeSnapshot(step: int, stage: str, sample_time: float, token: int, values: Mapping[str, object])` is immutable.
- `SnapshotStore.capture_pre(...) -> RuntimeSnapshot` and `capture_post(...) -> RuntimeSnapshot` create distinct tokens.
- `ReactionBuffer.allocate(device, count) -> ReactionBuffer` and `.clear_()` reuse force, torque, position, and index tensors.

- [ ] **Step 1: Write failing tests for token separation and buffer reuse**

```python
def test_pre_and_post_snapshots_have_distinct_tokens():
    store = SnapshotStore()
    pre = store.capture_pre(step=3, stage="approach", sample_time=.0003, values={"q": [1]})
    post = store.capture_post(step=3, stage="approach", sample_time=.0003, values={"q": [2]})
    assert pre.token != post.token
    assert pre.values["q"] == [1]
    assert post.values["q"] == [2]

def test_reaction_buffer_clears_in_place():
    buffer = ReactionBuffer.allocate("cpu", 4)
    identity = id(buffer.force)
    buffer.force.fill_(1.0)
    buffer.clear_()
    assert id(buffer.force) == identity
    assert bool((buffer.force == 0).all())
```

- [ ] **Step 2: Run tests and verify failure**

Run: `bash scripts/python.sh -m pytest tests/test_dh116_runtime_snapshot.py -q`

Expected: missing snapshot and buffer types.

- [ ] **Step 3: Implement immutable snapshots and reusable arrays**

Copy only at capture boundaries. Expose read-only mappings using `MappingProxyType`; reject attempts to mutate the snapshot object. Allocate reaction tensors once per arm and preserve device/dtype/shape. `clear_()` must be the only per-step reset operation.

- [ ] **Step 4: Integrate consumers without changing step order**

Use pre snapshots for control, prediction, and wire display scheduling. Use post snapshots after `sim.step` and `arm.update` for safety, contact, slip, and success checks. Replace duplicate index lookup and reaction tensor allocation with cached objects. Assert that each reaction submission references the current post token and occurs exactly once per step.

- [ ] **Step 5: Verify numerical identity**

Run the existing runtime and reaction tests plus a deterministic offline replay. Compare pre/post optimized output against the baseline for joint states, wire position/director/velocity, reaction impulse, minimum clearance, and stage transitions. Require exact evidence-row count and no changed safety decisions.

- [ ] **Step 6: Run full tests and commit**

```bash
bash scripts/python.sh -m pytest tests/test_dh116_runtime_snapshot.py tests/test_dh116_stage_reset_runtime.py tests/test_dh116_coordinated_closure.py -q
bash scripts/python.sh -m pytest tests -q
git add src/go2_pvcnn/control/dh116/runtime_snapshot.py \
  tests/test_dh116_runtime_snapshot.py scripts/verify_fixed_dual_panda_dh116.py \
  src/go2_pvcnn/control/dh116/grasp_evidence.py
git commit -m "perf: reuse DH116 runtime snapshots and reaction buffers"
```

### Task 4: Add buffered evidence flush policy

**Files:**
- Create: `tests/test_dh116_evidence_flush_policy.py`
- Modify: `src/go2_pvcnn/control/dh116/evidence_journal.py`
- Modify: `scripts/verify_fixed_dual_panda_dh116.py`

**Interfaces:**
- `EvidenceJournal(path, flush_policy="audit", flush_every=512, flush_seconds=.5)`.
- `EvidenceJournal.append(row, *, force_flush=False)`.
- `EvidenceJournal.flush()` remains public and is called on phase changes, exceptions, stop requests, and close.

- [ ] **Step 1: Add failing policy tests**

```python
def test_buffered_policy_flushes_on_count(tmp_path, monkeypatch):
    journal = EvidenceJournal(tmp_path / "trajectory.jsonl", flush_policy="buffered", flush_every=2)
    journal.append({"step": 0})
    assert journal.flush_count == 0
    journal.append({"step": 1})
    assert journal.flush_count == 1
    journal.close()

def test_phase_transition_forces_flush(tmp_path):
    journal = EvidenceJournal(tmp_path / "trajectory.jsonl", flush_policy="buffered")
    journal.append({"step": 0}, force_flush=True)
    assert journal.flush_count == 1
    journal.close()
```

- [ ] **Step 2: Implement policy with audit compatibility**

Keep `audit` behavior byte-for-byte compatible for rows and flush each append. In `buffered`, encode and write each row, flush by count or monotonic elapsed time, and always flush on explicit `force_flush`. Do not call `fsync` per step. Preserve exclusive file creation and strict JSON finite-value validation.

- [ ] **Step 3: Integrate stage and failure flushes**

Pass the CLI policy through the manifest. Force flush on every stage transition, `STOP_REQUESTED`, safety failure, exception, normal close, and stage-state export. Keep the complete JSONL row stream in both policies.

- [ ] **Step 4: Verify and commit**

```bash
bash scripts/python.sh -m pytest tests/test_dh116_evidence_flush_policy.py tests/test_dh116_stage_reset_runtime.py -q
bash scripts/python.sh -m pytest tests -q
git add src/go2_pvcnn/control/dh116/evidence_journal.py \
  tests/test_dh116_evidence_flush_policy.py scripts/verify_fixed_dual_panda_dh116.py
git commit -m "perf: add buffered DH116 evidence flushing"
```

### Task 5: Add reproducible performance comparison runner

**Files:**
- Create: `scripts/run_performance.py`
- Create: `scripts/collect_performance.py`
- Create: `tests/test_dh116_performance_runner.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- `scripts/run_performance.py --variant {baseline,visual,snapshot,buffered} --phase PHASE --steps N --repeat R --device {cuda:0,cpu}` validates arguments, creates exclusive outputs, and launches one verifier process at a time.
- `scripts/collect_performance.py RUN_DIR... --output REPORT.json` reads `perf_summary.json` and manifests, compares matched physical duration, and emits median/P95 speedup and physics identity checks.

- [ ] **Step 1: Write failing argument and comparison tests**

```python
def test_runner_rejects_nonpositive_repeats():
    with pytest.raises(SystemExit):
        parse_args(["--variant", "baseline", "--phase", "approach", "--repeat", "0"])

def test_report_uses_median_wall_time(tmp_path):
    report = compare_runs([{"wall_s": 3.0}, {"wall_s": 2.0}, {"wall_s": 4.0}])
    assert report["median_wall_s"] == 3.0
```

- [ ] **Step 2: Implement isolated runner**

Preserve all verifier validation, single-instance checks, output exclusivity, single-core default, and fixed physical parameters. `--device cpu` is an experimental comparison only and is recorded in the manifest. Reject combinations that alter the phase success gates or mix physical durations.

- [ ] **Step 3: Implement report generation**

Require three completed runs per variant. Match stage and simulated duration, report cold-start separately, calculate median/P95 wall time and SPS/RTF, and compare evidence row counts, stage transitions, contact loads, reaction impulse, minimum clearance, slip, and wire state. Mark a group `incomparable` if a run fails earlier or has different physical work.

- [ ] **Step 4: Run offline tests and commit**

```bash
bash scripts/python.sh -m pytest tests/test_dh116_performance_runner.py -q
bash scripts/python.sh -m pytest tests -q
git add scripts/run_performance.py scripts/collect_performance.py \
  tests/test_dh116_performance_runner.py README.md CHANGELOG.md
git commit -m "perf: add reproducible DH116 performance comparisons"
```

### Task 6: Execute matched hardware measurements

**Files:**
- Create: `runs/perf_20261020/` (ignored runtime output only)
- Create: `docs/superpowers/reports/2026-10-20-dh116-performance-report.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Record environment without starting Isaac**

Run the project environment probe, `nvidia-smi`, memory and filesystem checks, and capture the source commit and clean/dirty state in the report.

- [ ] **Step 2: Run three serial repetitions per variant**

Use the same assets, seed, phase and physical duration for A0, A1, A2 and A3. Keep `taskset -c 0`, one Isaac instance, and `physics_dt=1e-4`. Preserve GUI only for the visual variant; run headless comparisons separately.

- [ ] **Step 3: Collect and inspect reports**

Run `scripts/collect_performance.py` and inspect every run for completed steps, unchanged safety decisions, reaction balance, contact evidence, wire state and stage transitions.

- [ ] **Step 4: Decide the next optimization**

Only if a measured hotspot remains, write a separate approved design for CPU/GPU or contact-kernel work. If no matched speedup is measured, keep the candidate as an optional diagnostic and do not claim acceleration.

- [ ] **Step 5: Commit report and documentation**

```bash
git add docs/superpowers/reports/2026-10-20-dh116-performance-report.md README.md CHANGELOG.md
git commit -m "docs: record DH116 performance comparison evidence"
```

## Review checkpoints

- After Task 1: inspect `perf.jsonl` schema and full regression before changing runtime behavior.
- After Task 2: compare wire topology and stage evidence before proceeding.
- After Task 3: require deterministic physical identity before reusing snapshots.
- After Task 4: verify audit mode remains unchanged before using buffered mode.
- After Task 5: review runner argument validation and output isolation before starting Isaac.
- After Task 6: publish only measured results; do not infer whole-run acceleration from one section.

