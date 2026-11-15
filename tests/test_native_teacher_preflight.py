"""Pure CPU contracts. Fixture scope never grants native qualification."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_native_teacher_profile import ROOT


def api():
    path = ROOT / "src/go2_pvcnn/control/dh116/tabletop_native_teacher_preflight.py"
    assert path.is_file(), "C01 two-stage preflight implementation missing"
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_native_teacher_preflight")


def config(style="pinch"):
    return api().candidate_config(style=style)


def test_smoke_capabilities_exclude_full_success():
    required = api().required_capabilities("approach_wiring_smoke_v1")
    assert {"same_clock_distance", "native_health", "resolved_reset", "common66",
            "rod41x13", "accepted_request24", "sdk_target_readback", "original_guards",
            "current_parameters", "task_context", "eight_sdk_per_control", "finite_budget"} <= required
    assert not {"hold_motion", "fixed_material_slip", "tangential_slip",
                "no_external_support", "required_pad_contact"} & required


def test_full_requires_real_task_history():
    m = api()
    smoke = m.required_capabilities("approach_wiring_smoke_v1")
    full = m.required_capabilities("flexible_train_r1_full")
    assert smoke < full
    assert {"hold_motion", "fixed_material_slip", "tangential_slip", "required_pad_contact",
            "bilateral_support", "off_table", "no_external_support", "continuous_history"} <= full
    with pytest.raises(ValueError, match="profile"):
        m.required_capabilities("ready=True")


@pytest.fixture(scope="module")
def current_profile():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    return BackendProfile.load(ROOT)


@pytest.fixture
def missing_teacher_modules(monkeypatch):
    # Explicit absence fixture: integrating C02 must not change the pending
    # branch's meaning, and the real implementation presence is tested below.
    missing={ROOT/"src/go2_pvcnn/control/dh116"/name for name in (
        "tabletop_native_teacher_model.py","tabletop_native_teacher_context.py")}
    original=Path.is_file
    monkeypatch.setattr(Path,"is_file",lambda path: False if path in missing else original(path))


@pytest.mark.parametrize("style", ["pinch", "tripod", "power"])
def test_missing_teacher_modules_remain_pending_and_style_specific(current_profile,style,missing_teacher_modules):
    result = api().static_preflight(config(style), project=ROOT, backend_profile=current_profile)
    assert result["new_native_steps"] == 0
    assert result["live_reset_verified"] is False
    assert result["static_ready"] is False
    names = {c["name"]: c for c in result["components"]}
    assert names["backend"]["valid"] is True
    assert names["resolved_reset_manifest"]["valid"] is False
    assert names["finite_budget"]["valid"] is False
    assert names["model"]["blocked_reason"] == "model_implementation_pending"
    assert names["context"]["blocked_reason"] == "context_implementation_pending"
    assert names["same_clock_distance"]["valid"] is (style == "pinch")
    assert names["retained_reset"]["applicable"] is (style == "pinch")
    assert result["ledger"]["measured_selected"] == 66
    assert result["ledger"]["unknown_nan_false"] == 305
    assert "formal_r0_numeric_source_missing" not in result["blocked_reasons"]
    assert all(set(("source_complete", "applicable", "valid", "reader", "source_sha256",
                    "blocked_reason", "release_condition")) <= set(c) for c in result["components"])


@pytest.mark.parametrize("style", ["pinch", "tripod", "power"])
def test_actual_teacher_modules_are_present_without_granting_readiness(current_profile,style):
    result=api().static_preflight(config(style),project=ROOT,backend_profile=current_profile)
    names={item["name"]:item for item in result["components"]}
    for name in ("model","context"):
        item=names[name]
        assert item["valid"] is True and item["source_complete"] is True
        assert item["blocked_reason"] is None
        assert item["source_sha256"] == api().sha256_file(item["reader"])
    assert result["static_ready"] is False
    assert result["live_reset_verified"] is False and result["new_native_steps"] == 0
    assert names["same_clock_distance"]["valid"] is (style == "pinch")


@pytest.mark.parametrize("field,value", [
    ("model_schema", "dh116-actual-backend-measured-teacher"),
    ("physics_hz", 60), ("control_hz", 480), ("action_dim", 23),
    ("r2_enabled", True), ("force_proxy_enabled", True),
    ("original_guards_enabled", False), ("style", "other"), ("native_ready", True),
])
def test_malformed_contract_rejected_before_source_access(field, value):
    c = config()
    c[field] = value
    with pytest.raises(ValueError, match="schema|contract|unknown|style"):
        api().validate_config(c)


def test_budget_requires_separate_actual_authorization(tmp_path):
    m = api()
    c = config()
    c["budget"] = dict(max_sdk_steps=258, max_resets=1, max_solver_calls=3072,
                       max_episode_count=1, wall_timeout_s=180)
    with pytest.raises(ValueError, match="authorization"):
        m.validate_budget(c, project=ROOT)
    c["authorization"] = True
    with pytest.raises(ValueError, match="authorization"):
        m.validate_budget(c, project=ROOT)


@pytest.mark.parametrize("key,bad", [("max_sdk_steps", True), ("max_resets", 0),
    ("max_solver_calls", float("inf")), ("wall_timeout_s", -1)])
def test_budget_numbers_are_finite_explicit_and_positive(key, bad):
    c = config()
    c["budget"] = dict(max_sdk_steps=258, max_resets=1, max_solver_calls=3072,
                       max_episode_count=1, wall_timeout_s=180)
    c["budget"][key] = bad
    with pytest.raises(ValueError, match="budget"):
        api().validate_budget(c, project=ROOT)


def test_manifest_needs_original_source_and_no_pose_preview(tmp_path):
    m = api()
    c = config()
    manifest = dict(schema=m.RESET_MANIFEST_SCHEMA, style="pinch", pose_preview_only=True)
    path = tmp_path / "reset.json"
    path.write_text(json.dumps(manifest))
    c["resolved_reset_manifest"] = dict(path=str(path), sha256=m.sha256_file(path))
    with pytest.raises(ValueError, match="resolved|preview|source"):
        m.load_reset_manifest(c, project=ROOT)


def test_gamma_source_mismatch_rejected():
    class Reward:
        gamma = .999
    with pytest.raises(ValueError, match="gamma"):
        api().validate_reward_gamma(Reward(), {"gamma": .995})


def test_live_boolean_cannot_bypass(current_profile):
    with pytest.raises((TypeError, ValueError), match="owner|contract|argument"):
        api().live_preflight(True, True, backend_profile=current_profile, manifest={})


def test_cpu_numeric_live_reader_round_trip_and_cross_reset_rejected(case):
    # Real unchanged lower numeric validator with CAI CPU protocol doubles.
    # All qualification remains false; only reader/contract consistency is tested.
    from test_mesh_reset_numeric_state import execute
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    m = api()
    saved = []
    receipt = execute(case, saved.append)
    receipt["mesh_state_readback"] = dict(before={"source_instance_id": id(case.rope)},
                                         after={"source_instance_id": id(case.rope)})
    with source.source_scope():
        verified = m.validate_live_numeric(saved[0], reset_receipt=receipt,
                                          rope=case.rope, expected_clock=receipt["after_clock"])
    assert verified["source_instance_id"] == id(case.rope)
    changed = deepcopy(receipt)
    changed["mesh_reset"]["before_clock"]["rod"]["reset_id"] += 1
    with source.source_scope(), pytest.raises(ValueError):
        m.validate_live_numeric(saved[0], reset_receipt=changed,
                                rope=case.rope, expected_clock=receipt["after_clock"])


from test_mesh_reset_numeric_state import case


def test_cli_exclusive_output_and_exit_codes(tmp_path):
    m = api()
    output = tmp_path / "preflight.json"
    script = ROOT / "scripts/preflight_dh116_native_teacher.py"
    result = subprocess.run([sys.executable, "-B", str(script), "--style", "pinch",
                             "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 3, result.stderr
    assert json.loads(output.read_text())["new_native_steps"] == 0
    again = subprocess.run([sys.executable, "-B", str(script), "--style", "pinch",
                            "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert again.returncode == 2
    assert output.exists()


@pytest.mark.parametrize("damage", ["none", "current", "target", "clock", "owner"])
def test_real_cpu_mesh_reader_verifies_live_same_owner(monkeypatch, damage):
    from test_mesh_observation_profile import setup_readback, execute
    c, plan, reader = setup_readback(monkeypatch)
    receipt = execute(c, plan, reader)
    before_steps = c.rope.steps
    if damage in ("current", "target"):
        mapping = c.rope.current if damage == "current" else c.rope.targets
        mapping[0][0][0] += .001
    elif damage == "clock":
        c.sim.current_time_step_index += 1
        c.sim.current_time += 1/480
    elif damage == "owner":
        receipt["mesh_state_readback"]["after"]["source_instance_id"] += 1
    if damage == "none":
        result = api().validate_live_mesh(c.sim, c.rod, reset_receipt=receipt, mesh_plan=plan)
        assert result["eleven_reset_meshes_observed"] is True
    else:
        with pytest.raises(ValueError):
            api().validate_live_mesh(c.sim, c.rod, reset_receipt=receipt, mesh_plan=plan)
    assert c.rope.steps == before_steps


@pytest.mark.parametrize("damage", ["none", "sdk_q", "sdk_target", "clock", "reset_uid"])
def test_real_cpu_mounted_reader_reopens_current_sdk(damage):
    from test_dh116_teacher_mounted_wrist_source import hand, capture, clock
    from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import adapt_mounted_hands
    from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import HandBinding
    from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
    hands = [hand(s) for s in ("left", "right")]
    packets = [capture(h, side=s) for h, s in zip(hands, ("left", "right"))]
    observation = adapt_mounted_hands(packets, expected_identity=clock())
    bindings = tuple(HandBinding(s, h, {8: "/World/envs/env_8/"+s+"Hand"}, MountView(h))
                     for h, s in zip(hands, ("left", "right")))
    actual = clock()
    if damage == "sdk_q":
        hands[0].sdk_q[0, 0] += 1
    elif damage == "sdk_target":
        # Current body pose cannot be substituted by an old saved source packet.
        hands[0].sdk_pose[0, 1, 0] += 1
    elif damage == "clock":
        actual["timestamp_s"] += .01
    elif damage == "reset_uid":
        actual["reset_id"] += 1
    if damage == "none":
        result = api().validate_live_mounted(bindings, observation, read_identity=lambda: actual)
        assert result["valid_channels"] == 66
    else:
        with pytest.raises(ValueError):
            api().validate_live_mounted(bindings, observation, read_identity=lambda: actual)


@pytest.mark.parametrize("damage", ["none", "seed", "rod", "receipt_sha"])
def test_retained_manifest_is_derived_from_actual_small_raw_artifacts(tmp_path, damage):
    m = api()
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, encode_owned
    original_path = m.EVIDENCE / (
        "native-attempts/flex.f8-pinch-fast-evidence-native/"
        "f0b5bc8cd80210db27052f4f96f435c7ce48a7ff23d53fd533a53fd3ed63502f/attempt-2ynpy5si/receipt.json")
    original = json.loads(m.checked_file(original_path, m.ORIGINAL_RECEIPT_SHA))
    output = Path(original["output"])
    reset_path, numeric_path = output / "mesh_reset_state.json", output / "mesh_reset_numeric_state.json"
    saved = decode_owned(json.loads(reset_path.read_bytes()))["readback"]
    receipt = saved["completed_reset_receipt"]
    numeric = decode_owned(json.loads(numeric_path.read_bytes()))
    value = dict(schema=m.RESET_MANIFEST_SCHEMA, style="pinch", pose_preview_only=False,
        target_schema=m.TARGET_SCHEMA_VERSION, target_material_ids=list(range(41)),
        source_refs={
            "original_receipt": dict(path=str(original_path), sha256=m.ORIGINAL_RECEIPT_SHA),
            "reset_readback": dict(path=str(reset_path), sha256=original["artifact_sha256"][reset_path.name]),
            "numeric_reset": dict(path=str(numeric_path), sha256=original["artifact_sha256"][numeric_path.name])},
        reset_seed=receipt["seed"],
        rod_state={k: numeric["samples"][1]["state"][k].tolist()
                   for k in ("positions", "velocities", "quaternions")},
        mesh_bindings_sha256=m.digest(encode_owned(receipt["rod_reset"]["mesh_bindings"])))
    if damage == "seed":
        value["reset_seed"]["source_description"] = "invented"
    elif damage == "rod":
        value["rod_state"]["positions"][0][0][0] += .01
    elif damage == "receipt_sha":
        value["source_refs"]["original_receipt"]["sha256"] = "0"*64
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(value))
    c = config()
    c["resolved_reset_manifest"] = dict(path=str(path), sha256=m.sha256_file(path))
    if damage == "none":
        manifest = m.load_reset_manifest(c, project=ROOT)
        assert manifest.data()["source_refs"]["original_receipt"]["sha256"] == m.ORIGINAL_RECEIPT_SHA
        first = manifest.data()
        first["rod_state"]["positions"][0][0][0] = 999
        assert manifest.data()["rod_state"]["positions"][0][0][0] != 999
    else:
        with pytest.raises(ValueError):
            m.load_reset_manifest(c, project=ROOT)


def test_cli_malformed_schema_exits_two_and_writes_nothing(tmp_path):
    path, output = tmp_path / "bad.json", tmp_path / "out.json"
    c = config()
    c["authorization"] = True
    path.write_text(json.dumps(c))
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/preflight_dh116_native_teacher.py"),
        "--config", str(path), "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert not output.exists()


@pytest.mark.parametrize("held", [False, True])
def test_budget_reads_real_cpu_kernel_lock_and_corresponding_files(tmp_path, monkeypatch, held):
    # Explicit temporary CPU lock fixture, never the actual native singleton.
    import os
    from scripts import run_dh116_work_packages as scheduler
    m = api()
    c = config()
    c["budget"] = dict(max_sdk_steps=258, max_resets=1, max_solver_calls=3072,
                       max_episode_count=1, wall_timeout_s=180)
    auth = dict(schema="dh116-native-teacher-budget-authorization-v1",
        scope="new_native_teacher_ppo", profile=c["profile"], style=c["style"], budget=c["budget"],
        max_ppo_updates=1, config_sha256=m.authorization_config_digest(c),
        session_uid="cpu_fixture_only", owner_pid=os.getpid())
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(json.dumps(auth))
    c["authorization"] = dict(path=str(auth_path), sha256=m.sha256_file(auth_path))
    lock_path = tmp_path / "reservation.json"
    lock_path.write_text(json.dumps(dict(schema="dh116-native-resource-reservation-v1",
        session_uid=auth["session_uid"], resource_class="isaac_native_singleton",
        authorization_sha256=c["authorization"]["sha256"], pid=os.getpid())))
    c["resource_lock"] = dict(path=str(lock_path), sha256=m.sha256_file(lock_path))
    kernel_path = tmp_path / "cpu-fixture.lock"
    kernel_path.touch()
    monkeypatch.setattr(scheduler, "NATIVE_LOCK", kernel_path)
    if held:
        with scheduler.native_file_lock(kernel_path):
            assert m.validate_budget(c, project=ROOT) == c["budget"]
    else:
        with pytest.raises(ValueError, match="kernel.*lock"):
            m.validate_budget(c, project=ROOT)



def test_manifest_identity_requires_source_loader():
    with pytest.raises(TypeError, match="source"):
        api().ResolvedResetManifest("forged", "0"*64, b"{}")


@pytest.fixture(scope="module")
def retained_distance_case():
    """Real CPU reader construction and small pinned artifacts; no physical stream."""
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116.tabletop_bilateral_r0_approach import BilateralMeasuredPinchApproach
    m = api()
    path = m.EVIDENCE / (
        "native-attempts/flex.f8-pinch-fast-evidence-native/"
        "f0b5bc8cd80210db27052f4f96f435c7ce48a7ff23d53fd533a53fd3ed63502f/attempt-2ynpy5si/receipt.json")
    original = json.loads(m.checked_file(path, m.ORIGINAL_RECEIPT_SHA))
    output = Path(original["output"])
    def artifact(name):
        return decode_owned(json.loads(m.checked_file(output / name, original["artifact_sha256"][name])))
    reset = artifact("mesh_reset_state.json")["readback"]["completed_reset_receipt"]
    parameters = artifact("director_reset.json")["payload"]
    assert parameters["reward_radius_sample"]["source"]["backend_sha256"] == (
        "8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340")
    return BilateralMeasuredPinchApproach(ROOT), reset, parameters


def profile_reader_scope(profile):
    module = importlib.import_module("go2_pvcnn.control.dh116.tabletop_native_teacher_profile")
    assert hasattr(module, "profile_reader_scope"), "C01 verified profile reader scope missing"
    return module.profile_reader_scope(profile)


def test_current_profile_complete_reset_distance_bridge(current_profile, retained_distance_case):
    reader, reset, packet = retained_distance_case
    with profile_reader_scope(current_profile):
        result = reader.capture_reset(reset, packet, grasp="pinch")
    assert result["missing_reason"] is None
    assert result["reset_baseline_available"] is True
    assert result["auxiliary"]["identity"]["physics_step"] == 2
    values = result["reset_approach"]["approach_distance_m"]["value"]
    assert len(values) == 2 and all(v > 0 for v in values)
    assert result["source"]["radius_provenance"] == packet["reward_radius_sample"]
    assert result["source"]["boundary"] == "actual_reset_readback"
    assert result["source"]["geometry_contract_version"] == "dh116-bilateral-owned-geometry-v1"
    assert result["new_physics_steps"] == result["ppo_updates"] == 0
    assert result["native_qualification"] is result["teacher_admission"] is False
    assert result["r0_native_reset_complete"] is False


@pytest.mark.parametrize("backend", ["0" * 64,
    "2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"])
def test_current_profile_distance_rejects_foreign_backend(current_profile, retained_distance_case, backend):
    reader, reset, packet = retained_distance_case
    packet = deepcopy(packet)
    packet["reward_radius_sample"]["source"]["backend_sha256"] = backend
    with profile_reader_scope(current_profile), pytest.raises(ValueError, match="radius/provenance"):
        reader.capture_reset(reset, packet, grasp="pinch")


@pytest.mark.parametrize("exit_kind", ["normal", "exception", "nested"])
def test_profile_reader_scope_restores_all_bindings(current_profile, exit_kind):
    from scripts import replay_dh116_native_parameter_r0 as radius
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    bindings = source._ORIGINAL + ((radius, "BACKEND_SHA", source.OLD_BACKEND_SHA),)
    before = tuple(getattr(module, name) for module, name, _ in bindings)
    def check():
        assert radius.BACKEND_SHA == current_profile.binary_sha256
        assert all(getattr(module, name) == value for module, name, value in source._BINDINGS)
    try:
        with profile_reader_scope(current_profile):
            check()
            if exit_kind == "nested":
                with profile_reader_scope(current_profile):
                    check()
                check()
            if exit_kind == "exception":
                raise RuntimeError("CPU fixture exception")
    except RuntimeError:
        if exit_kind != "exception":
            raise
    assert before == tuple(getattr(module, name) for module, name, _ in bindings)
    assert radius.BACKEND_SHA == source.OLD_BACKEND_SHA


def test_profile_reader_scope_composes_with_existing_scope(current_profile):
    from scripts import replay_dh116_native_parameter_r0 as radius
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    with source.source_scope():
        with profile_reader_scope(current_profile):
            assert radius.BACKEND_SHA == current_profile.binary_sha256
        assert radius.BACKEND_SHA == source.OLD_BACKEND_SHA
        assert all(getattr(module, name) == value for module, name, value in source._BINDINGS)


def test_profile_reader_scope_rejects_foreign_alias_without_mutation(current_profile, monkeypatch):
    from scripts import replay_dh116_native_parameter_r0 as radius
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    before = tuple(getattr(module, name) for module, name, _ in source._ORIGINAL)
    monkeypatch.setattr(radius, "BACKEND_SHA", "foreign")
    with pytest.raises(ValueError, match="binding"):
        with profile_reader_scope(current_profile):
            pytest.fail("foreign binding entered")
    assert radius.BACKEND_SHA == "foreign"
    assert before == tuple(getattr(module, name) for module, name, _ in source._ORIGINAL)


def test_profile_reader_scope_requires_loaded_profile():
    with pytest.raises((ValueError, TypeError), match="profile|Profile"):
        with profile_reader_scope("8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340"):
            pytest.fail("caller SHA entered")


@pytest.mark.parametrize("target_kind", ["binary", "radius_reader"])
def test_profile_reader_scope_rechecks_actual_bytes_before_binding(current_profile, monkeypatch, target_kind):
    from scripts import replay_dh116_native_parameter_r0 as radius
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    target = (Path(current_profile.binary_path) if target_kind == "binary"
              else ROOT / "scripts/replay_dh116_native_parameter_r0.py")
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes",
        lambda path: original(path) + (b"tamper" if path == target else b""))
    with pytest.raises(ValueError, match="SHA|source"):
        with profile_reader_scope(current_profile):
            pytest.fail("tampered source entered")
    assert radius.BACKEND_SHA == source.OLD_BACKEND_SHA
    assert all(getattr(module, name) == value for module, name, value in source._ORIGINAL)

@pytest.mark.parametrize("digest_kind,sha_kind,expected",[
    ("callable","callable",None),("bool","short",None),
    ("short","valid_sha","b"*64),("valid_digest","callable","a"*64),
])
def test_module_source_identity_requires_full_string_sha(current_profile,monkeypatch,digest_kind,sha_kind,expected):
    # Module imports may expose digest()/sha256() functions. Those are never
    # source identities, and metadata extraction must never invoke them.
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_model as model
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_context as context
    calls=[]
    def forbidden():
        calls.append(True)
        raise AssertionError("metadata functions must not be called")
    samples={"callable":forbidden,"bool":True,"short":"a"*63,
        "valid_sha":"b"*64,"valid_digest":"a"*64}
    for module in (model,context):
        monkeypatch.setattr(module,"digest",samples[digest_kind],raising=False)
        monkeypatch.setattr(module,"sha256",samples[sha_kind],raising=False)
    result=api().static_preflight(config(),project=ROOT,backend_profile=current_profile)
    for component in result["components"]:
        if component["name"] in ("model","context"):
            assert component["source_sha256"] == (expected or api().sha256_file(component["reader"]))
    assert not calls
    json.dumps(result,allow_nan=False)
