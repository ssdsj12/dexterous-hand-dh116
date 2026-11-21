"""Factory-order doubles only; no physical or native qualification."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest


def mod():
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_session
    return tabletop_native_teacher_session


def arranged(monkeypatch):
    m=mod(); events=[]
    contract=SimpleNamespace(reset_seed={},ppo_config=object())
    live=SimpleNamespace(model_contract=object(),reward_contract=object())
    owner=Mock();owner.episode_uid="cpu-isolated:one";owner.owner_resets={}
    owner.reset.side_effect=lambda *a,**k: events.append("reset") or {}
    monkeypatch.setattr(m,"verify_static_contract",lambda c: contract)
    monkeypatch.setattr(m,"verify_reset_and_sources",lambda *a: events.append("verify_live") or live)
    monkeypatch.setattr(m,"NativeTeacherCollector",lambda *a: "constructed")
    fs=m.NativeFactories(episode=lambda c: events.append("owner") or owner,
        policy=lambda c: events.append("policy") or object(),
        trainer=lambda p,c: events.append("trainer") or object(),
        reward=lambda c: events.append("reward") or object())
    return m,events,owner,fs


def test_factory_order_after_actual_reset(monkeypatch):
    m,e,o,f=arranged(monkeypatch)
    assert m.create_native_collector({},f)=="constructed"
    assert e==["owner","reset","verify_live","policy","trainer","reward"]
    o.reset.assert_called_once_with({},episode_uid=o.episode_uid,owner_resets=o.owner_resets)
    o.close.assert_not_called()


@pytest.mark.parametrize("stage",["static","reset","live","policy","trainer","reward","collector"])
def test_every_failed_stage_preserves_original_and_closes_once(monkeypatch,stage):
    from dataclasses import replace
    m,e,o,f=arranged(monkeypatch);original=ValueError("original-"+stage)
    def fail(*a,**k): raise original
    if stage=="static": monkeypatch.setattr(m,"verify_static_contract",fail)
    elif stage=="reset": o.reset.side_effect=fail
    elif stage=="live": monkeypatch.setattr(m,"verify_reset_and_sources",fail)
    elif stage=="collector": monkeypatch.setattr(m,"NativeTeacherCollector",fail)
    else: f=replace(f,**{stage:fail})
    with pytest.raises(ValueError) as exc: m.create_native_collector({},f)
    assert exc.value is original
    assert o.close.call_count==(0 if stage=="static" else 1)
    if stage=="live": assert "policy" not in e
    if stage=="static": assert not e


def test_cleanup_failure_is_durable_without_masking_primary(monkeypatch):
    m,e,o,f=arranged(monkeypatch);original=ValueError("live");cleanup=RuntimeError("close")
    def fail(*a): raise original
    monkeypatch.setattr(m,"verify_reset_and_sources",fail);o.close.side_effect=cleanup
    with pytest.raises(ValueError) as exc: m.create_native_collector({},f)
    assert exc.value is original
    o.record_cleanup_failure.assert_called_once_with(cleanup)


def test_actual_api_forwards_new_explicit_config_without_breaking_keywords(monkeypatch,tmp_path):
    m=mod()
    from go2_pvcnn.control.dh116 import tabletop_no_camera_collector as old
    marker=object();config={"style":"pinch"}
    monkeypatch.setattr(m,"create_native_collector",lambda c,f: (c,f))
    p=lambda c:marker;t=lambda p,c:marker;e=lambda c:marker
    actual,fs=old.create_actual_collector(tmp_path,"pinch",policy_factory=p,
        trainer_factory=t,episode_factory=e,config=config)
    assert actual is config and fs.policy is p and fs.trainer is t and fs.episode is e


def test_owner_close_disarms_callbacks_once_and_never_closes_outer_app(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    m=mod();owner=object.__new__(m.NativeEpisodeOwner)
    owner._closed=False;events=[];owner._restores=[lambda:events.append("restore")]
    owner.ledger=NativeJournal(tmp_path/"journal.jsonl")
    owner.app=Mock();owner.sim=Mock();owner.rod=Mock()
    sim,rod=owner.sim,owner.rod
    owner.close();owner.close()
    assert events==["restore"]
    owner.app.close.assert_not_called();sim.stop.assert_not_called();rod.close.assert_not_called()
    assert owner.sim is owner.rod is owner.episode is None
    owner.record_cleanup_failure(ValueError("retained cleanup"))
    assert "retained cleanup" in owner.ledger.path.read_text()


def test_default_actual_admission_uses_same_static_contract(monkeypatch,tmp_path):
    from go2_pvcnn.control.dh116 import tabletop_no_camera_collector as old
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as pre
    calls=[]
    def static(config,**kw):
        calls.append(config)
        return dict(static_ready=False,blocked_reasons=["resolved_reset_manifest_source_missing"],
            live_reset_verified=False,new_native_steps=0,native_qualification=False)
    monkeypatch.setattr(pre,"static_preflight",static)
    result=old.actual_admission(tmp_path,"pinch")
    assert result["admitted"] is False and len(calls)==1
    assert result["blocked_reasons"]==["resolved_reset_manifest_source_missing"]


def test_cleanup_diagnostic_failure_cannot_mask_original_error(monkeypatch):
    m,e,o,f=arranged(monkeypatch);original=ValueError("original source")
    def fail(*a):raise original
    monkeypatch.setattr(m,"verify_reset_and_sources",fail)
    o.close.side_effect=RuntimeError("cleanup")
    o.record_cleanup_failure.side_effect=OSError("diagnostic disk failed")
    with pytest.raises(ValueError) as exc:m.create_native_collector({},f)
    assert exc.value is original


def test_shared_budget_keeps_setup_solver_calls_and_internal_substeps_separate(monkeypatch):
    m=mod();limits=dict(max_sdk_steps=29,max_solver_calls=27,max_resets=3,max_episode_count=3,wall_timeout_s=3.)
    budget=m.SessionBudget(limits)
    owner=SimpleNamespace(sim=SimpleNamespace(current_time_step_index=2,current_time=2/480),rod=SimpleNamespace(rope=object()))
    budget.record_setup(owner,[dict(sdk_step=0,sdk_time_s=0.),dict(sdk_step=2,sdk_time_s=2/480)])
    budget.reserve_reset(owner)
    for step in range(3,11):budget.charge(dict(rod=dict(physics_step=step)),12)
    assert budget.sdk_steps==10 and budget.solver_calls==8 and budget.internal_solver_substeps==96
    before=budget.started;budget.record_setup(owner,[]);budget.reserve_reset(owner)
    assert budget.started==before and budget.sdk_steps==10
    with pytest.raises(ValueError,match="replayed"):budget.record_setup(owner,[dict(sdk_step=0,sdk_time_s=0.)])
    budget.reserve_reset(owner)
    with pytest.raises(RuntimeError,match="budget"):budget.reserve_reset(owner)


def test_legacy_cpu_implementation_ast_is_identical_to_approved_base():
    import ast,hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    names={'ENDPOINT', 'CPUFixtureCollector', 'VERSION', 'EPOCH', 'TASK', '_ObserveTerminal', '_model_weights', 'SCOPE', 'ExclusiveJournal', '_origin', '_endpoint'}
    nodes=[]
    for n in ast.parse((root/"src/go2_pvcnn/control/dh116/tabletop_no_camera_collector.py").read_text()).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:nodes.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets):nodes.append(n)
    digest=hashlib.sha256("\n".join(ast.dump(n,include_attributes=False) for n in nodes).encode()).hexdigest()
    # Frozen AST from integration a6a429, not recomputed from candidate bytes.
    assert digest=="900f04d64dda22198b89a9ecb5013adebd1fc32e3185bb560f2bf176a753fd40"


@pytest.mark.parametrize("target",["src/go2_pvcnn/control/dh116/tabletop_no_camera_collector.py",
    "tests/test_dh116_no_camera_collector.py"])
def test_current_cpu_source_lock_rejects_tampering_without_global_patch(tmp_path,target):
    from pathlib import Path
    from go2_pvcnn.control.dh116.tabletop_cpu_fixture_owners import CPUFixtureOwners,SOURCE_SHA256
    root=Path(__file__).resolve().parents[1]
    for name in SOURCE_SHA256:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((root/name).read_bytes())
    path=tmp_path/target;path.write_bytes(path.read_bytes()+b"\n# changed bytes\n")
    owner=object.__new__(CPUFixtureOwners);owner.project=tmp_path
    with pytest.raises(ValueError,match="source SHA differs"):
        owner._classes()


def test_approved_reward_ppo_parameters_are_mapped_without_old_trainer_defaults(monkeypatch,tmp_path):
    import hashlib
    from pathlib import Path
    from test_dh116_flexible_measured_common_teacher import setup
    from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import candidate_config
    m=mod();root=Path(__file__).resolve().parents[1]
    raw=root/"configs/dh116_training/approach_wiring_smoke_v1.json"
    pre=candidate_config();pre["reward_config"]=dict(path=str(raw),sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    limits=dict(max_sdk_steps=29,max_solver_calls=27,max_resets=3,max_episode_count=3,wall_timeout_s=3.)
    # Only source/manifest/budget verifiers are isolated here. No native claim.
    monkeypatch.setattr(m,"_static_preflight_with_manifest",lambda *a,**k:(
        dict(static_ready=True),SimpleNamespace(data=lambda:dict(reset_seed={}))))
    monkeypatch.setattr(m,"validate_budget",lambda *a,**k:limits)
    model_config,*_=setup("pinch")
    config=m.NativeSessionConfig(pre,model_config,tmp_path/"journal",2.,32,root)
    contract=m.verify_static_contract(config)
    actual=contract.ppo_config
    assert actual.gamma==.995 and actual.lr==.0001 and actual.epochs==actual.minibatches==1
    assert actual.lam==.95 and actual.clip==.2 and actual.target_kl==.02 and actual.kl_backtrack_steps==3
    pre["ppo"]["epochs"]=5
    with pytest.raises(ValueError,match="numeric config differs"):m.verify_static_contract(config)


def test_actual_adapter_does_not_turn_valid_original_guard_into_source_failure(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import NativePinchSources
    from go2_pvcnn.control.dh116 import tabletop_deformable_clearance_guard as clearance
    # Isolate native capture here to prove the adapter classification itself.
    # The integration tests separately bind real CPU physical rows/history/SDF.
    source=object.__new__(NativePinchSources);source._error=None
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile,ROOT
    source.contract=SimpleNamespace(backend_profile=BackendProfile.load(ROOT))
    source.distance=SimpleNamespace(capture=lambda *a,**k:{"auxiliary":{}})
    source.owner=SimpleNamespace(history=object(),ledger=Mock(),rod=SimpleNamespace(rope=object()))
    source.parameters=dict(source_instance_id=id(source.owner.rod.rope))
    expected=object();source._endpoint=lambda *a:expected
    monkeypatch.setattr(clearance,"validate_post_step_clearance",lambda raw:dict(terminal_reason="right:joint_limit"))
    event=dict(exchange=dict(original_guard_failure="right:joint_limit",guard_failure="right:joint_limit",
        closure_geometry=dict(passed=False,source_complete=True,failure_reasons=["original:right:joint_limit"]),
        native_teacher_distance={"auxiliary":{}},material_director=source.parameters,rod={},native_teacher_health={}))
    assert source.observe(event) is expected


def test_mesh_reset_limit_is_explicit_and_never_fakes_a_late_sdk_clock():
    m=mod();owner=object.__new__(m.NativeEpisodeOwner)
    owner.sim=SimpleNamespace(current_time_step_index=240)
    result=m.reset_clock_preflight(owner)
    assert result["actual_reset_sdk_step"]==240 and result["max_reset_sdk_step"]==240
    assert result["long_term_reset_supported"] is False and result["policy_interval_sdk_cap"] is None
    owner.sim.current_time_step_index=241
    with pytest.raises(ValueError,match="bounded_mesh_reset_clock_v2_source_missing"):
        m.reset_clock_preflight(owner)
    assert owner.sim.current_time_step_index==241


def test_late_actual_reset_is_denied_before_mutation_or_budget_reservation():
    m=mod();owner=object.__new__(m.NativeEpisodeOwner)
    owner._closed=False;owner.sim=SimpleNamespace(current_time_step_index=258)
    owner.ledger=Mock();owner.contract=SimpleNamespace(runtime=Mock())
    with pytest.raises(ValueError,match="bounded_mesh_reset_clock_v2_source_missing"):
        owner.reset({},episode_uid="cpu-late",owner_resets={})
    owner.contract.runtime.reserve_reset.assert_not_called()
    assert owner.sim.current_time_step_index==258



@pytest.mark.parametrize("failure", [
    "parameter_constructor", "record_setup", "setup_ledger", "cleanup_diagnostic"])
def test_constructor_failure_restores_actual_episode_callback_and_drops_borrowed_refs(monkeypatch, tmp_path, failure):
    """Real owner and reader ctor with original CPU SDK/rod fixtures; native=0."""
    import numpy as np
    from pathlib import Path
    from test_dh116_dual_hand_episode import setup
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, ROOT
    m = mod()
    ep, rod, hands, sim, seed, owners, rows = setup(monkeypatch, tmp_path)
    original = ep._record
    original_owners = dict(owners)
    limits = dict(max_sdk_steps=1 if failure == "record_setup" else 100,
        max_solver_calls=100, max_resets=2, max_episode_count=2, wall_timeout_s=1000.)
    nodes = np.zeros((1, 41, 3), np.float32)
    nodes[0, :, 0] = np.arange(41) * .005
    contract = SimpleNamespace(
        config=SimpleNamespace(preflight=dict(style="pinch", profile="approach_wiring_smoke_v1"), project=ROOT),
        backend_profile=BackendProfile.load(ROOT), runtime=m.SessionBudget(limits),
        reset_manifest=SimpleNamespace(data=lambda: dict(rod_state=dict(positions=nodes))))
    ledger = NativeJournal(tmp_path / "constructor.jsonl", scope="cpu_fixture")
    emit = ledger.emit
    primary = OSError("setup ledger failed")
    cleanup = OSError("cleanup diagnostic failed")
    def recording(operation, payload):
        if failure == "setup_ledger" and operation == "setup":
            raise primary
        if failure == "cleanup_diagnostic" and operation in ("owner_lease_closed", "cleanup_failure"):
            raise cleanup
        return emit(operation, payload)
    monkeypatch.setattr(ledger, "emit", recording)
    closed = []
    close = m.NativeEpisodeOwner.close
    def capture_close(self):
        closed.append(self)
        return close(self)
    monkeypatch.setattr(m.NativeEpisodeOwner, "close", capture_close)
    monkeypatch.setattr(m, "verify_static_contract", lambda config: contract)
    def factory(actual):
        return m.NativeEpisodeOwner(ep, actual, episode_uid="cpu-constructor-failure",
            owner_resets=owners, mesh_plan=None, ledger=ledger,
            setup_events=[dict(kind="setup", sdk_step=0, sdk_time_s=0.),
                dict(kind="setup", sdk_step=sim.current_time_step_index, sdk_time_s=sim.current_time)],
            initial_geometry=lambda: ())
    def never(*args):
        raise AssertionError("no Reset/model/trainer/reward operation after constructor failure")
    try:
        error_type = OSError if failure == "setup_ledger" else RuntimeError if failure == "record_setup" else ValueError
        message = "setup ledger failed" if failure == "setup_ledger" else "scene setup exceeds SDK budget" if failure == "record_setup" else "explicit CUDA device required"
        with pytest.raises(error_type, match=message) as caught:
            m.create_native_collector(None, m.NativeFactories(factory, never, never, never))
        if failure == "setup_ledger":
            assert caught.value is primary
        # In the source failure case this is the original reader constructor,
        # whose explicit string validation correctly rejects a CPU SDK device.
        assert ep._record is original
        assert len(closed) == 1
        owner = closed[0]
        assert owner._closed and owner._restores == []
        assert owner.episode is owner.sim is owner.rod is owner.bindings is None
        assert owner._initial_geometry is None and owner.owner_resets == {}
        assert owners == original_owners  # drop our closure refs; do not mutate caller owners
        assert contract.runtime.sdk_steps == contract.runtime.setup_sdk_steps == 2
        assert contract.runtime.solver_calls == contract.runtime.resets == contract.runtime.episodes == 0
        assert rod.rope.steps == 0 and sim.current_time_step_index == 2 and ep._state == "new"
        close(owner)  # already closed: no extra lease cleanup/logging
        if failure == "cleanup_diagnostic":
            fallback = ledger.path.with_suffix(".cleanup.jsonl").read_text()
            assert "cleanup diagnostic failed" in fallback
    finally:
        ledger.close()

@pytest.mark.parametrize("derive_mesh", [False, True])
def test_fresh_pinch_factory_hands_one_journal_and_profile_to_original_owner(monkeypatch, tmp_path, derive_mesh):
    from contextlib import contextmanager
    from pathlib import Path
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_profile as profile
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal

    m = mod()
    events = []
    marker = object()
    setup_events = tuple(dict(kind="setup", sdk_step=t, sdk_time_s=t/480) for t in (0, 2))
    rod = object()
    episode = SimpleNamespace(_rod=rod)
    fresh = builder.FreshPinchEpisode(episode, setup_events, lambda: ())
    prepared = object()
    if derive_mesh:
        from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
        from go2_pvcnn.control.dh116 import tabletop_mesh_reset_order as reset
        import sys

        def prepare(project):
            assert project == config.project
            events.append("prepare")
            return prepared

        def build_mesh(project, current_scene, current_rod, *, sdf_module):
            assert project == config.project
            assert current_scene is prepared and current_rod is rod and sdf_module is marker
            events.append("mesh")
            return marker

        monkeypatch.setattr(scene, "prepare_scene", prepare)
        monkeypatch.setattr(reset, "prepare_mesh_reset_plan", build_mesh)
        monkeypatch.setitem(sys.modules, "ropesim.sdf", marker)
    config = m.NativeSessionConfig(dict(style="pinch", profile="approach_wiring_smoke_v1"),
                                   {}, tmp_path/"training.jsonl", 1., 32)
    profile_value = SimpleNamespace(binary_path="/path/to/locked/backend", binary_sha256="e"*64) if derive_mesh else marker
    contract = m.StaticContract(config, profile_value, None, {}, None, None, {"max_sdk_steps":258}, None, "")
    trace = Mock()

    @contextmanager
    def opened(value):
        assert value is profile_value
        events.append("profile")
        yield
        events.append("profile_return")

    def build(sim, owners, **kwargs):
        assert sim is owners is marker
        assert kwargs["physics_budget_end"] == 258
        assert kwargs["expected_reset_seed"] is contract.reset_seed
        if derive_mesh:
            assert kwargs["prepared"] is prepared
            assert builder.BACKEND == Path(profile_value.binary_path)
            assert builder.BACKEND_SHA == profile_value.binary_sha256
        for row in setup_events:
            kwargs["record"](row)
        events.append("build")
        return fresh

    def owner(current, source, **kwargs):
        assert current is episode and source is contract
        assert kwargs["mesh_plan"] is marker
        assert kwargs["ledger"] is journal and kwargs["setup_events"] == setup_events
        assert kwargs["initial_geometry"] is fresh.initial_geometry
        events.append("owner")
        return marker

    monkeypatch.setattr(profile, "profile_reader_scope", opened)
    monkeypatch.setattr(builder, "build_fresh_owned", build)
    monkeypatch.setattr(m, "NativeEpisodeOwner", owner)
    journal = NativeJournal(tmp_path/"training.jsonl")
    try:
        result = m.build_fresh_pinch_owner(
            contract, sim=marker, owners=marker, prepared=None, ownership=trace, ledger=journal,
            run_uuid="source", episode_uid="episode", owner_resets={}, mesh_plan=None if derive_mesh else marker)
        assert result is marker
        if derive_mesh:
            assert events == ["profile", "prepare", "build", "mesh", "owner", "profile_return"]
        else:
            assert events == ["profile", "build", "owner", "profile_return"]
        assert trace.event.call_count == 2
        assert journal.path.read_text() == ""
    finally:
        journal.close()




def test_training_alias_scope_only_applies_to_exact_pinch_receipt(monkeypatch, tmp_path):
    import hashlib
    from scripts import prepare_dh116_pinch_margin_contact as legacy
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import _canonical_pinch_history

    canonical = tmp_path / "real"
    canonical.mkdir()
    original = canonical / "receipt.json"
    original.write_text("{}")
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(canonical, target_is_directory=True)
    alias = alias_dir / original.name
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    monkeypatch.setattr(legacy, "OLD_PAIR", alias)
    monkeypatch.setattr(legacy, "OLD_PAIR_SHA", digest)
    read_locked = legacy.read_locked
    with pytest.raises(ValueError, match="locked evidence changed"):
        read_locked(alias, digest)
    with _canonical_pinch_history():
        assert legacy.read_locked(alias, digest) == {}
        with pytest.raises(ValueError, match="locked evidence changed"):
            legacy.read_locked(alias, "a"*64)
        original.write_text("[]")
        with pytest.raises(ValueError, match="locked evidence changed"):
            legacy.read_locked(alias, digest)
    assert legacy.read_locked is read_locked
