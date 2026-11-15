"""Fixed-anchor lifecycle CPU fixtures; no native source qualification."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from go2_pvcnn.control.dh116.tabletop_training_reward import Measurement, SourceFailure, SourceStamp


def module():
    from go2_pvcnn.control.dh116 import tabletop_native_reward_sources_v2
    return tabletop_native_reward_sources_v2


def refs(tick=2):
    cls = module().FixedAnchorV2
    return (cls("left", "material", "m", tick, (.01,)),
            cls("right", "material", "m", tick, (.02,)),
            cls("left", "tangential", "tl", tick, (0.,)),
            cls("right", "tangential", "tr", tick, (0.,)))


def observe(owner, tick=2, *, episode_uid="fixture:e", contacting=True,
            references=None, pairs=((0., .00001),) * 4, geometry_drift=()):
    return owner.observe(episode_uid=episode_uid, tick=tick, contacting=contacting,
        references=refs() if references is None else references,
        pairs=pairs, geometry_drift=geometry_drift)


def anchored():
    owner = module().SlipLifecycleOwnerV2()
    first = observe(owner)
    return owner, first


def geometry(tick=3, **changes):
    stamp = SourceStamp("fixture:e", tick, tick / 480, "cpu_fixture", "fixture:geometry")
    value = Measurement(.0009, "m", stamp)
    return replace(value, **changes)


def test_old_slip_retains_anchor_after_detachment():
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import FlexibleGraspEpisode
    anchor = ((.01, .02), (.00001, .00001))
    task = SimpleNamespace(material_anchor=anchor)
    assert FlexibleGraspEpisode._slip(task, (None, None), False) == (anchor, None, None, None)


def test_fixed_anchor_contact_loss_keeps_reference_and_explicit_na():
    owner, first = anchored()
    lost = observe(owner, 3, contacting=False, pairs=None)
    assert first.state == "contacting"
    assert lost.state == "detached"
    assert lost.references == refs() == first.references
    assert lost.contact_pairs is None
    assert lost.na_reason == "fixed_anchor_detached"


def test_never_anchored_does_not_invent_references():
    owner = module().SlipLifecycleOwnerV2()
    receipt = observe(owner, contacting=False, references=(), pairs=None)
    assert (receipt.state, receipt.references, receipt.contact_pairs, receipt.na_reason) == (
        "never_anchored", (), None, "no_bilateral_contact_since_reset")
    first = observe(owner, 3, references=refs(3))
    assert first.references == refs(3)


def test_never_anchored_rejects_supplied_references_and_pairs_atomically():
    owner = module().SlipLifecycleOwnerV2()
    for references, pairs in ((refs(), None), ((), ((0., .0),) * 4)):
        with pytest.raises(SourceFailure):
            observe(owner, contacting=False, references=references, pairs=pairs)
    # Failed first observations do not bind the owner to an episode or a tick.
    result = observe(owner, 8, episode_uid="fixture:other", references=refs(8))
    assert result.state == "contacting"


def test_recontact_preserves_nonzero_drift_against_original_anchor():
    owner, first = anchored()
    observe(owner, 3, contacting=False, pairs=None)
    current = observe(owner, 4, pairs=((.0007, .00001),) * 4)
    assert current.references == first.references
    assert current.contact_pairs == ((.0007, .00001),) * 4
    assert current.state == "contacting" and current.na_reason is None


@pytest.mark.parametrize("contacting", [False, True])
@pytest.mark.parametrize("change", ["missing", "uid", "reference", "tick", "reordered"])
def test_frozen_reference_cannot_disappear_or_change(contacting, change):
    owner, first = anchored()
    bad = list(refs())
    if change == "missing":
        bad = []
    elif change == "reordered":
        bad.reverse()
    else:
        field = {"uid": "uid", "reference": "reference", "tick": "reference_tick"}[change]
        value = {"uid": "new-nearest-point", "reference": (.03,), "tick": 3}[change]
        bad[0] = replace(bad[0], **{field: value})
    with pytest.raises(SourceFailure):
        observe(owner, 3, contacting=contacting, references=bad,
                pairs=((0., .00001),) * 4 if contacting else None)
    assert owner.current_receipt == first
    assert observe(owner, 3).references == first.references


@pytest.mark.parametrize("pairs", [
    None, (), ((0., .00001),) * 3, ((0., .00001),) * 5,
    ((float("nan"), .0),) * 4, ((.0, float("nan")),) * 4,
    ((.0, -.00001),) * 4, ((-.1, .0),) * 4, ((True, .0),) * 4,
    ((.0, float("inf")),) * 4, ((.0,),) * 4, ((.0, .0, .0),) * 4,
    ((1e308, 1e308),) * 4, ((10**400, .0),) * 4,
])
def test_contact_requires_four_finite_nonnegative_pairs_atomically(pairs):
    owner, first = anchored()
    with pytest.raises(SourceFailure):
        observe(owner, 3, pairs=pairs)
    assert owner.current_receipt == first
    assert observe(owner, 3).state == "contacting"


@pytest.mark.parametrize("contacting", [None, 0, 1, "true", float("nan"), [], object()])
def test_unknown_contact_boolean_is_rejected_atomically(contacting):
    owner, first = anchored()
    with pytest.raises(SourceFailure):
        observe(owner, 3, contacting=contacting)
    assert owner.current_receipt == first
    assert observe(owner, 3).state == "contacting"


def test_detached_rejects_contact_pairs_instead_of_treating_geometry_as_contact():
    owner, first = anchored()
    with pytest.raises(SourceFailure):
        observe(owner, 3, contacting=False, geometry_drift=(geometry(),))
    assert owner.current_receipt == first
    detached = observe(owner, 3, contacting=False, pairs=None, geometry_drift=(geometry(),))
    assert detached.geometry_drift == (geometry(),)
    assert detached.contact_pairs is None


def test_geometry_survives_detachment_and_never_anchored():
    owner, _ = anchored()
    lost = observe(owner, 3, contacting=False, pairs=None, geometry_drift=(geometry(),))
    assert lost.geometry_drift == (geometry(),)
    fresh = module().SlipLifecycleOwnerV2()
    never = observe(fresh, 3, contacting=False, references=(), pairs=None,
                    geometry_drift=(geometry(),))
    assert never.geometry_drift == (geometry(),) and never.references == ()


def test_geometry_cannot_replace_missing_contact_pairs():
    owner, first = anchored()
    with pytest.raises(SourceFailure):
        observe(owner, 3, pairs=None, geometry_drift=(geometry(),))
    assert owner.current_receipt == first
    assert observe(owner, 3).state == "contacting"


@pytest.mark.parametrize("bad", [
    object(), SimpleNamespace(raw=.1, unit="m", stamp=SimpleNamespace(episode_uid="fixture:e", physics_tick=3)),
    geometry(raw=None), geometry(raw=True), geometry(raw=-.1), geometry(raw=float("nan")),
    geometry(unit="dimensionless"), geometry(valid=False), geometry(valid=1),
    geometry(applicable=False), geometry(applicable=1), geometry(history_reason="unproven"),
    geometry(stamp=replace(geometry().stamp, episode_uid="fixture:other")),
    geometry(stamp=replace(geometry().stamp, physics_tick=4)),
    geometry(stamp=replace(geometry().stamp, physics_tick=True)),
    geometry(stamp=replace(geometry().stamp, solver_time_s=float("nan"))),
    geometry(stamp=replace(geometry().stamp, scope="native")),
    geometry(stamp=replace(geometry().stamp, env_id=True)),
    geometry(stamp=replace(geometry().stamp, frame="local")),
    geometry(stamp=replace(geometry().stamp, reset_uid="")),
])
def test_geometry_requires_real_valid_same_clock_measurement_atomically(bad):
    owner, first = anchored()
    with pytest.raises(SourceFailure):
        observe(owner, 3, contacting=False, pairs=None, geometry_drift=(bad,))
    assert owner.current_receipt == first
    assert observe(owner, 3, contacting=False, pairs=None).state == "detached"


@pytest.mark.parametrize("field,value", [
    ("solver_time_s", .5), ("env_id", 1), ("reset_uid", "other-reset"),
    ("rod_uid", "other-rod"), ("scope", "unqualified_source"),
])
def test_geometry_bundle_requires_one_complete_clock_identity(field, value):
    owner, first = anchored()
    changed = geometry(stamp=replace(geometry().stamp, **{field: value}))
    with pytest.raises(SourceFailure):
        observe(owner, 3, geometry_drift=(geometry(), changed))
    assert owner.current_receipt == first
    assert observe(owner, 3).state == "contacting"


@pytest.mark.parametrize("tick", [True, False, None, -1, 2., float("inf"), float("nan"), "2", 10**400])
def test_invalid_typed_clock_is_atomic(tick):
    owner = module().SlipLifecycleOwnerV2()
    with pytest.raises(SourceFailure):
        observe(owner, tick)
    assert observe(owner).state == "contacting"


@pytest.mark.parametrize("tick", [2, 1, 4])
def test_clock_requires_exactly_one_forward_tick(tick):
    owner, first = anchored()
    with pytest.raises(SourceFailure):
        observe(owner, tick)
    assert owner.current_receipt == first
    assert observe(owner, 3).state == "contacting"


@pytest.mark.parametrize("uid", ["", " ", None, True, 1])
def test_episode_uid_must_be_nonempty_primitive_text(uid):
    owner = module().SlipLifecycleOwnerV2()
    with pytest.raises(SourceFailure):
        observe(owner, episode_uid=uid)
    assert observe(owner).state == "contacting"


def test_new_episode_requires_reset_and_uid_cannot_be_reused():
    owner, first = anchored()
    with pytest.raises(SourceFailure):
        observe(owner, 3, episode_uid="fixture:new")
    assert owner.current_receipt == first
    reset = owner.reset(episode_uid="fixture:new", tick=20)
    assert reset == owner.current_receipt
    assert (reset.state, reset.references, reset.contact_pairs, reset.geometry_drift) == (
        "never_anchored", (), None, ())
    assert reset.na_reason == "no_bilateral_contact_since_reset"
    for reused in ("fixture:e", "fixture:new"):
        with pytest.raises(SourceFailure):
            owner.reset(episode_uid=reused, tick=0)
        assert owner.current_receipt == reset
    assert observe(owner, 21, episode_uid="fixture:new", references=refs(21)).state == "contacting"


def test_clear_is_a_new_reset_with_full_history_retention():
    owner, _ = anchored()
    cleared = owner.clear(episode_uid="fixture:clear", tick=50)
    assert cleared.state == "never_anchored" and cleared.references == ()
    with pytest.raises(SourceFailure):
        owner.clear(episode_uid="fixture:e", tick=0)
    with pytest.raises(SourceFailure):
        owner.clear(episode_uid="fixture:clear", tick=0)
    assert owner.current_receipt == cleared
    assert observe(owner, 51, episode_uid="fixture:clear",
                   contacting=False, references=(), pairs=None).state == "never_anchored"


def test_invalid_reset_does_not_clear_anchor_or_consume_new_uid():
    owner, first = anchored()
    with pytest.raises(SourceFailure):
        owner.reset(episode_uid="fixture:new", tick=True)
    assert owner.current_receipt == first
    assert owner.reset(episode_uid="fixture:new", tick=20).references == ()


@pytest.mark.parametrize("field,value", [
    ("side", "middle"), ("kind", "nearest"), ("uid", ""),
    ("reference_tick", True), ("reference_tick", -1), ("reference_tick", float("nan")),
    ("reference", ()), ("reference", (float("nan"),)), ("reference", ([.1],)),
    ("reference", (True,)),
])
def test_fixed_anchor_schema_is_strict(field, value):
    fields = dict(side="left", kind="material", uid="m", reference_tick=2, reference=(.01,))
    fields[field] = value
    with pytest.raises(SourceFailure):
        module().FixedAnchorV2(**fields)


def test_four_distinct_anchor_groups_and_first_reference_clock_are_required():
    for bad in (refs()[:3], (refs()[0],) * 4, refs(3), refs(1),
                (SimpleNamespace(**vars(geometry())),) * 4):
        owner = module().SlipLifecycleOwnerV2()
        with pytest.raises(SourceFailure):
            observe(owner, references=bad)
        assert observe(owner).state == "contacting"


def test_owned_snapshot_survives_input_list_mutation_and_is_frozen():
    cls = module().FixedAnchorV2
    coordinates = [.01]
    anchor = cls("left", "material", "m", 2, coordinates)
    coordinates[0] = .99
    anchors = [anchor, *refs()[1:]]
    pairs = [[.0007, .00001] for _ in range(4)]
    owner = module().SlipLifecycleOwnerV2()
    receipt = observe(owner, references=anchors, pairs=pairs, geometry_drift=[geometry(2)])
    anchors.clear()
    pairs[0][0] = 0.
    assert receipt.references[0].reference == (.01,)
    assert receipt.contact_pairs == ((.0007, .00001),) * 4
    assert type(receipt.references) is tuple and type(receipt.geometry_drift) is tuple
    for item, field, value in ((receipt, "state", "detached"), (anchor, "uid", "replacement")):
        with pytest.raises(FrozenInstanceError):
            setattr(item, field, value)


def test_result_and_public_dataclass_never_grant_native_qualification():
    owner, receipt = anchored()
    assert receipt.source_scope == "cpu_math_fixture"
    assert receipt.native_ready is False
    with pytest.raises(TypeError):
        replace(receipt, native_ready=True)
    with pytest.raises(TypeError):
        owner.observe(episode_uid="fixture:e", tick=3, contacting=True,
                      references=refs(), pairs=((0., .00001),) * 4, native_ready=True)
    assert observe(owner, 3).state == "contacting"


def test_current_receipt_requires_successful_observation_or_reset():
    owner = module().SlipLifecycleOwnerV2()
    with pytest.raises(SourceFailure):
        _ = owner.current_receipt



# Public dataclasses are not a history boundary: object.__setattr__ can bypass
# frozen=True. All probes below use public receipts only, never owner internals.
def test_public_observe_receipt_reference_replacement_cannot_reanchor():
    owner, public = anchored()
    changed = (replace(public.references[0], uid="new-nearest", reference=(.99,)),
               *public.references[1:])
    object.__setattr__(public, "references", changed)
    with pytest.raises(SourceFailure):
        observe(owner, 3, references=changed)
    assert owner.current_receipt.references == refs()
    assert observe(owner, 3).references == refs()


def test_public_nested_anchor_mutation_cannot_reanchor():
    owner, public = anchored()
    object.__setattr__(public.references[0], "reference", (.99,))
    changed = tuple(public.references)
    with pytest.raises(SourceFailure):
        observe(owner, 3, references=changed)
    assert owner.current_receipt.references == refs()
    assert observe(owner, 3).references == refs()


def test_public_reference_erasure_cannot_erase_anchor_history():
    owner, public = anchored()
    object.__setattr__(public, "references", ())
    with pytest.raises(SourceFailure):
        observe(owner, 3, contacting=False, references=(), pairs=None)
    detached = observe(owner, 3, contacting=False, pairs=None)
    assert detached.state == "detached" and detached.references == refs()
    with pytest.raises(SourceFailure):
        observe(owner, 4, references=refs(4))
    assert observe(owner, 4).references == refs()


@pytest.mark.parametrize("reset_method", ["reset", "clear"])
def test_public_reset_receipt_future_anchor_injection_cannot_bypass_clock(reset_method):
    owner, _ = anchored()
    public = getattr(owner, reset_method)(episode_uid="fixture:new", tick=10)
    object.__setattr__(public, "references", refs(100))
    with pytest.raises(SourceFailure):
        observe(owner, 11, episode_uid="fixture:new", references=refs(100))
    assert owner.current_receipt.references == ()
    assert owner.current_receipt.state == "never_anchored"
    assert observe(owner, 11, episode_uid="fixture:new", references=refs(11)).references == refs(11)


@pytest.mark.parametrize("surface", ["observe", "current_receipt"])
@pytest.mark.parametrize("field,value", [
    ("references", ()), ("contact_pairs", ((9., 9.),) * 4),
    ("state", "never_anchored"), ("na_reason", "forged"),
    ("geometry_drift", ()),
])
def test_public_receipt_field_mutation_is_isolated_and_failure_stays_atomic(surface, field, value):
    owner = module().SlipLifecycleOwnerV2()
    initial = observe(owner, geometry_drift=(geometry(2),))
    expected = module().SlipLifecycleV2("contacting", refs(), ((0., .00001),) * 4,
                                       None, (geometry(2),))
    public = initial if surface == "observe" else owner.current_receipt
    sibling = owner.current_receipt
    object.__setattr__(public, field, value)
    assert owner.current_receipt == expected
    assert sibling == expected
    with pytest.raises(SourceFailure):
        observe(owner, 3, pairs=((float("nan"), .0),) * 4)
    assert owner.current_receipt == expected
    recovered = observe(owner, 3, contacting=False, pairs=None, geometry_drift=(geometry(),))
    assert recovered.state == "detached"
    assert recovered.references == refs() and recovered.geometry_drift == (geometry(),)


@pytest.mark.parametrize("surface", ["observe", "current_receipt"])
@pytest.mark.parametrize("leaf", ["anchor", "measurement", "stamp"])
def test_public_dataclass_leaf_mutation_is_isolated_across_returns(surface, leaf):
    owner = module().SlipLifecycleOwnerV2()
    initial = observe(owner, geometry_drift=(geometry(2),))
    public = initial if surface == "observe" else owner.current_receipt
    sibling = owner.current_receipt
    if leaf == "anchor":
        object.__setattr__(public.references[0], "reference", (.99,))
    elif leaf == "measurement":
        object.__setattr__(public.geometry_drift[0], "raw", 99.)
    else:
        object.__setattr__(public.geometry_drift[0].stamp, "physics_tick", 100)
    expected = module().SlipLifecycleV2("contacting", refs(), ((0., .00001),) * 4,
                                       None, (geometry(2),))
    assert owner.current_receipt == expected
    assert sibling == expected
    assert observe(owner, 3).references == refs()


def test_public_returns_have_distinct_dataclass_leaves_from_inputs_and_each_other():
    owner = module().SlipLifecycleOwnerV2()
    input_refs = refs()
    input_geometry = (geometry(2),)
    initial = observe(owner, references=input_refs, geometry_drift=input_geometry)
    first = owner.current_receipt
    second = owner.current_receipt
    views = (initial, first, second)
    assert len({id(value) for value in views}) == 3
    for index in range(4):
        assert len({id(value.references[index]) for value in views} |
                   {id(input_refs[index])}) == 4
    assert len({id(value.geometry_drift[0]) for value in views} |
               {id(input_geometry[0])}) == 4
    assert len({id(value.geometry_drift[0].stamp) for value in views} |
               {id(input_geometry[0].stamp)}) == 4
    reset = owner.reset(episode_uid="fixture:next", tick=10)
    assert reset is not owner.current_receipt
    cleared = owner.clear(episode_uid="fixture:third", tick=20)
    assert cleared is not owner.current_receipt


def test_old_public_return_mutations_cannot_rewrite_later_geometry_or_anchor_history():
    owner = module().SlipLifecycleOwnerV2()
    previous = observe(owner, geometry_drift=(geometry(2),))
    current = observe(owner, 3, geometry_drift=(geometry(),))
    object.__setattr__(previous.references[0], "uid", "changed-old-return")
    object.__setattr__(previous.geometry_drift[0].stamp, "episode_uid", "forged")
    object.__setattr__(previous, "contact_pairs", None)
    assert current.references == refs()
    assert owner.current_receipt.references == refs()
    assert owner.current_receipt.geometry_drift == (geometry(),)
    assert observe(owner, 4).references == refs()
