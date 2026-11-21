"""Fresh 17-slot composition: no prior Reset/settling and no native launch."""
from copy import deepcopy
import numpy as np
import pytest

from test_dh116_power_end_supported_rod import supported_fixture
from test_dual_hand_rod_runtime import Grid
from go2_pvcnn.control.dh116 import tabletop_power_shared_rod as power


def fresh_proximal_case(monkeypatch):
    base, poses, centers, kwargs = supported_fixture(monkeypatch)
    proximal = {}
    for side, token in zip(power.SIDES, ("a", "b"), strict=True):
        key = side + "." + power.PROXIMAL_LINK
        kwargs["bindings"][key] = dict(kwargs["bindings"][side+".base_link"],
            body_name=power.PROXIMAL_LINK, body_index=15,
            prim_path=f"/World/{side}Hand/{power.PROXIMAL_LINK}",
            capture_sha256=token*64, source_geometry_sha256=token*64,
            cooked_geometry_sha256=token*64, sdf_sha256=token*64)
        proximal[key] = (np.array([0., .01, .4]), np.array([0., 0., 0., 1.]))
    kwargs.update(proximal_grids={key: Grid() for key in proximal}, proximal_poses=proximal)
    return base, kwargs


def test_fresh_supported_constructor_registers_all17_without_a_hidden_reset(monkeypatch):
    base, kwargs = fresh_proximal_case(monkeypatch)
    identity, gravity = id(base.rope), deepcopy(base.cfg["rod"]["gravity_world_m_s2"])
    runtime = power.PowerSharedRodRuntime(base, **kwargs)
    assert id(runtime.rope) == identity
    assert runtime.mesh_names == power.PROXIMAL_POWER_MESH_NAMES
    assert len(runtime.rope.slots) == 17
    assert runtime.rope.resets == runtime.rope.steps == 0 and runtime.rope.time == 0.
    assert runtime.clock()["physics_step"] is None and runtime.clock()["reset_id"] == 0
    assert runtime.clock()["clock_valid"] is False
    assert not runtime.rope.pinned.any() and runtime._base.cfg["rod"]["gravity_world_m_s2"] == gravity
    for key in power.PROXIMAL_NAMES:
        slot = runtime.mesh_names.index(key)
        assert slot in (15, 16)
        np.testing.assert_array_equal(runtime.rope.slots[slot][1]["pos"], kwargs["proximal_poses"][key][0].astype(np.float32))


def test_fresh_constructor_rejects_partial_or_aliased_proximal_sources_before_append(monkeypatch):
    # Reuse the same untouched six-slot base for both failed source preflights.
    base, kwargs = fresh_proximal_case(monkeypatch)
    missing = dict(kwargs, proximal_poses={"left.finger52_link": kwargs["proximal_poses"]["left.finger52_link"]})
    with pytest.raises(ValueError, match="proximal"):
        power.PowerSharedRodRuntime(base, **missing)
    assert len(base.rope.slots) == 6 and not getattr(base, "_dual_hand_composition_claimed", False)
    for field in ("capture_sha256", "source_geometry_sha256", "cooked_geometry_sha256", "sdf_sha256"):
        kwargs["bindings"]["right.finger52_link"][field] = kwargs["bindings"]["left.finger52_link"][field]
    with pytest.raises(ValueError, match="proximal"):
        power.PowerSharedRodRuntime(base, **kwargs)
    assert len(base.rope.slots) == 6 and base.rope.steps == base.rope.resets == 0
