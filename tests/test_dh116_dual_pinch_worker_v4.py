"""A fresh worker binds the actual CuboidCfg collision prim without changing v3."""

from pathlib import Path

from test_dh116_pinch_v3 import BLOCKER, cpu
from scripts import verify_dh116_dual_pinch_interval_v4 as entry


ROOT = Path(__file__).resolve().parents[1]


def test_byte_only_v4_preflight_checks_installed_spawner_and_exact_v3_failure():
    result = cpu(BLOCKER + """
from pathlib import Path
from scripts import verify_dh116_dual_pinch_interval_v4 as entry
r = entry.preflight(Path.cwd())
assert r['v3_failure_receipt_sha256'] == entry.FAILED_SHA
assert r['sources']['external_sources'][str(entry.SHAPE_SOURCE)] == entry.SHAPE_SHA
assert 'isaaclab' not in sys.modules and 'ropesim' not in sys.modules
print('DUAL_V4_PREFLIGHT')
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "DUAL_V4_PREFLIGHT"


def test_only_this_worker_swaps_the_table_source(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_bimanual_table_contacts as original
    from go2_pvcnn.control.dh116 import tabletop_bimanual_table_contacts_v2 as resolved

    legacy = original.open_bimanual_table_contacts
    calls = []

    def check(args, project, contract):
        assert original.open_bimanual_table_contacts is resolved.open_bimanual_table_contacts
        calls.append((args, project, contract))

    monkeypatch.setattr(entry, "execute_previous", check)
    entry.execute("args", ROOT, "contract")
    assert calls == [("args", ROOT, "contract")]
    assert original.open_bimanual_table_contacts is legacy


from test_dh116_dual_pinch_builder import prepared, geometry


def test_descendant_layout_runs_original_owned_eight_tick_path(monkeypatch, tmp_path, prepared, geometry):
    from pxr import Sdf, UsdGeom, UsdPhysics
    from test_dh116_dual_pinch_entry import test_actual_builder_journal_eight_ticks_and_lifecycle_failure_boundary
    from go2_pvcnn.control.dh116 import tabletop_bimanual_table_contacts as original
    from go2_pvcnn.control.dh116 import tabletop_bimanual_table_contacts_v2 as resolved
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from scripts import prepare_dh116_dual_pinch_interval as interval
    from scripts import verify_dh116_dual_pinch_interval_v3 as previous

    author = scene.author_source_stage
    table = prepared["prefix"] + "/Table"

    def author_nested(project, config):
        stage = author(project, config)
        layer = stage.GetRootLayer()
        temporary = Sdf.Path(table + "Temporary")
        assert Sdf.CopySpec(layer, Sdf.Path(table), layer, temporary)
        assert stage.RemovePrim(table)
        root = UsdGeom.Xform.Define(stage, table).GetPrim()
        UsdGeom.Xform.Define(stage, table + "/geometry")
        assert Sdf.CopySpec(layer, temporary, layer, Sdf.Path(table + "/geometry/mesh"))
        assert stage.RemovePrim(temporary)
        assert not root.HasAPI(UsdPhysics.CollisionAPI)
        assert stage.GetPrimAtPath(table + "/geometry/mesh").HasAPI(UsdPhysics.CollisionAPI)
        return stage

    real = interval.run_existing_app

    def native_default(*args, **kwargs):
        kwargs["prepared"] = None
        return previous.run_existing_app(*args, **kwargs)

    monkeypatch.setattr(scene, "author_source_stage", author_nested)
    monkeypatch.setattr(scene, "prepare_scene", lambda _: prepared)
    monkeypatch.setattr(previous, "original_interval", real)
    monkeypatch.setattr(interval, "run_existing_app", native_default)
    monkeypatch.setattr(original, "open_bimanual_table_contacts", resolved.open_bimanual_table_contacts)
    test_actual_builder_journal_eight_ticks_and_lifecycle_failure_boundary(
        monkeypatch, tmp_path, prepared, geometry, None
    )
    import json
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    rows = (tmp_path / "entry" / "physical.jsonl").read_text().splitlines()[1:]
    found = [decode_owned(row["payload"])["hand_table_contacts"]
             for row in map(json.loads, rows) if row["operation"] == "physical"]
    assert len(found) == 8
    assert all(row["source"]["table_colliders"]["collider_paths"] == [table + "/geometry/mesh"] for row in found)
    assert all(row["total_contacts"] is None for row in found)
