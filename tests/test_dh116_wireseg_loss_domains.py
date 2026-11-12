import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import audit_dh116_wireseg_loss_domains as audit


def row(name, count, mae, loss, roi_count, roi_mae, *, pixels=16):
    def metric(n, error):
        return dict(valid_pixels=n, mae_m=error if n else None,
                    rmse_m=error if n else None, p95_m=error if n else None)
    return dict(file_name=name, family="fixture", split="train", view="center",
                pair_sha256="a" * 64, checkpoint_sha256="b" * 64, loss=loss,
                valid_depth_fraction=count / pixels, wire_fraction=roi_count / pixels,
                depth=dict(all_valid=metric(count, mae), wire_roi=metric(roi_count, roi_mae)),
                constant=dict(all_valid=metric(count, .2), wire_roi=metric(roi_count, .2)))


def test_domain_aggregation_separates_image_and_pixel_weights():
    rows = [row("one", 1, 1., .995, 1, 1.), row("two", 9, 0., 0., 3, 0.)]
    result = audit.summarize_domain(rows, ["one", "two"], pixels_per_image=16)
    assert result["loss"]["equal_image_smooth_l1"] == pytest.approx(.4975)
    assert result["loss"]["pixel_weighted_smooth_l1"] == pytest.approx(.0995)
    assert result["depth"]["all_valid"]["equal_image_mae_m"] == pytest.approx(.5)
    assert result["depth"]["all_valid"]["pixel_mae_m"] == pytest.approx(.1)
    assert result["depth"]["all_valid"]["pixel_rmse_m"] == pytest.approx(np.sqrt(.1))
    assert result["depth"]["background"]["valid_pixels"] == 6
    assert result["depth"]["background"]["pixel_mae_m"] == 0.
    assert result["depth"]["all_valid"]["pixel_p95_m"] is None
    assert result["p95_status"] == "not_reconstructible_from_cached_per_image_summaries"


def test_partial_report_keeps_fixed_denominators_and_missing_images():
    result = audit.summarize_domain([row("one", 4, .1, .095, 4, .1)],
                                   ["one", "two"], pixels_per_image=16)
    assert result["expected_images"] == 2 and result["completed_images"] == 1
    assert result["missing_images"] == ["two"]
    assert result["fixed_total_pixels"] == 32 and result["observed_total_pixels"] == 16
    assert result["valid_fraction_observed"] == .25
    assert result["valid_fraction_fixed_lower_bound"] == .125


def test_cached_completed_view_skips_model_inference(monkeypatch):
    sample = SimpleNamespace(file_name="one", scene_family="fixture", pair_sha256="a" * 64)
    cached = row("one", 4, .1, .095, 4, .1)
    cache = {("train", "one", "center"): cached}
    monkeypatch.setattr(audit, "inspect_sample", lambda *a: pytest.fail("cached view repeated inference"))
    result = audit.inspect_or_reuse(None, sample, "train", "center", 4, 1., "b" * 64, cache)
    assert result["reused"] is True
    cached["pair_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        audit.inspect_or_reuse(None, sample, "train", "center", 4, 1., "b" * 64, cache)


def test_background_derived_from_disjoint_roi_not_image_fraction():
    record = row("one", 10, .2, .195, 2, .5)
    record["depth"]["all_valid"]["rmse_m"] = .25  # 2 errors .5 and 8 errors .125.
    result = audit.summarize_domain([record], ["one"], pixels_per_image=16)
    assert result["depth"]["background"]["pixel_mae_m"] == pytest.approx(.125)
    with pytest.raises(ValueError, match="ROI"):
        audit.summarize_domain([row("bad", 1, .2, .195, 2, .5)], ["bad"], pixels_per_image=16)


def test_cached_report_requires_sha_checkpoint_and_view_binding(tmp_path):
    import json
    from scripts.relay_wireseg_blobs import digest_file
    path = tmp_path / "cache.json"
    binding = audit.view_binding(4)
    report = dict(format=audit.FORMAT, source_sha256=binding["aggregation"]["audit_source_sha256"],
                  real_training_updates=0, training_constant_depth_m=1.,
                  checkpoint_sha256="b" * 64, view_binding=binding,
                  records=[dict(row("one", 4, .1, .095, 4, .1), checkpoint_head="depth-only")])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="SHA256"):
        audit.load_cache(path, "f" * 64, "b" * 64, 1., binding)
    assert len(audit.load_cache(path, digest_file(path), "b" * 64, 1., binding)) == 1
    with pytest.raises(ValueError, match="view"):
        audit.load_cache(path, digest_file(path), "b" * 64, 1., {**binding, "size": 5})
    with pytest.raises(ValueError, match="checkpoint"):
        audit.load_cache(path, digest_file(path), "c" * 64, 1., binding)


def test_runtime_helper_change_rejects_cached_rows(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    original_digest = audit.digest_file
    binding = audit.view_binding(192)
    report = dict(format=audit.FORMAT, source_sha256=original_digest(Path(audit.__file__)),
                  real_training_updates=0, training_constant_depth_m=1.,
                  checkpoint_sha256="b" * 64, view_binding=binding, records=[])
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(report))
    sha = original_digest(path)
    monkeypatch.setattr(audit, "digest_file", lambda p:
                        "f" * 64 if Path(p).name == "audit_dh116_wireseg_depth_error.py" else original_digest(p))
    with pytest.raises(ValueError, match="helper|view"):
        audit.load_cache(path, sha, "b" * 64, 1., audit.view_binding(192))


def test_new_cache_rejects_changed_aggregation_source(tmp_path):
    import json
    binding = audit.view_binding(192)
    report = dict(format=audit.FORMAT, source_sha256="f" * 64,
                  real_training_updates=0, training_constant_depth_m=1.,
                  checkpoint_sha256="b" * 64, view_binding=binding, records=[])
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source|aggregation"):
        audit.load_cache(path, audit.digest_file(path), "b" * 64, 1., binding)


@pytest.mark.parametrize("legacy_domain", [False, True])
def test_legacy_cache_rejects_changed_runtime_helper_before_checkpoint_read(tmp_path, monkeypatch, legacy_domain):
    import json
    from pathlib import Path
    original_digest = audit.digest_file
    report = dict(source_sha256=audit.LEGACY_DOMAIN_SOURCE_SHA if legacy_domain else audit.LEGACY_AUDIT_SOURCE_SHA,
                  real_training_updates=0, training_constant_depth_m=1., source_checkpoints=[], records=[])
    if legacy_domain:
        report.update(format=audit.FORMAT, checkpoint_sha256="b" * 64,
                      view_binding=dict(size=192, views=["center", "full_field_resize"],
                                        implementation_sha256=audit.LEGACY_VIEW_IMPLEMENTATION))
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(report))
    sha = original_digest(path)
    monkeypatch.setattr(audit, "digest_file", lambda p:
                        "f" * 64 if Path(p).name == "audit_dh116_wireseg_depth_error.py" else original_digest(p))
    monkeypatch.setattr(audit, "read_checkpoint", lambda *a: pytest.fail("changed helper reached checkpoint loading"))
    with pytest.raises(ValueError, match="helper"):
        audit.load_cache(path, sha, "b" * 64, 1., audit.view_binding(192))


def test_runtime_aggregation_function_change_rejects_cached_rows(tmp_path, monkeypatch):
    import json
    binding = audit.view_binding(192)
    report = dict(format=audit.FORMAT, source_sha256=binding["aggregation"]["audit_source_sha256"],
                  real_training_updates=0, training_constant_depth_m=1.,
                  checkpoint_sha256="b" * 64, view_binding=binding, records=[])
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(report))
    sha = audit.digest_file(path)
    monkeypatch.setattr(audit, "summarize_domain", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="aggregation"):
        audit.load_cache(path, sha, "b" * 64, 1., audit.view_binding(192))
