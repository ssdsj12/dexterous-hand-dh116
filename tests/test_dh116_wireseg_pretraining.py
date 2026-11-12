import importlib
import io
import json
import tarfile

import numpy as np
from PIL import Image
import pytest
import torch


def module():
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_wireseg_pretraining")


def test_coco_masks_follow_column_major_and_signed_delta_runs():
    mask = module().decode_coco_rle({"size": [3, 3], "counts": "1230N"}, (3, 3))
    assert mask.tolist() == [[False, False, True], [True, False, True], [True, False, False]]
    with pytest.raises(ValueError, match="cover"):
        module().decode_coco_rle({"size": [3, 3], "counts": [1, 2]}, (3, 3))


def test_identity_split_is_source_exclusive_and_real_held_out():
    wire = module()
    assert wire.sample_split("flying_wire") == "train"
    assert wire.sample_split("wire_on_plane") == "validation"
    assert wire.sample_split("data_center") == "test"
    assert wire.sample_split("real_hard") == "real_test"
    assert wire.depth_member("flying_wire/2_cable/000001.png", "flying_wire") == (
        "flying_wire/2_cable/000001.npy"
    )
    with pytest.raises(ValueError):
        wire.depth_member("flying_wire/../2_cable/000001.png", "flying_wire")


def test_metric_depth_requires_same_pixel_grid_and_explicit_validity(tmp_path):
    wire = module()
    rgb = io.BytesIO()
    Image.new("RGB", (2, 2)).save(rgb, format="PNG")
    row = {"file_name": "flying_wire/2_cable/0001.png", "scene_family": "flying_wire",
           "image": {"bytes": rgb.getvalue()}, "masks_rle": json.dumps([{"size": [2, 2], "counts": [1, 1, 2]}]),
           "num_instances": 1, "n_cable": 2}
    depth_path = tmp_path / "flying_wire/2_cable/0001.npy"
    depth_path.parent.mkdir(parents=True)
    depth = np.asarray([[0.0, 2.5], [np.nan, 8.25]], dtype=np.float32)
    np.save(depth_path, depth)
    sample = wire.load_wireseg_sample(row, tmp_path)
    assert sample.depth_valid.tolist() == [[False, True], [False, True]]
    assert sample.depth_m[sample.depth_valid].tolist() == [2.5, 8.25]
    assert len(sample.pair_sha256) == 64
    row["file_name"] = "flying_wire/2_cable/0002.png"
    missing = wire.load_wireseg_sample(row, tmp_path)
    assert missing.depth_m is None and missing.depth_valid is None and missing.pair_sha256 is None


def test_streaming_depth_extraction_rejects_links_and_preserves_raw_payload(tmp_path):
    wire = module()
    archive = tmp_path / "depth.tar.gz"
    name = "flying_wire/2_cable/0001.npy"
    raw = io.BytesIO()
    np.save(raw, np.ones((2, 2), dtype=np.float32))
    with tarfile.open(archive, "w:gz") as tf:
        item = tarfile.TarInfo(name)
        item.size = len(raw.getvalue())
        tf.addfile(item, io.BytesIO(raw.getvalue()))
    target = tmp_path / "decoded"
    assert wire.extract_depth_shard(archive, target) == [name]
    assert wire.extract_depth_shard(archive, target) == [name]
    assert (target / name).read_bytes() == raw.getvalue()
    with tarfile.open(archive, "w:gz") as tf:
        item = tarfile.TarInfo("flying_wire/2_cable/link.npy")
        item.type = tarfile.SYMTYPE
        item.linkname = name
        tf.addfile(item)
    with pytest.raises(ValueError, match="regular"):
        wire.extract_depth_shard(archive, target)


def test_mono_pretraining_updates_exact_stereo_backbone_and_all_supervised_heads():
    wire = module()
    from go2_pvcnn.control.dh116.tabletop_stereo_encoder import StereoEncoder

    model = wire.WireSegPretrainer(embed_dim=32, max_instances=4)
    rgb = torch.rand((1, 3, 32, 32))
    masks = torch.zeros((2, 32, 32), dtype=torch.bool)
    masks[0, 2:15, 8:10] = True
    masks[1, 15:25, 9:10] = True
    depth = torch.full((1, 32, 32), 2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[0, 0, 0] = False
    depth[0, 0, 0] = torch.nan
    result = model(rgb)
    losses = wire.pretraining_losses(result, [masks], depth, valid)
    assert set(losses) == {"instance", "visible", "boundary", "metric_depth", "total"}
    losses["total"].backward()
    for head in (model.backbone.stem[0], model.instance_head, model.visible_head, model.boundary_head, model.depth_head):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    stereo = StereoEncoder(embed_dim=32)
    original_attention = {key: value.clone() for key, value in stereo.cross_attention.state_dict().items()}
    wire.transfer_wireseg_backbone(stereo, model.transfer_state())
    for key, value in model.backbone.state_dict().items():
        assert torch.equal(stereo.backbone.state_dict()[key], value)
    for key, value in stereo.cross_attention.state_dict().items():
        assert torch.equal(value, original_attention[key])


def test_explicit_verified_depth_inventory_excludes_orphan_file(tmp_path):
    wire = module()
    rgb = io.BytesIO()
    Image.new("RGB", (2, 2)).save(rgb, format="PNG")
    row = {"file_name": "flying_wire/2_cable/0001.png", "scene_family": "flying_wire",
           "image": {"bytes": rgb.getvalue()}, "masks_rle": "[]", "num_instances": 0}
    depth = tmp_path / "flying_wire/2_cable/0001.npy"
    depth.parent.mkdir(parents=True)
    np.save(depth, np.ones((2, 2), dtype=np.float32))
    result = wire.load_wireseg_sample(row, tmp_path, depth_members=set())
    assert result.depth_m is None and result.pair_sha256 is None


def test_cli_rejects_claimed_revision_with_wrong_blob_catalogue(tmp_path):
    from scripts.pretrain_dh116_wireseg import verified_sources

    wire = module()
    path = tmp_path / "hf_revision_manifest.json"
    path.write_text(json.dumps({"id": wire.WIRESEG_REPOSITORY, "sha": wire.WIRESEG_REVISION,
                                "siblings": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="blob catalogue"):
        verified_sources(tmp_path)
