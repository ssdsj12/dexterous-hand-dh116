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


def test_wireseg_subset_rejects_unverified_sources_and_wrong_split(monkeypatch, tmp_path):
    from scripts import pretrain_dh116_wireseg as cli

    train_blob = 'data/train-a.parquet'
    validation_blob = 'data/train-b.parquet'
    blobs = {train_blob: 'a' * 64, validation_blob: 'b' * 64}
    train_row = dict(file_name='flying_wire/2_cable/1.png', scene_family='flying_wire')
    val_row = dict(file_name='wire_on_plane/2_cable/2.png', scene_family='wire_on_plane')

    def rows(root, names, split, *, include_source=False, metadata_only=False):
        for name, row in ((train_blob, train_row), (validation_blob, val_row)):
            if name in names and module().sample_split(row['scene_family']) == split:
                yield (name, row) if include_source else row

    monkeypatch.setattr(cli, 'wireseg_rows', rows)
    subset = cli.make_subset(tmp_path, blobs, 'f' * 64, 'train', 1)
    path = tmp_path / 'subset.json'
    path.write_text(json.dumps(subset))
    assert list(cli.read_subset(tmp_path, path, blobs, 'f' * 64, 'train')) == [train_row]
    with pytest.raises(ValueError, match='split'):
        list(cli.read_subset(tmp_path, path, blobs, 'f' * 64, 'validation'))
    with pytest.raises(ValueError, match='verified'):
        list(cli.read_subset(tmp_path, path, {}, 'f' * 64, 'train'))


def test_wireseg_resume_requires_the_same_mode_and_subset(tmp_path):
    from scripts import pretrain_dh116_wireseg as cli

    path = tmp_path / 'resume.pt'
    contract = {'mode': 'seg-only', 'train_subset_sha256': 'a' * 64}
    torch.save({'contract': contract}, path)
    with pytest.raises(ValueError, match='contract'):
        cli.restore_training(path, {'mode': 'seg-depth'}, None, None, torch.device('cpu'))


def test_wireseg_resume_preserves_adam_steps_and_deterministic_update(tmp_path):
    from scripts import pretrain_dh116_wireseg as cli

    model = torch.nn.Linear(2, 1)
    adam = torch.optim.AdamW(model.parameters(), lr=0.01)
    device = torch.device('cpu')
    contract = {'mode': 'seg-only', 'train_subset_sha256': 'a' * 64}
    model.transfer_state = lambda: {}
    x = torch.ones((1, 2))

    def update(net, optimizer):
        optimizer.zero_grad(set_to_none=True)
        net(x * torch.rand_like(x)).square().sum().backward()
        optimizer.step()

    update(model, adam)
    path = tmp_path / 'resume.pt'
    cli.save_training(path, model, adam, device, contract, [{'file_name': 'row1'}], {'rgb': 1})
    update(model, adam)
    resumed = torch.nn.Linear(2, 1)
    resumed_adam = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    step, rows, counts = cli.restore_training(path, contract, resumed, resumed_adam, device)
    assert (step, rows, counts) == (1, [{'file_name': 'row1'}], {'rgb': 1})
    update(resumed, resumed_adam)
    assert all(torch.equal(left, right) for left, right in zip(model.parameters(), resumed.parameters()))
