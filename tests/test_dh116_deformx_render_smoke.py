import json

import numpy as np
import pytest

from scripts.run_dh116_deformx_render_smoke import inspect_render_products


def test_products_require_distinct_numeric_native_id_and_radial_depth(tmp_path):
    from PIL import Image

    capture = tmp_path / "captures" / "frame"
    for folder in ("rgb", "depth", "seg"):
        (capture / folder).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), (2, 3, 4)).save(capture / "rgb/rgb_0000.png")
    np.save(capture / "depth/distance_to_camera_0000.npy",
            np.full((3, 4), 0.6, dtype=np.float32))
    np.save(capture / "seg/native_instance_id.npy", np.full((3, 4), 2, dtype=np.uint32))
    labels = capture / "seg/native_instance_id_labels.json"
    labels.write_text(json.dumps({"2": "/World/Wire_0/Mesh"}))

    products = inspect_render_products(capture.parent, width=4, height=3)
    assert products["depth"]["encoding"] == "distance_to_camera_m"
    assert products["native_id"]["encoding"] == "uint32"
    assert products["native_id"]["wire_label_count"] == 1
    assert len(products["native_id"]["sha256"]) == 64

    np.save(capture / "seg/native_instance_id.npy", np.ones((3, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint32"):
        inspect_render_products(capture.parent, width=4, height=3)


def test_multiple_captures_cannot_pass_as_one_frame(tmp_path):
    from PIL import Image

    (tmp_path / "rgb").mkdir()
    for i in range(2):
        Image.new("RGB", (4, 3)).save(tmp_path / "rgb" / f"rgb_{i}.png")
    with pytest.raises(ValueError, match="exactly one RGB"):
        inspect_render_products(tmp_path, width=4, height=3)


def test_annotation_owns_stable_render_product_path_on_exception():
    from types import SimpleNamespace

    from scripts import run_dh116_deformx_render_smoke as smoke

    assert hasattr(smoke, "read_single_annotation")
    events = []

    def get_data(**kwargs):
        assert kwargs == {"device": "cpu", "do_array_copy": True}
        raise ValueError("sample failure")

    annotator = SimpleNamespace(
        attach=lambda path: events.append(("attach", path)),
        detach=lambda path: events.append(("detach", path)),
        get_data=get_data,
    )
    rep = SimpleNamespace(
        AnnotatorRegistry=SimpleNamespace(get_annotator=lambda *args, **kwargs: annotator),
        orchestrator=SimpleNamespace(step=lambda **kwargs: events.append(("step", kwargs))),
    )
    app = SimpleNamespace(update=lambda: events.append(("update",)))
    product = SimpleNamespace(path="/Render/OwnedProduct")

    with pytest.raises(ValueError, match="sample failure"):
        smoke.read_single_annotation(rep, app, [product], "rgb")
    assert events == [
        ("attach", product.path), ("step", {"rt_subframes": 1}), ("update",),
        ("detach", product.path),
    ]
