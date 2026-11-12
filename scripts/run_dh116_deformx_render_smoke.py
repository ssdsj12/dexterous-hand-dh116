#!/usr/bin/env python3
"""Capture one bounded DeformX asset render using the existing Isaac installation."""

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

if __package__:
    from .audit_dh116_deformx_assets import _sha256
else:
    from audit_dh116_deformx_assets import _sha256


UPSTREAM = "04f6a11eb4d832df761f631cf6ce1bc00da75eb1"
CASES = {
    0: ("usd/rod_drop_multi_2_plane.usdc", "wires_traj_data/drop_n2_100.npz"),
    3: ("usd/rod_hang_flying.usdc", "wires_traj_data/hang_n2_100.npz"),
}


def _write_json(path, document):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def inspect_render_products(capture, *, width, height):
    from PIL import Image

    candidates = {
        "RGB": list(capture.glob("**/rgb/rgb_*.png")),
        "radial depth": list(capture.glob("**/depth/distance_to_camera_*.npy")),
        "native instance ID": list(capture.glob("**/seg/native_instance_id.npy")),
        "native instance mapping": list(capture.glob("**/seg/native_instance_id_labels.json")),
    }
    for name, paths in candidates.items():
        if len(paths) != 1:
            raise ValueError(f"expected exactly one {name} file, found {len(paths)}")
    rgb, depth, labels, mapping = (paths[0] for paths in candidates.values())
    with Image.open(rgb) as image:
        image.load()
        if image.size != (width, height) or image.mode not in ("RGB", "RGBA"):
            raise ValueError("RGB product geometry or mode differs")
    distances = np.load(depth, allow_pickle=False)
    if (distances.shape != (height, width) or distances.dtype.kind != "f"
            or not np.any(np.isfinite(distances) & (distances > 0))):
        raise ValueError("invalid distance_to_camera capture")
    ids = np.load(labels, allow_pickle=False)
    if ids.shape != (height, width) or ids.dtype != np.uint32:
        raise ValueError("expected numeric uint32 native instance IDs")
    names = json.loads(mapping.read_text(encoding="utf-8"))
    present = set(map(str, np.unique(ids)))
    wire_labels = [key for key, value in names.items()
                   if key in present and "/World/Wire_" in json.dumps(value)]
    if not wire_labels:
        raise ValueError("no captured native instance ID resolves to a wire")
    result = {}
    for field, path, encoding in (
        ("rgb", rgb, "png"),
        ("depth", depth, "distance_to_camera_m"),
        ("native_id", labels, "uint32"),
        ("native_id_mapping", mapping, "idToLabels_json"),
    ):
        result[field] = {
            "file": path.relative_to(capture).as_posix(),
            "encoding": encoding,
            "sha256": _sha256(path),
        }
    result["native_id"]["wire_label_count"] = len(wire_labels)
    return result


def read_single_annotation(rep, app, products, name, *, init_params=None):
    if len(products) != 1:
        raise ValueError("bounded render requires exactly one product")
    product_path = products[0].path
    annotator = rep.AnnotatorRegistry.get_annotator(name, init_params=init_params)
    annotator.attach(product_path)
    try:
        rep.orchestrator.step(rt_subframes=1)
        app.update()
        return annotator.get_data(device="cpu", do_array_copy=True)
    finally:
        annotator.detach(product_path)


def _capture_stage(stage, output):
    from pxr import Usd, UsdGeom, UsdLux

    axis = str(UsdGeom.GetStageUpAxis(stage))
    units = float(UsdGeom.GetStageMetersPerUnit(stage))
    camera_paths, wire_meshes, domes = [], {}, {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        name = str(prim.GetPath())
        if name.startswith("/World/_CaptureProxies/") and prim.IsA(UsdGeom.Camera):
            camera_paths.append(name)
        if name.startswith("/World/Wire_") and prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            if points.shape != (2652, 3) or not np.isfinite(points).all():
                raise ValueError(f"unexpected wire mesh vertex data: {name}")
            wire_meshes[name] = len(points)
        if prim.IsA(UsdLux.DomeLight):
            texture = UsdLux.DomeLight(prim).GetTextureFileAttr().Get()
            domes[name] = None if texture is None else texture.resolvedPath
    wire_roots = {path.split("/")[2] for path in wire_meshes}
    if (axis != "Z" or units != 1.0 or len(camera_paths) != 1
            or wire_roots != {"Wire_0", "Wire_1"} or not domes.get("/World/DomeLight")):
        raise ValueError("composed stage camera, lighting, wire geometry or units failed")
    snapshot = output / "composed_stage.usdc"
    if not stage.Flatten().Export(str(snapshot)):
        raise ValueError("failed to persist composed stage")
    _write_json(output / "native_stage.json", {
        "stage": snapshot.name,
        "stage_sha256": _sha256(snapshot),
        "up_axis": axis,
        "meters_per_unit": units,
        "camera_paths": camera_paths,
        "wire_meshes": wire_meshes,
        "dome_textures": domes,
    })


def _preflight(args):
    source = args.upstream.resolve(strict=True)
    head = subprocess.check_output(
        ["git", "-c", f"safe.directory={source}", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != UPSTREAM:
        raise ValueError(f"upstream DeformX commit differs: {head}")
    prepared = args.prepared.resolve(strict=True)
    manifest_path = prepared / "render_scene_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not data["derived_authored_usd_dependencies_closed"]:
        raise ValueError("prepared USD dependency closure is missing")
    assets = prepared / "assets"
    scene, trajectory = CASES[args.variant]
    if not data["overrides"][scene]["only_locked_default_changed"]:
        raise ValueError("required DomeLight override has no authored comparison")
    for name, row in data["original_inventory"]["files"].items():
        asset = (assets / name).resolve(strict=True)
        if not asset.is_relative_to(assets) or asset != assets / name:
            raise ValueError(f"invalid prepared asset path: {name}")
        expected = data["overrides"].get(name, {}).get("derived_sha256", row["sha256"])
        if _sha256(asset) != expected:
            raise ValueError(f"prepared asset drift: {name}")
    output = args.output.resolve()
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (source, prepared)):
        raise ValueError("output must be independent of source and prepared assets")
    if output.exists() or args.output.is_symlink():
        raise FileExistsError(output)
    return source, assets, output, {
        "variant": args.variant,
        "frame": 0,
        "seed": 42,
        "cameras": 1,
        "accumulation": [1, 1],
        "scene": scene,
        "trajectory": trajectory,
        "prepared_manifest_sha256": _sha256(manifest_path),
        "upstream_commit": head,
    }


def _run(source, assets, output, plan):
    os.environ["DEFORMX_ROOT"] = str(source)
    os.environ["DEFORMX_ASSET_ROOT"] = str(assets)
    os.environ["DEFORMX_OUTPUT_ROOT"] = str(output)
    sys.path.insert(0, str(source))
    from dataset_generator.drop_hang.config import RenderConfig, default_config
    from dataset_generator.drop_hang.generator import DatasetGenerator
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
    from isaacsim import SimulationApp

    class EvidenceGenerator(DatasetGenerator):
        def __init__(self, app, cfg):
            super().__init__(app, cfg)
            self._owned_products = ()
            self._deferred_scene_close = False

        def _write_one_rgb(self, products, directory):
            from PIL import Image

            self._owned_products = tuple(products)
            started = time.time()
            rgb = read_single_annotation(self._rep(), self._sim_app, products, "rgb")
            Image.fromarray(np.asarray(rgb)).save(Path(directory) / "rgb_0000.png")
            return started

        def _write_one_seg(self, products, directory):
            started = time.time()
            packet = read_single_annotation(
                self._rep(), self._sim_app, products, "instance_id_segmentation_fast",
                init_params={"colorize": False})
            ids = np.asarray(packet["data"])
            if ids.dtype.kind not in "iu" or ids.dtype.itemsize != 4:
                raise ValueError("native ID annotator did not supply 32-bit integers")
            ids = ids.view(np.uint32).reshape(cfg.capture_height, cfg.capture_width)
            np.save(Path(directory) / "native_instance_id.npy", ids)
            _write_json(Path(directory) / "native_instance_id_labels.json",
                        {str(k): v for k, v in packet["info"]["idToLabels"].items()})
            return started

        def _write_one_depth(self, products, directory):
            started = time.time()
            depth = read_single_annotation(self._rep(), self._sim_app, products, "distance_to_camera")
            values = np.asarray(depth).reshape(cfg.capture_height, cfg.capture_width)
            np.save(Path(directory) / "distance_to_camera_0000.npy", values)
            stage = self._omni_usd().get_context().get_stage()
            _capture_stage(stage, output)
            return started

        def _close_scene(self):
            # The upstream caller still holds the native stage and render products.
            self._deferred_scene_close = True

        def close_owned_scene(self):
            if not self._deferred_scene_close:
                return
            try:
                rep = self._rep()
                rep.orchestrator.stop()
                rep.orchestrator.set_capture_on_play(False)
                rep.orchestrator.wait_until_complete()
                for product in self._owned_products:
                    product.destroy()
                self._owned_products = ()
                self._sim_app.update()
            finally:
                super()._close_scene()

    cfg = default_config()
    spec = cfg.scene_variant_specs[plan["variant"]]
    scene, trajectory, wires = spec
    if Path(scene) != assets / plan["scene"] or Path(trajectory) != assets / plan["trajectory"] or wires != 2:
        raise ValueError("upstream render variant differs from bounded plan")
    capture = output / "captures"
    cfg = replace(cfg, scene_variant_specs=(spec,), scene_usd=scene, npz_path=trajectory,
                  num_wires=wires, capture_out_dir=str(capture), cams_sample_per_frame=1)
    with native_instance_lock():
        output.mkdir(parents=True)
        _write_json(output / "render_plan.json", plan)
        app = SimulationApp({"headless": True, "fast_shutdown": False})
        generator = None
        try:
            generator = EvidenceGenerator(app, cfg)
            generator.run([0], RenderConfig(do_seg=True, do_depth=True, seed=42, accum_steps=1,
                                            accum_subframes=1, frame_end=0))
            products = inspect_render_products(capture, width=cfg.capture_width,
                                               height=cfg.capture_height)
        except Exception:
            _write_json(output / "native_failure.json", {"traceback": traceback.format_exc()})
            raise
        finally:
            try:
                if generator is not None:
                    generator.close_owned_scene()
            finally:
                generator = None
                gc.collect()
                _write_json(output / "native_cleanup_started.json", {
                    "capture_queue": "synchronous_direct_annotators",
                    "stage_close": "after_render_frame_return",
                    "wait_for_replicator": False,
                })
                app.close(wait_for_replicator=False)
        _write_json(output / "completion.json", {
            "products": products,
            "app_close_returned": True,
            "outer_exit_verified": False,
            "native_render_verified": False,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", type=int, choices=CASES, required=True)
    parser.add_argument("--execute-native", action="store_true")
    args = parser.parse_args()
    source, assets, output, plan = _preflight(args)
    if not args.execute_native:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    _run(source, assets, output, plan)


if __name__ == "__main__":
    main()
