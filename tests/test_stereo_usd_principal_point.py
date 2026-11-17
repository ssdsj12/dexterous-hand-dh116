"""CPU USD/Gf projection checks; SDK boundaries are substitutes, never native proof."""
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from pxr import Gf, Usd, UsdGeom
import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_stereo_calibration as lens
from go2_pvcnn.control.dh116 import tabletop_stereo_camera as camera_module


ROOT = Path(__file__).resolve().parents[1]
RESOLUTIONS = [(224, 96), (320, 240), (640, 360), (640, 480)]


def calibration(width, height, off_centre=True):
    original = lens.StereoCalibration.from_json(
        ROOT / "configs/dh116_training/stereo_camera_v1.json")
    data = original.to_dict()
    data.update(width=640, height=480, rectified=False)
    data["K_left"] = [[400., 0., 350. if off_centre else 320.],
                      [0., 400., 260. if off_centre else 240.], [0., 0., 1.]]
    data["K_right"] = [[430., 0., 305. if off_centre else 320.],
                       [0., 370., 225. if off_centre else 240.], [0., 0., 1.]]
    return lens.resize_calibration(lens.StereoCalibration.from_dict(data), width, height)


def cpu_reader(monkeypatch, c, *, sdk_focal_scale=1.):
    """Run the real author/readback against real USD; replace only Isaac services."""
    stage = Usd.Stage.CreateInMemory()
    calls = []
    class View:
        def __init__(self, **kwargs):
            assert kwargs["camera_resolution"] == (c.width, c.height)
            assert kwargs["output_annotators"] == ["rgb"]
            assert kwargs["reset_xform_properties"] is False
            # Deliberately reverse slots, to exercise actual numeric reordering.
            self.prim_paths = sorted(
                (str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)),
                reverse=True)
            self.tiled_resolution = (len(self.prim_paths)*c.width, c.height)

        def cameras(self):
            return [UsdGeom.Camera(stage.GetPrimAtPath(p)) for p in self.prim_paths]

        def get_world_poses(self, **kwargs):
            assert kwargs == dict(camera_axes="usd", usd=True)
            poses = [Gf.Transform(cam.GetCamera(Usd.TimeCode.Default()).transform)
                     for cam in self.cameras()]
            rotations = [pose.GetRotation().GetQuat() for pose in poses]
            return (np.array([tuple(pose.GetTranslation()) for pose in poses]),
                    np.array([[q.GetReal(), *q.GetImaginary()] for q in rotations]))

        def get_focal_lengths(self):
            # CameraView's getter unit is one tenth of the USD focalLength unit.
            return np.array([cam.GetFocalLengthAttr().Get()/10.*sdk_focal_scale
                             for cam in self.cameras()])

        def get_horizontal_apertures(self):
            return np.array([cam.GetHorizontalApertureAttr().Get() for cam in self.cameras()])

        def get_vertical_apertures(self):
            return np.array([cam.GetVerticalApertureAttr().Get() for cam in self.cameras()])

        def get_render_product_path(self):
            return "/CPU/UnusedRenderProduct"

    manager = SimpleNamespace(set_extension_enabled_immediate=lambda *args: calls.append(args))
    substitute = {
        "omni.kit.app": SimpleNamespace(get_app=lambda: SimpleNamespace(
            get_extension_manager=lambda: manager)),
        "isaacsim.sensors.camera": SimpleNamespace(CameraView=View),
        "omni.usd": SimpleNamespace(get_context=lambda: SimpleNamespace(get_stage=lambda: stage)),
        "omni.replicator.core": SimpleNamespace(AnnotatorRegistry=SimpleNamespace(
            get_annotator=lambda *args, **kwargs: SimpleNamespace(attach=lambda *args: None))),
        "isaacsim.core.nodes.bindings._isaacsim_core_nodes":
            SimpleNamespace(acquire_interface=lambda: object()),
    }
    real_import = importlib.import_module
    def substitute_import(name, package=None):
        return substitute[name] if name in substitute else real_import(name, package)
    monkeypatch.setattr(camera_module.importlib, "import_module", substitute_import)
    origins = torch.tensor([[0., 0., 0.], [1.2, -.3, .1]], dtype=torch.float64)
    reader = camera_module._NativeRGBPairReader(None, origins, c, "cpu")
    assert calls == [("isaacsim.sensors.camera", True)]
    return reader, stage, origins


def assert_projection(camera, optical, k, width, height):
    view = camera.frustum.ComputeViewMatrix()
    projection = camera.frustum.ComputeProjectionMatrix()
    for xyz in ((0., 0., 1.), (.03, -.02, .7), (-.1, .08, 1.3)):
        world = optical @ np.array([*xyz, 1.])
        ndc = projection.Transform(view.Transform(Gf.Vec3d(*world[:3].tolist())))
        actual = np.array([(ndc[0]+1)*width/2, (1-ndc[1])*height/2])
        expected = np.array([float(k[0, 0])*xyz[0]/xyz[2]+float(k[0, 2]),
                             float(k[1, 1])*xyz[1]/xyz[2]+float(k[1, 2])])
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=0)


@pytest.mark.parametrize("width,height", RESOLUTIONS)
@pytest.mark.parametrize("off_centre", [False, True])
def test_real_author_projects_both_resized_independent_intrinsics(
        monkeypatch, width, height, off_centre):
    c = calibration(width, height, off_centre)
    reader, stage, origins = cpu_reader(monkeypatch, c)
    for env, origin in enumerate(origins.numpy()):
        for side, (k, transform) in enumerate(((c.K_left, c.T_world_left),
                                              (c.K_right, c.T_world_right))):
            prim = UsdGeom.Camera(stage.GetPrimAtPath(
                f"/World/DH116StereoRGB/camera_{2*env+side}"))
            gf = prim.GetCamera(Usd.TimeCode.Default())
            optical = transform.numpy().copy()
            optical[:3, 3] += origin
            # Test the production optical/world-to-USD transform independently.
            np.testing.assert_allclose(np.asarray(gf.transform).T,
                optical @ np.diag([1., -1., -1., 1.]), atol=1e-12, rtol=0)
            assert gf.focalLength == 24.
            assert_projection(gf, optical, k, width, height)
            if not off_centre:
                assert gf.horizontalApertureOffset == gf.verticalApertureOffset == 0.
    actual_k, actual_t = reader._readback()
    for side, (k, transform) in enumerate(((c.K_left, c.T_world_left),
                                          (c.K_right, c.T_world_right))):
        np.testing.assert_allclose(actual_k[:, side], k.numpy()[None].repeat(2, axis=0),
                                   atol=2e-4, rtol=0)
        expected = transform.numpy()[None].repeat(2, axis=0)
        expected[:, :3, 3] += origins.numpy()
        np.testing.assert_allclose(actual_t[:, side], expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("width,height", RESOLUTIONS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_pure_helpers_match_actual_gf_projection(width, height, side):
    assert hasattr(lens, "usd_aperture_offsets"), "USD projection helpers missing"
    c = calibration(width, height)
    k = c.K_left if side == "left" else c.K_right
    optical = (c.T_world_left if side == "left" else c.T_world_right).numpy()
    horizontal, vertical = lens.usd_aperture_offsets(k, width, height, 24.)
    gf = Gf.Camera()
    gf.focalLength = 24.
    gf.horizontalAperture = width*24./float(k[0, 0])
    gf.verticalAperture = height*24./float(k[1, 1])
    gf.horizontalApertureOffset = horizontal
    gf.verticalApertureOffset = vertical
    gf.transform = Gf.Matrix4d(*(optical @ np.diag([1., -1., -1., 1.])).T.reshape(-1))
    assert_projection(gf, optical, k, width, height)
    readback = lens.usd_aperture_intrinsics(width, height, gf.focalLength,
        gf.horizontalAperture, gf.verticalAperture,
        gf.horizontalApertureOffset, gf.verticalApertureOffset)
    torch.testing.assert_close(readback, k, atol=2e-4, rtol=0)


def test_original_default_digest_and_centre_output_are_preserved(monkeypatch):
    c = lens.StereoCalibration.from_json(ROOT / "configs/dh116_training/stereo_camera_v1.json")
    assert c.sha256 == "0d96cdb45d30a4ed36f5d2f0a8d263a296d33122a3fda2f67536b8f0025f538f"
    reader, stage, _ = cpu_reader(monkeypatch, c)
    for path in reader.paths:
        gf = UsdGeom.Camera(stage.GetPrimAtPath(path)).GetCamera(Usd.TimeCode.Default())
        assert gf.horizontalApertureOffset == gf.verticalApertureOffset == 0.
        assert gf.focalLength == 24.
        assert gf.horizontalAperture == pytest.approx(c.width*24./float(c.K_left[0, 0]))
        assert gf.verticalAperture == pytest.approx(c.height*24./float(c.K_left[1, 1]))
    reader._readback()


def test_bad_sdk_focal_units_fail_readback_instead_of_relabelling(monkeypatch):
    reader, _, _ = cpu_reader(monkeypatch, calibration(640, 480), sdk_focal_scale=.1)
    with pytest.raises(ValueError, match="readback mismatch"):
        reader._readback()


@pytest.mark.parametrize("matrix_case", ["shape", "nan", "zero_fx", "negative_fy",
                                        "skew_xy", "skew_yx", "last_row"])
def test_offsets_reject_malformed_intrinsic_matrix(matrix_case):
    assert hasattr(lens, "usd_aperture_offsets"), "USD projection helpers missing"
    k = calibration(640, 480).K_left.clone()
    if matrix_case == "shape":
        k = k[:2]
    elif matrix_case == "nan":
        k[0, 2] = float("nan")
    elif matrix_case == "zero_fx":
        k[0, 0] = 0.
    elif matrix_case == "negative_fy":
        k[1, 1] = -1.
    elif matrix_case == "skew_xy":
        k[0, 1] = 1.
    elif matrix_case == "skew_yx":
        k[1, 0] = 1.
    else:
        k[2, 2] = 2.
    with pytest.raises(ValueError):
        lens.usd_aperture_offsets(k, 640, 480, 24.)


@pytest.mark.parametrize("width,height", [(0, 480), (640, -1), (True, 480),
                                          (640., 480), (640, 480.)])
@pytest.mark.parametrize("operation", ["offsets", "intrinsics"])
def test_lens_helpers_reject_invalid_dimensions(width, height, operation):
    assert hasattr(lens, "usd_aperture_offsets"), "USD projection helpers missing"
    with pytest.raises(ValueError):
        if operation == "offsets":
            lens.usd_aperture_offsets(torch.eye(3), width, height, 24.)
        else:
            lens.usd_aperture_intrinsics(width, height, 24., 38.4, 28.8, 0., 0.)


@pytest.mark.parametrize("focal", [0., -1., float("nan"), float("inf")])
@pytest.mark.parametrize("operation", ["offsets", "intrinsics"])
def test_lens_helpers_reject_invalid_focal_length(focal, operation):
    assert hasattr(lens, "usd_aperture_offsets"), "USD projection helpers missing"
    with pytest.raises(ValueError):
        if operation == "offsets":
            lens.usd_aperture_offsets(torch.eye(3), 640, 480, focal)
        else:
            lens.usd_aperture_intrinsics(640, 480, focal, 38.4, 28.8, 0., 0.)


@pytest.mark.parametrize("index,value", [(0, 0.), (1, -1.), (0, float("inf")),
    (1, float("nan")), (2, float("nan")), (3, float("inf"))])
def test_intrinsics_reject_invalid_aperture_or_offsets(index, value):
    assert hasattr(lens, "usd_aperture_intrinsics"), "USD projection helpers missing"
    values = [38.4, 28.8, -1.8, 1.2]
    values[index] = value
    with pytest.raises(ValueError):
        lens.usd_aperture_intrinsics(640, 480, 24., *values)


def test_camera_and_lens_import_without_isaac():
    code = """
import importlib
import sys
class RejectNative:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('omni', 'isaacsim', 'isaaclab'):
            raise AssertionError('unexpected native import: ' + fullname)
sys.meta_path.insert(0, RejectNative())
m = importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_calibration')
importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_camera')
assert m.usd_aperture_offsets([[400,0,350],[0,400,260],[0,0,1]],640,480,24.) == (-1.8,1.2)
assert not any(name.split('.')[0] in ('omni', 'isaacsim', 'isaaclab') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
