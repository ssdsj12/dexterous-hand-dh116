"""Owned cleanup for the installed Isaac 4.5 CameraView layout.

Cleanup is conservative: errors are reported, never retried, and the rig is
retained if a preceding step failed. Native teardown still needs validation.
"""
import uuid

RIG_PATH = "/World/DH116StereoRGB"
_MARKER = "dh116RgbCaptureOwner"


class RGBResources:
    def __init__(self):
        self.stage = None
        self.token = None
        self.view = None
        self.attachments = []
        self.unsafe = False
        self.closed = False
        self.error = None

    def define_rig(self, stage, define):
        if stage.GetPrimAtPath(RIG_PATH).IsValid():
            raise RuntimeError("stereo RGB rig already exists")
        define(stage, RIG_PATH)
        prim = stage.GetPrimAtPath(RIG_PATH)
        token = uuid.uuid4().hex
        # The marker is the proof needed to remove a prim. If authoring fails,
        # leave it in place rather than claim ownership.
        prim.SetCustomDataByKey(_MARKER, token)
        self.stage, self.token = stage, token

    def bind_view(self, view):
        # The installed CameraView destructor handles _tiled_render_product,
        # but its setup actually uses _render_product and _annotators.
        self.unsafe = True
        if (getattr(view, "_tiled_render_product", object()) is not None
                or not isinstance(getattr(view, "_annotators", None), dict)
                or getattr(view, "_render_product", None) is None
                or view.get_render_product_path() != view._render_product.path):
            raise RuntimeError("unsupported CameraView resource layout")
        self.view = view
        self.unsafe = False

    def attach(self, annotator, path, do_attach):
        # An attach can throw after side effects, so mark the attempt first.
        self.attachments.append((annotator, path))
        do_attach()

    def close(self):
        if self.closed:
            if self.error: raise RuntimeError(self.error)
            return
        self.closed = True
        errors = []
        if self.unsafe:
            errors.append("unknown camera resources")
        for annotator, path in reversed(self.attachments):
            try: annotator.detach([path])
            except Exception as exc: errors.append("annotator detach: " + repr(exc))
        if self.view is not None:
            view = self.view
            for annotator in view._annotators.values():
                try: annotator.detach([view._render_product_path])
                except Exception as exc: errors.append("camera annotator detach: " + repr(exc))
            if not errors:
                try:
                    view._render_product.destroy()
                    view._render_product = None
                    view._render_product_path = None
                    view._annotators.clear()
                except Exception as exc:
                    errors.append("render product destroy: " + repr(exc))
        if self.stage is not None and not errors:
            try:
                prim = self.stage.GetPrimAtPath(RIG_PATH)
                if not prim.IsValid() or prim.GetCustomDataByKey(_MARKER) != self.token:
                    raise RuntimeError("rig ownership marker changed")
                if not self.stage.RemovePrim(RIG_PATH):
                    raise RuntimeError("rig removal failed")
            except Exception as exc:
                errors.append("rig removal: " + repr(exc))
        if errors:
            self.error = "; ".join(errors)
            raise RuntimeError(self.error)
