"""Local preview-only adaptation for IsaacLab's optional toolbar widgets.

No SDK installation is patched. Absent buttons stay absent; existing buttons
retain IsaacLab's hide/disable behavior. This owns no simulation or physics.
"""
ADAPTER_VERSION = "dh116-preview-optional-toolbar-v1"

def _native_toolbar():
    import omni.kit.widget.toolbar
    return omni.kit.widget.toolbar.get_instance()

def preview_launcher_type(base, *, toolbar_getter=None):
    if not isinstance(base,type):
        raise TypeError("AppLauncher class required")
    getter = _native_toolbar if toolbar_getter is None else toolbar_getter
    class PreviewLauncher(base):
        def __init__(self, **kwargs):
            try:
                super().__init__(**kwargs)
            except BaseException as error:
                # The upstream constructor can fail after SimulationApp exists.
                # Attempt normal close once and preserve the original failure.
                partial=getattr(self,"_app",None)
                try:
                    if partial is not None:
                        partial.close(wait_for_replicator=False)
                except BaseException as cleanup:
                    error.preview_startup_cleanup_error = str(cleanup)
                finally:
                    self._app = None
                    partial = None
                raise

        def _group(self):
            if self._livestream < 1 and self._headless:
                return None
            ui=getter()
            builtin=getattr(ui,"_builtin_tools",None)
            return getattr(builtin,"_play_button_group",None)

        def _hide_stop_button(self):
            group=self._group()
            stop=getattr(group,"_stop_button",None)
            if stop is not None:
                stop.visible=False
                stop.enabled=False
                group._stop_button=None

        def _hide_play_button(self, flag):
            group=self._group()
            play=getattr(group,"_play_button",None)
            if play is not None:
                play.visible=not flag
                play.enabled=not flag
    return PreviewLauncher
