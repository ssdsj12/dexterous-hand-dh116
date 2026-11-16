"""Local preview-only adaptation for IsaacLab's optional toolbar widgets.

No SDK installation is patched. Absent buttons stay absent; existing buttons
retain IsaacLab's hide/disable behavior. This owns no simulation or physics.
"""
ADAPTER_VERSION = "dh116-preview-owned-launcher-v2"

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

def install_fatal_defaults(signal_api):
    """Let genuine native fatal signals terminate, without reentering app.close.

    Only the standalone preview process uses this; it never converts a signal
    into success or sends a synthetic signal to terminate a healthy process.
    """
    for name in ("SIGSEGV","SIGABRT","SIGBUS"):
        signal_api.signal(getattr(signal_api,name),signal_api.SIG_DFL)


class ManagedPreviewApp:
    """Own launcher callbacks explicitly; one normal close attempt, no forced exit."""
    def __init__(self,launcher,*,signal_api):
        self._launcher=launcher
        self._app=launcher.app
        self._signals=signal_api
        self._closed=False
        self._close_error=None
        if signal_api is not None:install_fatal_defaults(signal_api)

    def update(self):
        if self._closed:raise RuntimeError("preview app is closed")
        return self._app.update()

    def close(self,**kwargs):
        if self._closed:
            if self._close_error is not None:raise RuntimeError(self._close_error)
            return
        self._closed=True
        launcher,app=self._launcher,self._app
        errors=[]
        try:
            for name in ("_hide_play_button_callback","_unhide_play_button_callback"):
                subscription=getattr(launcher,name,None)
                try:
                    if subscription is not None:subscription.unsubscribe()
                except BaseException as error:errors.append(name+": "+repr(error))
                finally:
                    setattr(launcher,name,None)
                    subscription=None
            if self._signals is not None:
                for name in ("SIGTERM","SIGABRT","SIGSEGV","SIGBUS"):
                    number=getattr(self._signals,name)
                    handler=None
                    try:
                        handler=self._signals.getsignal(number)
                        if getattr(handler,"__self__",None) is launcher:
                            self._signals.signal(number,self._signals.SIG_DFL)
                    except BaseException as error:errors.append(name+": "+repr(error))
                    finally:handler=None
            try:app.close(**kwargs)
            except BaseException as error:errors.append("app.close: "+repr(error))
        finally:
            launcher._app=None
            self._app=self._launcher=self._signals=None
            launcher=app=None
        if errors:
            self._close_error="; ".join(errors)
            raise RuntimeError(self._close_error)


def launch_managed_preview_app(launcher_factory,contract):
    import signal
    kwargs=dict(contract["launcher"])
    kwargs.update(experience=contract["experience"]["path"],fast_shutdown=False)
    launcher=preview_launcher_type(launcher_factory)(**kwargs)
    return ManagedPreviewApp(launcher,signal_api=signal)
