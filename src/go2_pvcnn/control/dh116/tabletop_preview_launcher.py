"""Local preview-only adaptation for IsaacLab's optional toolbar widgets.

No SDK installation is patched. Absent buttons stay absent; existing buttons
retain IsaacLab's hide/disable behavior. This owns no simulation or physics.
"""
from .tabletop_preview_extension_cache import prepare_loaded_extension_cache,validate_extension_cache_receipt
from .tabletop_preview_window_owner import wrap_preview_hotkey_destroy,prepare_loaded_window_owner,validate_window_release_receipt

ADAPTER_VERSION = "dh116-preview-owned-launcher-v6"

def _native_toolbar():
    import omni.kit.widget.toolbar
    return omni.kit.widget.toolbar.get_instance()

def preview_launcher_type(base, *, toolbar_getter=None, partial_stop_presentation=None,
        partial_prepare_native_shutdown=None,partial_prepare_framework_shutdown=None):
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
                cleanup_errors=[]
                window_receipt=framework_receipt=None
                try:
                    if partial is not None:
                        if partial_stop_presentation is not None:
                            try:partial_stop_presentation()
                            except BaseException as cleanup:cleanup_errors.append(str(cleanup))
                        if partial_prepare_native_shutdown is not None:
                            try:window_receipt=partial_prepare_native_shutdown()
                            except BaseException as cleanup:cleanup_errors.append(str(cleanup))
                        if partial_prepare_framework_shutdown is not None:
                            try:framework_receipt=partial_prepare_framework_shutdown(partial)
                            except BaseException as cleanup:cleanup_errors.append(str(cleanup))
                        try:partial.close(wait_for_replicator=False)
                        except BaseException as cleanup:cleanup_errors.append(str(cleanup))
                        try:validate_extension_cache_receipt(framework_receipt)
                        except BaseException as cleanup:cleanup_errors.append(str(cleanup))
                        try:validate_window_release_receipt(window_receipt)
                        except BaseException as cleanup:cleanup_errors.append(str(cleanup))
                    if cleanup_errors:error.preview_startup_cleanup_error="; ".join(cleanup_errors)
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
    def __init__(self,launcher,*,signal_api,stop_presentation=None,prepare_native_shutdown=None,prepare_framework_shutdown=None):
        self._launcher=launcher
        self._app=launcher.app
        self._signals=signal_api
        self._closed=False
        self._close_error=None
        self._stop_presentation=stop_presentation
        self._prepare_native_shutdown=prepare_native_shutdown
        self._prepare_framework_shutdown=prepare_framework_shutdown
        self._presentation_attempted=False
        self._presentation_error=None
        self._presentation_receipt=None
        if signal_api is not None:install_fatal_defaults(signal_api)

    def update(self):
        if self._closed:raise RuntimeError("preview app is closed")
        return self._app.update()

    def stop_presentation(self):
        """One explicit owned-stream stop; a failure is sticky, never retried."""
        if self._presentation_attempted:
            if self._presentation_error:raise RuntimeError(self._presentation_error)
            return self._presentation_receipt
        self._presentation_attempted=True
        stop,self._stop_presentation=self._stop_presentation,None
        try:
            if stop is not None:self._presentation_receipt=stop()
        except BaseException as error:
            self._presentation_error=type(error).__name__+": "+str(error)
        finally:
            stop=None
        if self._presentation_error:raise RuntimeError(self._presentation_error) from None
        return self._presentation_receipt

    def close(self,**kwargs):
        if self._closed:
            if self._close_error is not None:raise RuntimeError(self._close_error)
            return
        self._closed=True
        launcher,app=self._launcher,self._app
        errors=[]
        try:
            try:self.stop_presentation()
            except BaseException as error:errors.append("presentation: "+str(error))
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
            prepare,self._prepare_native_shutdown=self._prepare_native_shutdown,None
            window_receipt=None
            try:
                if prepare is not None:window_receipt=prepare()
            except BaseException as error:errors.append("window owner preparation: "+str(error))
            finally:prepare=None
            framework_receipt=None
            prepare,self._prepare_framework_shutdown=self._prepare_framework_shutdown,None
            try:
                if prepare is not None:framework_receipt=prepare(app)
            except BaseException as error:errors.append("extension cache preparation: "+str(error))
            finally:prepare=None
            try:app.close(**kwargs)
            except BaseException as error:errors.append("app.close: "+repr(error))
            try:validate_extension_cache_receipt(framework_receipt)
            except BaseException as error:errors.append("extension cache release: "+str(error))
            try:validate_window_release_receipt(window_receipt)
            except BaseException as error:errors.append("window release evidence: "+str(error))
        finally:
            launcher._app=None
            self._app=self._launcher=self._signals=None
            launcher=app=None
        if errors:
            self._close_error="; ".join(errors)
            raise RuntimeError(self._close_error)


def launch_managed_preview_app(launcher_factory,contract,*,shutdown_output=None):
    import signal
    verify_preview_shutdown_sources()
    kwargs=dict(contract["launcher"])
    kwargs.update(experience=contract["experience"]["path"],fast_shutdown=False)
    def prepare():
        data=verify_preview_shutdown_sources()
        def record(row):
            if shutdown_output is None:return
            import json,os,time
            from pathlib import Path
            path=Path(shutdown_output)/"preview_window_release.jsonl"
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(),**row),allow_nan=False)+"\n")
                stream.flush();os.fsync(stream.fileno())
        return prepare_loaded_window_owner(data["sources"],record=record)
    def prepare_framework(app):
        try:return prepare_loaded_extension_cache(app,output=shutdown_output)
        finally:app=None
    launcher=preview_launcher_type(launcher_factory,partial_stop_presentation=stop_native_presentation,
        partial_prepare_native_shutdown=prepare,partial_prepare_framework_shutdown=prepare_framework)(**kwargs)
    return ManagedPreviewApp(launcher,signal_api=signal,stop_presentation=stop_native_presentation,
        prepare_native_shutdown=prepare,prepare_framework_shutdown=prepare_framework)


def verify_preview_shutdown_sources(root=None):
    """Read installed API/binary identities without starting Kit or changing SDK."""
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[4] if root is None else Path(root)
    data=json.loads((root/"configs/dh116_execution/preview_shutdown_api_lock_v3.json").read_text())
    expected={"manager","simulation_app","timeline","livestream_stub","livestream_binding","livestream_plugin","timeline_python","hotkeys_trigger","hotkeys_extension","extension_loader","appwindow_stub","appwindow_binding"}
    if data.get("version")!="dh116-owned-preview-shutdown-api-v3" or set(data.get("sources",{}))!=expected:
        raise ValueError("complete owned shutdown API contract required")
    for name,item in data["sources"].items():
        path=Path(item["path"])
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]):
            raise ValueError("installed shutdown source differs: "+name)
    return data


def quiesce_stream_interface(interface):
    """Use actual installed disable/shutdown APIs, never force the process out."""
    errors=[];result={}
    try:
        for name,args in (("enable_livestream",(False,)),("shutdown",())):
            try:
                method=getattr(interface,name)
                value=method(*args)
                result[name]=value is True
                if value is not True:raise RuntimeError("returned "+repr(value))
            except BaseException as error:
                label="disable" if name=="enable_livestream" else name
                errors.append(label+": "+type(error).__name__+": "+str(error))
            finally:
                method=None
    finally:
        interface=None
    if errors:raise RuntimeError("; ".join(errors)) from None
    return dict(result,native_thread_quiescence_qualified=False)


def stop_native_presentation():
    verify_preview_shutdown_sources()
    import omni.kit.livestream.bind as bind
    interface=bind.acquire_livestream_interface()
    try:return quiesce_stream_interface(interface)
    finally:interface=None


def commit_preview_stop(timeline,*,freeze,read_clock):
    """Dispatch real pending STOP callbacks without rendering or App.update.

    The installed Timeline.commit API applies pending changes and callbacks on
    the main thread. Manager release is still checked by the existing lifecycle.
    Never write the manager's view field or invent a successful readback.
    """
    import threading
    try:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("STOP commit requires the main thread")
        if freeze() is not False:raise RuntimeError("STOP physics freeze failed")
        before=read_clock()
        timeline.stop()
        timeline.commit()
        if timeline.is_stopped() is not True:raise RuntimeError("timeline STOP did not read back true")
        if freeze() is not False:raise RuntimeError("STOP physics freeze failed after commit")
        after=read_clock()
        # A weakref becoming dead is allowed; creation or changed SDK clocks is not.
        if after is not None and after!=before:raise RuntimeError("STOP commit changed physical clock")
        return dict(stop_commit_returned=True,clock_before=before,clock_after=after,
            app_updates=0,explicit_physics_steps_requested=0,
            physical_clock_unchanged=(True if before is not None and after==before else None),
            native_release_verified=False)
    finally:
        timeline=freeze=read_clock=None


def finish_preview_resources(app,camera,render,holder,journal):
    """Stop presentation before removing any owned render product or stage prim."""
    errors=[]
    try:
        operations=[("presentation",app.stop_presentation)]
        if camera is not None:operations.append(("camera",camera.close))
        if render is not None:operations.append(("render",render.close))
        if holder is not None:operations.append(("holder",holder.clear))
        if journal is not None:operations.append(("journal",journal.close))
        for name,callback in operations:
            try:callback()
            except BaseException as error:errors.append(name+": "+type(error).__name__+": "+str(error))
            finally:callback=None
    finally:
        app=camera=render=holder=journal=operations=callback=None
    if errors:raise RuntimeError("; ".join(errors)) from None
