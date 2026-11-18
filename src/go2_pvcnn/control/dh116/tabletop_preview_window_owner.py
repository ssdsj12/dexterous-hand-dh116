"""Source-bound release of one existing UI owner in the private preview App.

No window is acquired/created, no SDK class is patched, and the standard
SimulationApp.close/framework unload path is retained.
"""
from pathlib import Path
import sys
import weakref


def wrap_preview_hotkey_destroy(owner,*,source,record=None):
    expected='omni.kit.hotkeys.core.trigger.HotkeyTrigger'
    name=type(owner).__module__+'.'+type(owner).__name__
    original=getattr(type(owner),'destroy',None)
    code=getattr(original,'__code__',None)
    field='_HotkeyTrigger__app_window'
    if (name!=expected or 'destroy' in vars(owner) or code is None
            or Path(code.co_filename).resolve()!=Path(source).resolve()
            or field not in vars(owner)):
        raise ValueError('unsupported owned hotkey destroy layout/source')
    reference=weakref.ref(owner)
    receipt=dict(owner_type=name,field=field,hook_installed=True,destroy_calls=0,
        sdk_destroy_returned=False,window_reference_absent=False,
        diagnostic_errors=[],native_qualification=False)

    def emit(row):
        if record is not None:
            try:record(row)
            except BaseException as error:
                # Logging must not interrupt the SDK extension shutdown chain.
                receipt['diagnostic_errors'].append(type(error).__name__+": "+str(error))

    def destroy():
        target=reference()
        if target is None:raise RuntimeError('owned UI trigger disappeared before destroy')
        receipt['destroy_calls']+=1
        errors=[]
        try:
            try:
                original(target)
                receipt['sdk_destroy_returned']=True
            except BaseException as error:errors.append(type(error).__name__+": "+str(error))
            finally:
                # Release while the AppWindow native plugin is still loaded.
                try:
                    setattr(target,field,None)
                    receipt['window_reference_absent']=getattr(target,field) is None
                except BaseException as error:errors.append(type(error).__name__+": "+str(error))
                emit(dict(event='owned_window_reference_cleared',
                    sdk_destroy_returned=receipt['sdk_destroy_returned'],errors=list(errors),
                    window_reference_absent=receipt['window_reference_absent'],
                    native_qualification=False))
        finally:target=None
        if errors:raise RuntimeError("; ".join(errors)) from None

    owner.destroy=destroy
    emit(dict(event='owned_window_release_hook_installed',owner_type=name,
        field=field,window_present=getattr(owner,field) is not None,native_qualification=False))
    return receipt


def validate_window_release_receipt(receipt):
    """Report failed evidence only after SDK shutdown has had its normal turn."""
    if receipt is None:return
    errors=list(receipt['diagnostic_errors'])
    if (receipt['destroy_calls']!=1 or not receipt['sdk_destroy_returned']
            or not receipt['window_reference_absent']):
        errors.append('owned window SDK destroy/release not confirmed exactly once')
    if errors:raise RuntimeError("; ".join(errors)) from None


def prepare_loaded_window_owner(sources,*,record=None):
    """Find only the already-started, source-locked private App extension owner."""
    loader=registry=container=entry=extension=owner=targets=None
    try:
        loader=sys.modules.get('omni.ext._impl._internal')
        if loader is None or Path(loader.__file__).resolve()!=Path(sources['extension_loader']['path']).resolve():
            raise ValueError('owned extension loader source unavailable')
        registry=vars(loader).get('_extensions')
        if type(registry) is not dict:raise ValueError('owned extension registry unavailable')
        targets=[]
        for ext_id,container in registry.items():
            if type(ext_id) is not str or not (ext_id=='omni.kit.hotkeys.core' or ext_id.startswith('omni.kit.hotkeys.core-')):
                continue
            started=getattr(container,'_started_extensions',None)
            if type(started) is not list:raise ValueError('unknown started-extension layout')
            for entry in started:
                if type(entry) is not tuple or len(entry)!=2:raise ValueError('unknown started-extension entry')
                extension=entry[0]
                method=getattr(type(extension),'on_shutdown',None)
                code=getattr(method,'__code__',None)
                if (type(extension).__module__!='omni.kit.hotkeys.core.extension'
                        or type(extension).__name__!='HotkeysExtension' or code is None
                        or Path(code.co_filename).resolve()!=Path(sources['hotkeys_extension']['path']).resolve()):
                    raise ValueError('owned HotkeysExtension source differs')
                owner=vars(extension).get('_hotkey_trigger')
                if owner is not None:targets.append(owner)
        if len(targets)!=1:raise ValueError('exactly one existing owned hotkey trigger required')
        return wrap_preview_hotkey_destroy(targets[0],source=sources['hotkeys_trigger']['path'],record=record)
    finally:
        loader=registry=container=entry=extension=owner=targets=started=None
