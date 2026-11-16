"""Private preview teardown of the installed extension-interface cache."""
from pathlib import Path
import hashlib
import json
import sys


def prepare_extension_cache_release(app,getter,*,source,record=None):
    try:
        code=getattr(getter,'__code__',None)
        if code is None or Path(code.co_filename).resolve()!=Path(source).resolve():
            raise ValueError('extension cache source differs')
        if (type(vars(getter).get('extensions')).__module__+'.'+
                type(vars(getter).get('extensions')).__name__!='omni.ext._extensions.IExtensions'):
            raise ValueError('existing IExtensions cache required')
        framework=app._framework
        if not callable(getattr(framework,'unload_all_plugins',None)):
            raise ValueError('original Framework unload required')
        receipt=dict(unload_attempted=False,cache_reference_removed=False,framework_unload_returned=False,
            diagnostic_errors=[],native_qualification=False)
        class OwnedFramework:
            def __getattr__(self,name):return getattr(framework,name)
            def unload_all_plugins(self):
                if receipt['unload_attempted']:
                    raise RuntimeError('framework release cannot be retried')
                receipt['unload_attempted']=True
                # App.shutdown has completed, plugins are still loaded here.
                try:
                    del getter.extensions
                    receipt['cache_reference_removed']=True
                except BaseException as error:receipt['diagnostic_errors'].append(str(error))
                if record is not None:
                    try:record(dict(event='extension_cache_reference_removed',native_qualification=False))
                    except BaseException as error:receipt['diagnostic_errors'].append(str(error))
                result=framework.unload_all_plugins()
                receipt['framework_unload_returned']=True
                return result
        app._framework=OwnedFramework()
        return receipt
    finally:app=None


def prepare_loaded_extension_cache(app,*,record=None,output=None,apply=False):
    try:
        root=Path(__file__).resolve().parents[4]
        source=json.loads((root/'configs/dh116_execution/preview_extension_cache_source_v1.json').read_text())
        path=Path(source['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('installed extension cache source changed')
        module=sys.modules.get('omni.ext._impl')
        getter=None if module is None else vars(module).get('get_extensions_interface')
        public=sys.modules.get('omni.ext')
        if public is None or vars(public).get('get_extensions_interface') is not getter:
            raise ValueError('public extension getter alias differs')
        if record is None and output is not None:
            def record(row):
                import os,time
                target=Path(output)/'preview_extension_cache_release.jsonl'
                with target.open('a') as stream:
                    stream.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(),**row))+'\n')
                    stream.flush();os.fsync(stream.fileno())
        if not apply:
            receipt=dict(mode='audit_only',cache_present=hasattr(getter,'extensions'),
                cached_type=type(getattr(getter,'extensions',None)).__module__+'.'+
                    type(getattr(getter,'extensions',None)).__name__,
                diagnostic_errors=[],native_qualification=False)
            if record is not None:
                try:record(dict(event='extension_cache_read_only_probe',**receipt))
                except BaseException as error:receipt['diagnostic_errors'].append(str(error))
            return receipt
        return prepare_extension_cache_release(app,getter,source=path,record=record)
    finally:app=None


def validate_extension_cache_receipt(receipt):
    if receipt is None:return
    if receipt.get('mode')=='audit_only':
        if receipt['diagnostic_errors']:raise RuntimeError(str(receipt['diagnostic_errors']))
        return
    if (not receipt['cache_reference_removed'] or not receipt['framework_unload_returned']
            or receipt['diagnostic_errors']):
        raise RuntimeError('extension cache release/unload not confirmed: '+str(receipt))
