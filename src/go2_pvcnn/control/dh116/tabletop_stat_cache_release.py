"""Source-bound Kit cache release for an explicitly selected diagnostic only.

No SDK imports, native release claims, physics steps, frame clearing or new GC.
"""
import hashlib
import functools
import importlib
import math
from pathlib import Path
import sys
import types

VERSION="dh116-kit-stat-cache-release-v1"
MODULE="omni.ext._impl.stat_cache"
BASE=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/omni/kernel/py/omni/ext/_impl")
SOURCES={"cache":(BASE/"stat_cache.py","d09905d34ad52577c95f997d1f69fa152b3b3d71dc7e8570e777842ef064cd6b"),
         "settings":(BASE/"ext_settings.py","9ae0326d15a7573db685a1b6bed9d22d4a31d9008c2743e6bfacecb766f37fd4"),
         "startup":(BASE/"_internal.py","6dfde3b8a85cbb33746e0c506e5a9ba5cd2a7ad90caeb568098c274c14126283")}
FUNCTIONS={"_os_stat_cached":11,"refresh_stat_cache":26,"reset_stat_cache":43}


def stat_cache_source_contract():
    sources={}
    for name,(path,expected) in SOURCES.items():
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError("installed stat cache source differs: "+name)
        sources[name]={"path":str(path.resolve(strict=True)),"sha256":actual}
    return {"version":VERSION,"module":MODULE,"sources":sources,"functions":dict(FUNCTIONS),
            "max_reset_calls":1,"app_updates_added":0,"native_release_verified":False}


def _source_codes(path):
    # Compile for comparison only. Never execute SDK imports or native code.
    compiled=compile(Path(path).read_bytes(),str(path),"exec",
                     dont_inherit=True,optimize=sys.flags.optimize)
    return {code.co_name:code for code in compiled.co_consts
            if type(code) is types.CodeType}


def _validate_function(fn,fields,path,expected,*,defaults=None):
    if (type(fn) is not types.FunctionType or fn.__globals__ is not fields
            or Path(fn.__code__.co_filename).resolve()!=Path(path)
            or fn.__code__!=expected or fn.__defaults__!=defaults or fn.__kwdefaults__ is not None):
        raise ValueError("loaded stat cache function source/body differs: "+expected.co_name)


def _validated_fields(module,contract):
    if contract!=stat_cache_source_contract():raise ValueError("stat cache source contract differs")
    if type(module) is not types.ModuleType or module.__name__!=MODULE or sys.modules.get(MODULE) is not module:
        raise ValueError("loaded stat cache module identity differs")
    fields=vars(module);path=contract["sources"]["cache"]["path"]
    if Path(fields.get("__file__","")).resolve()!=Path(path):
        raise ValueError("loaded stat cache module source differs")
    codes=_source_codes(path)
    for name in FUNCTIONS:
        _validate_function(fields.get(name),fields,path,codes[name])
    if importlib._bootstrap_external._path_stat is not fields["_os_stat_cached"]:
        raise ValueError("installed stat cache hook is not active; refusing to enable it")
    settings_name="omni.ext._impl.ext_settings"
    settings=sys.modules.get(settings_name)
    settings_path=contract["sources"]["settings"]["path"]
    if (type(settings) is not types.ModuleType or settings.__name__!=settings_name
            or Path(vars(settings).get("__file__","")).resolve()!=Path(settings_path)):
        raise ValueError("loaded settings source identity differs")
    settings_fields=vars(settings);settings_codes=_source_codes(settings_path)
    _validate_function(settings_fields.get("_get_setting"),settings_fields,settings_path,
                       settings_codes["_get_setting"],defaults=(None,))
    wrapper_type=type(functools.lru_cache()(stat_cache_source_contract))
    for name in ("_is_reentrant_test_mode","_is_path_stat_cache_enabled"):
        predicate=settings_fields.get(name)
        if type(predicate) is not wrapper_type:
            raise ValueError("source settings predicate wrapper required")
        _validate_function(predicate.__wrapped__,settings_fields,settings_path,settings_codes[name])
    predicate=settings_fields["_is_path_stat_cache_enabled"]
    if fields.get("_is_path_stat_cache_enabled") is not predicate:
        raise ValueError("source settings predicate identity differs")
    if predicate() is not True:raise ValueError("stat cache must already be enabled")
    if type(fields.get("_stat_cache")) is not dict:
        raise ValueError("ordinary source stat cache dictionary required")
    return fields


def _exception_record(error):
    tb=frame=current=None
    try:
        tb=error.__traceback__;records=[];seen=set();tb_count=0;truncated=False
        while tb is not None and tb_count<32:
            tb_count+=1;frame=tb.tb_frame;current=frame
            depth=0
            while current is not None and depth<32:
                identity=id(current)
                if identity in seen:break
                if len(records)>=128:
                    truncated=True;break
                seen.add(identity)
                records.append({"object_id":identity,"function":current.f_code.co_name,
                    "filename":current.f_code.co_filename,"line":current.f_lineno,
                    "f_back_id":id(current.f_back) if current.f_back is not None else None})
                current=current.f_back;depth+=1
            if current is not None and depth>=32:truncated=True
            tb=tb.tb_next
        if tb is not None:truncated=True
        related=any((r["filename"].endswith("/torch/cuda/__init__.py")
                      and r["function"] in ("_lazy_init","_register_triton_kernels"))
                    or r["filename"].endswith("/isaaclab/sim/simulation_context.py")
                    or r["filename"].endswith("[force-diagnostic-only]") for r in records)
        return {"exception_id":id(error),"exception_type":type(error).__name__,
                "traceback_nodes_examined":tb_count,"frame_chain":records,
                "frame_descriptions_truncated":truncated,"runtime_related":related}
    finally:
        error=tb=frame=current=None


def snapshot_stat_cache(module,*,contract,max_entries=4096,max_records=16):
    if type(max_entries) is not int or not 1<=max_entries<=4096:
        raise ValueError("cache scan budget must be 1..4096")
    if type(max_records) is not int or not 1<=max_records<=16:
        raise ValueError("cache record budget must be 1..16")
    fields=cache=key=value=error=None
    try:
        fields=_validated_fields(module,contract);cache=fields["_stat_cache"]
        primary=[];other=[];examined=exceptions=related=unsupported=0
        for key in reversed(cache):
            if examined>=max_entries:break
            examined+=1;value=cache[key]
            if type(value) is not tuple or len(value)!=2:
                unsupported+=1;continue
            error=value[1]
            if error is None:continue
            exceptions+=1
            if not isinstance(error,BaseException) or type(error).__module__!="builtins":
                unsupported+=1;continue
            record=_exception_record(error)
            record["key_sha256"]=(hashlib.sha256(key.encode("utf-8","surrogatepass")
                                   if type(key) is str else key).hexdigest()
                                  if type(key) in (str,bytes) else None)
            if record["runtime_related"]:
                related+=1
                if len(primary)<max_records:primary.append(record)
            elif len(other)<max_records:other.append(record)
        records=primary+other[:max_records-len(primary)]
        epoch=fields.get("_stat_cache_update_number")
        return {"version":VERSION,"source_validated":True,"active_patch_id":id(fields["_os_stat_cached"]),
                "cache_entries_total":len(cache),"entries_scanned":examined,
                "entries_scan_truncated":len(cache)>examined,"cached_exceptions":exceptions,
                "runtime_related_entries":related,"exception_records":records,
                "exception_records_truncated":exceptions>len(records),"unsupported_entries":unsupported,
                "cache_epoch":epoch if type(epoch) is int else None,
                "native_release_verified":False,"root_cause_proven":False}
    finally:
        module=fields=cache=key=value=error=None


def _clock(reader):
    row=reader()
    if type(row) is not dict or set(row)!={"sdk_step","sdk_time_s","app_update_number"}:
        raise ValueError("complete SDK/App clock required")
    step,t,update=row["sdk_step"],row["sdk_time_s"],row["app_update_number"]
    if (type(step) is not int or step!=2 or type(t) not in (float,int)
            or not math.isfinite(t) or not math.isclose(t,2/480,rel_tol=0,abs_tol=2e-6)
            or type(update) is not int or update<0):
        raise ValueError("SDK2 and valid non-sentinel App clock required")
    return dict(row)


def verify_stat_cache_empty(module,*,contract,clock_reader,expected_clock):
    """Read-only final guard; never reset twice, write evidence or advance App."""
    fields=None
    try:
        fields=_validated_fields(module,contract)
        expected=_clock(lambda:expected_clock)
        before=_clock(clock_reader)
        snapshot=snapshot_stat_cache(module,contract=contract)
        after=_clock(clock_reader)
        if before!=expected or after!=expected:
            raise ValueError("clock changed after stat cache evidence IO")
        # Last direct check of the live dictionary, AFTER the last reader.
        # No callback, source import, reset or evidence IO can follow it here.
        if type(fields.get("_stat_cache")) is not dict or len(fields["_stat_cache"])!=0:
            raise ValueError("stat cache repopulated after evidence IO; no reset retry")
        return snapshot
    finally:
        module=fields=clock_reader=None


def release_owned_stat_cache(module,*,contract,clock_reader,observer=None):
    fields=None
    io_errors=[]
    def emit(phase,row):
        if observer is None:return
        try:observer(phase,row)
        except Exception as error:
            io_errors.append(type(error).__name__+": "+str(error))
            error.__traceback__=None
    try:
        fields=_validated_fields(module,contract)
        before=snapshot_stat_cache(module,contract=contract)
        clock_before=_clock(clock_reader)
        # Even a process interruption in evidence writing cannot bypass the
        # source-qualified cache reset. The original owned GC still follows.
        try:emit("before",dict(before,clock=clock_before))
        finally:fields["reset_stat_cache"]()
        clock_after=_clock(clock_reader)
        if clock_after!=clock_before:raise ValueError("clock changed during stat cache API")
        after=snapshot_stat_cache(module,contract=contract)
        if after["cache_entries_total"]!=0:raise ValueError("stat cache reset did not leave an empty cache")
        receipt={"version":VERSION,"source_validated":True,"reset_api_calls":1,
                 "reset_api_returned":True,"before":before,"after":after,
                 "clock_before":clock_before,"clock_after":clock_after,
                 "patch_preserved":before["active_patch_id"]==after["active_patch_id"],
                 "native_release_verified":False,"root_cause_proven":False}
        emit("after",dict(after,clock=clock_after,reset_api_calls=1))
        if io_errors:raise RuntimeError("stat cache evidence IO failed: "+"; ".join(io_errors))
        receipt["post_observer_check"]=verify_stat_cache_empty(module,contract=contract,
            clock_reader=clock_reader,expected_clock=clock_before)
        return receipt
    finally:
        module=fields=clock_reader=observer=None
