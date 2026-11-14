"""Source-bound optional SimulationContext callback cleanup and owner diagnostics.

No SDK import, physics step, STOP, native frame clearing, or extra gc.collect.
The known subscription can be detached; an independent retained constructor
frame remains unresolved until an owner trace identifies its actual holder.
"""
import gc
import hashlib
from pathlib import Path
import traceback
import types
import weakref

from .tabletop_native_lifetime_audit import _describe_referrer

VERSION="dh116-simulation-owner-cleanup-cpu-v1"
CORE=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py")
LAB=Path("/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py")
EVENTS=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/omni/kernel/py/carb/events/_events.pyi")
HASHES=dict(core="0be3b0eaa44e6110cec520be9ac2c08cea17ba97db14b1b0aa60d87824deceb3",
    lab="d95f2eaa138b4fff845ac72439caa2eca5deab4f0d5609e06f163d20eef4df93",
    events="5b65d0b62734824f532de87aca92829a8f5d025b2f02d33378335e62afcc501f")


def simulation_source_contract(*,core_source=CORE,lab_source=LAB,events_source=EVENTS):
    sources={}
    for name,path in (("core",core_source),("lab",lab_source),("events",events_source)):
        path=Path(path).resolve(strict=True)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=HASHES[name]:raise ValueError(name+" installed source differs")
        sources[name]=dict(path=str(path),sha256=sha)
    return dict(version=VERSION,sources=sources,
        subscription_field="_on_post_physics_ready_callback",
        bound_method="_on_post_physics_ready",
        original_clear_omits_ready_subscription=True,
        constructor_frame_upstream_owner_known=False,
        native_qualification=False)


def clear_known_simulation_instance(sim,*,contract):
    """Detach only the known ready subscription, then call original clear_instance.

    Future approved owners may use this as their clear_sim callback. Source and
    method-file guards run before mutation. A failed unsubscribe does not clear
    the subscription field; original singleton/callback cleanup still runs.
    Completion means the API calls returned, never that native release is proven.
    All callback/native aliases are dropped even when an exception is retained.
    """
    handle=callback=clear=code=None
    validated=False
    errors=[]
    result=dict(version=VERSION,subscription_status="unavailable",
        unsubscribe_returned=False,subscription_field_cleared=False,
        original_clear_instance_returned=False,extra_gc_calls=0,
        native_release_verified=False,simulation_liveness_after_return=None,
        constructor_frame_owner_resolved=False,physics_steps_executed=0)
    try:
        expected=simulation_source_contract(**{
            name+"_source":contract["sources"][name]["path"] for name in ("core","lab","events")})
        if contract!=expected:raise ValueError("source contract differs")
        callback=getattr(sim,contract["bound_method"],None)
        code=getattr(getattr(callback,"__func__",None),"__code__",None)
        if code is None or Path(code.co_filename).resolve()!=Path(contract["sources"]["core"]["path"]):
            raise ValueError("actual SimulationContext callback source differs")
        clear=type(sim).clear_instance
        code=getattr(getattr(clear,"__func__",None),"__code__",None)
        paths={Path(contract["sources"][name]["path"]) for name in ("core","lab")}
        if code is None or Path(code.co_filename).resolve() not in paths:
            raise ValueError("actual clear_instance source differs")
        validated=True
        field=contract["subscription_field"]
        if not hasattr(sim,field):raise ValueError("ready subscription ownership is unavailable")
        handle=getattr(sim,field)
        if handle is None:
            result.update(subscription_status="already_absent",subscription_field_cleared=True)
        else:
            unsubscribe=getattr(handle,"unsubscribe",None)
            try:
                if not callable(unsubscribe):raise ValueError("installed ISubscription.unsubscribe API unavailable")
                unsubscribe()
                result["unsubscribe_returned"]=True
                setattr(sim,field,None)
                result.update(subscription_status="explicitly_unsubscribed",subscription_field_cleared=True)
            finally:
                unsubscribe=None
    except BaseException as error:
        errors.append(type(error).__name__+": "+str(error))
        traceback.clear_frames(error.__traceback__)
        error.__traceback__=None
    try:
        if validated:
            clear()
            result["original_clear_instance_returned"]=True
    except BaseException as error:
        errors.append(type(error).__name__+": "+str(error))
        traceback.clear_frames(error.__traceback__)
        error.__traceback__=None
    finally:
        sim=handle=callback=clear=code=None
    if errors:raise RuntimeError("\n".join(errors)) from None
    return result


def probe_simulation_owner_edges(reference,*,max_referrers=16):
    """Bounded scalar two-hop trace of constructor-frame / ready-method owners.

    Never reads f_locals, clears frames, collects objects or follows native
    threads. gc.get_referrers enumerates Python owners, not all C++ holders.
    The returned record retains no referents, methods, frames or native handles.
    """
    if type(reference) is not weakref.ReferenceType:
        raise TypeError("one explicit weak reference required")
    if type(max_referrers) is not int or not 1<=max_referrers<=64:
        raise ValueError("max_referrers must be 1..64")
    target=direct=owner=upstream=parent=None
    result=dict(version=VERSION,alive=None,direct_referrers=[],selected_owner_edges=[],
        root_cause_proven=False,native_release_verified=False,native_thread_attribution=None,
        frame_locals_read=False,diagnostic_handles_retained=False)
    try:
        target=reference()
        result["alive"]=target is not None
        if target is None:return result
        direct=gc.get_referrers(target)
        result.update(object_id=id(target),referrer_count=len(direct),
                      referrers_truncated=len(direct)>max_referrers)
        for i in range(min(len(direct),max_referrers)):
            owner=direct[i]
            description=_describe_referrer(owner,target)
            result["direct_referrers"].append(description)
            selected=(type(owner) is types.FrameType and owner.f_code.co_name=="__init__"
                      or type(owner) is types.MethodType and owner.__func__.__name__=="_on_post_physics_ready")
            if not selected:continue
            upstream=gc.get_referrers(owner)
            summaries=[]
            visible=0
            for j in range(len(upstream)):
                parent=upstream[j]
                if parent is direct:continue
                if (type(parent) is types.FrameType and parent.f_code.co_filename==__file__
                        and parent.f_code.co_name=="probe_simulation_owner_edges"):continue
                visible+=1
                if len(summaries)<max_referrers:
                    summaries.append(_describe_referrer(parent,owner))
            result["selected_owner_edges"].append(dict(direct=description,upstream=summaries,
                upstream_referrer_count=visible,upstream_truncated=visible>max_referrers))
            upstream=parent=None
        return result
    finally:
        target=direct=owner=upstream=parent=None
