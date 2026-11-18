#!/usr/bin/env python3
"""Independent source-bound stat-cache cleanup diagnostic; default only freezes."""
import argparse
from pathlib import Path
import json
import sys
import weakref

from scripts import diagnose_dh116_teacher_initialization as legacy
from go2_pvcnn.control.dh116 import tabletop_stat_cache_release as cache
from go2_pvcnn.control.dh116 import tabletop_contact_offset_baseline as offsets

VERSION="dh116-teacher-stat-cache-release-native-v1"


def validate_execution_context(root):
    root=Path(root).resolve()
    if Path(__file__).resolve()!=root/"scripts/diagnose_dh116_teacher_stat_cache_release.py":
        raise ValueError("stat cache worker source differs from project")
    legacy.validate_execution_context(root)


def freeze_plan(root,output,binding_reports,*,contact_offset_baseline=False):
    plan=legacy.freeze_plan(root,output,binding_reports)
    contract=cache.stat_cache_source_contract()
    plan["version"]=VERSION
    plan["stat_cache_release"]={"max_reset_calls":1,"phase":"before_existing_owned_gc",
                                "source_contract":contract}
    for pin in contract["sources"].values():
        plan["frozen_files"][pin["path"]]=pin["sha256"]
    if contact_offset_baseline:
        offset_contract = offsets.source_contract()
        plan["contact_offset_baseline"] = offset_contract
        plan["frozen_files"][offset_contract["api_path"]] = offset_contract["api_sha256"]
    return plan


def verify_plan(plan,approved):
    if approved!=legacy.digest(plan):raise ValueError("exact approved stat-cache frozen SHA required")
    if plan.get("version")!=VERSION:raise ValueError("independent stat cache plan version required")
    rebuilt=freeze_plan(plan["project"],plan["output"],plan["binding_reports"],
                        contact_offset_baseline="contact_offset_baseline" in plan)
    if plan!=rebuilt:raise ValueError("stat cache frozen source/contract/budget differs")
    if Path(plan["output"]).exists():raise ValueError("fresh exclusive output directory required")
    return plan


def make_before_collect(plan):
    output=Path(plan["output"])
    contract=plan["stat_cache_release"]["source_contract"]
    if plan["stat_cache_release"]["max_reset_calls"]!=1:raise ValueError("one reset budget required")
    called=False
    def before_collect(simulation):
        nonlocal called
        if called:raise RuntimeError("stat cache reset is once only")
        called=True
        reference=None;module=None
        try:
            reference=weakref.ref(simulation)
            simulation=None
            module=sys.modules.get(cache.MODULE)
            if module is None:raise ValueError("already loaded SDK cache module required")
            app_module=sys.modules.get("omni.kit.app")
            if app_module is None:raise ValueError("already loaded Kit app module required")
            def read_clock():
                target=None
                try:
                    target=reference()
                    if target is None:raise ValueError("live SDK clock owner unavailable")
                    row=legacy.clock(target)
                    row["app_update_number"]=app_module.get_app().get_update_number()
                    return row
                finally:target=None
            def observe(phase,row):
                legacy.write_new(output/("stat_cache_"+phase+".json"),row)
            receipt=cache.release_owned_stat_cache(module,contract=contract,
                         clock_reader=read_clock,observer=observe)
            receipt["final_post_write_guard_required"]=True
            receipt["final_post_write_guard_attested_by_this_file"]=False
            legacy.write_new(output/"stat_cache_release.json",receipt)
            # Last operation after all successful evidence writes. No second
            # write/reset/update follows this read-only guard.
            cache.verify_stat_cache_empty(module,contract=contract,clock_reader=read_clock,
                                          expected_clock=receipt["clock_before"])
        except Exception as error:
            message=type(error).__name__+": "+str(error)
            error.__traceback__=None
            try:legacy.write_new(output/"stat_cache_error.json",
                    {"error":message,"native_release_verified":False,"reset_retried":False})
            except Exception:
                pass
            raise RuntimeError(message) from None
        finally:
            simulation=reference=module=None
    return before_collect


def make_offset_capture(plan):
    output = Path(plan["output"])
    contract = plan["contact_offset_baseline"]

    def capture(sim, simulation_view, bindings, expected_clock):
        try:
            from pxr import UsdGeom

            api = sys.modules.get(offsets.API_MODULE)
            if api is None or str(Path(api.__file__).resolve()) != contract["api_path"]:
                raise ValueError("loaded Physics Tensor API does not match frozen source")
            result = offsets.read_contact_offsets(
                simulation_view, bindings, meters_per_unit=UsdGeom.GetStageMetersPerUnit(sim.stage),
                read_clock=lambda: legacy.clock(sim), expected_clock=expected_clock)
            result["frozen_api_source_sha256"] = contract["api_sha256"]
            legacy.write_new(output/"contact_offset_baseline.json", result)
        except Exception as error:
            # The raw cooked receipt is already durable. Closure stays on the
            # original owner-release path if this optional diagnostic fails.
            text = f"{type(error).__name__}: {error}"
            error.__traceback__ = None
            legacy.write_new(output/"contact_offset_baseline_error.json", {"error": text})
        finally:
            sim = simulation_view = None

    return capture


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--binding-reports",type=Path,default=legacy.BINDINGS)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--approved-frozen-sha256")
    parser.add_argument("--contact-offset-baseline",action="store_true",
                        help="include an exact-link, read-only PhysX offset baseline")
    args=parser.parse_args(argv)
    validate_execution_context(args.project)
    if args.execute_native:
        plan=json.loads(args.plan.read_text())
        verify_plan(plan,args.approved_frozen_sha256)
        if (str(args.project.resolve())!=plan["project"] or str(args.output.resolve())!=plan["output"]
                or str(args.binding_reports.resolve())!=plan["binding_reports"]):
            raise ValueError("CLI and frozen stat cache identity differ")
        if args.contact_offset_baseline != ("contact_offset_baseline" in plan):
            raise ValueError("CLI contact offset baseline differs from approved plan")
        before_collect = make_before_collect(plan)
        if args.contact_offset_baseline:
            legacy.execute_native(plan, before_owner_collect=before_collect,
                                  after_cooked_capture=make_offset_capture(plan))
        else:
            legacy.execute_native(plan, before_owner_collect=before_collect)
    else:
        plan=freeze_plan(args.project,args.output,args.binding_reports,
                         contact_offset_baseline=args.contact_offset_baseline)
        legacy.write_new(args.plan,plan)
        print(json.dumps({"native_execution_performed":False,"plan":str(args.plan),
                          "frozen_sha256":legacy.digest(plan),"cache_api_executed":False}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
