"""Publish process outcomes without overwriting the last training counters."""
import json
from pathlib import Path


def publish_terminal_monitor(path, report, exit_code):
    path=Path(path)
    try:
        previous=json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        previous={}
    if not isinstance(previous,dict):previous={}
    for key in ('error','interrupted'):
        previous.pop(key,None)
    previous.update({key:report[key] for key in ('status','error','interrupted') if key in report})
    previous['process_exit_code']=exit_code
    temporary=path.with_suffix('.terminal.tmp')
    temporary.write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def finalize_run(output, report, cleanups):
    """Attempt every cleanup; publish success only after cleanup succeeded."""
    import traceback
    output=Path(output)

    def failure(context):
        detail=context+'\n'+traceback.format_exc()
        report.update(status='error',error=report.get('error','')+'\n'+detail)
        print(detail,flush=True)

    def code():
        return 2 if (report.get('error') or report.get('interrupted') or report.get('training_skipped')
                     or report.get('training_stopped')=='physical_safety_failure') else 0

    for cleanup in cleanups:
        try:
            cleanup()
        except BaseException:
            failure('run cleanup failed')
    report['process_exit_code']=code()
    try:
        publish_terminal_monitor(output/'monitor.json',report,code())
    except Exception:
        failure('terminal monitor publication failed')
    report['process_exit_code']=code()
    try:
        temporary=output/'summary.terminal.tmp'
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(output/'summary.json')
    except Exception:
        failure('terminal summary publication failed')
        report['process_exit_code']=code()
        try:
            publish_terminal_monitor(output/'monitor.json',report,code())
        except Exception:
            failure('terminal error monitor publication failed')
    return code()
