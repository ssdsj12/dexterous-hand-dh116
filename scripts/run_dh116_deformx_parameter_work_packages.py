"""Parameter-only worker using the already tested bounded scheduler transport."""
from pathlib import Path
from scripts import run_dh116_deformx_pair_work_packages as base
from scripts.verify_dh116_deformx_parameter_getters import ENTRY, WORKER_ENV, validate_artifacts

WRAPPER="scripts/run_dh116_deformx_parameter_work_packages.py"

def main():
    values={"__file__":str(Path(__file__).resolve()),"ENTRY":ENTRY,"WRAPPER":WRAPPER,
            "WORKER_ENV":WORKER_ENV,"validate_artifacts":validate_artifacts}
    saved={k:getattr(base,k) for k in values}
    try:
        for k,v in values.items():setattr(base,k,v)
        return base.main()
    finally:
        for k,v in saved.items():setattr(base,k,v)

if __name__=="__main__":raise SystemExit(main())
