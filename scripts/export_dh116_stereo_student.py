"""Export an independently versioned actor-only deployment artifact."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from go2_pvcnn.control.dh116.tabletop_stereo_student_cli import main

if __name__=='__main__':
    raise SystemExit(main('export'))
