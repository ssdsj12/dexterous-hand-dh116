"""Explicit independent stereo Student PPO; use --check-config before native opt-in."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from go2_pvcnn.control.dh116.tabletop_stereo_student_cli import main

if __name__=='__main__':
    raise SystemExit(main('train'))
