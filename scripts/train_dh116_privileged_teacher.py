"""Use the existing scripts/python.sh wrapper; new Teacher is explicit opt-in."""
from go2_pvcnn.control.dh116.tabletop_privileged_cli import main

if __name__ == '__main__':
    raise SystemExit(main('train'))
