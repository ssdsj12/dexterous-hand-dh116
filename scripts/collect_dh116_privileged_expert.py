"""Collect new-model native candidates; never auto-admit failed demonstrations."""
from go2_pvcnn.control.dh116.tabletop_privileged_cli import main

if __name__ == '__main__':
    raise SystemExit(main('collect'))
