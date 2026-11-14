"""Offline saved-native diagnostics only; no Isaac entry point."""
from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import main

if __name__=='__main__':
    raise SystemExit(main())
