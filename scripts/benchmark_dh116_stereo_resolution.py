"""Run inside the existing scripts/python.sh environment; native is explicit."""
from go2_pvcnn.control.dh116.tabletop_stereo_resolution_benchmark import main

if __name__=='__main__':
    raise SystemExit(main())
