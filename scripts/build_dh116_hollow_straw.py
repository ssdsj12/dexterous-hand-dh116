"""Build the rigid open-bore Y-axis straw asset without starting Isaac."""
import argparse
import json
from pathlib import Path

from go2_pvcnn.control.dh116.hollow_straw import build_straw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,
        default=Path(__file__).resolve().parents[1]/'assets/dh116_hollow_straw_v1')
    args = parser.parse_args()
    print(json.dumps(build_straw(args.output),indent=2))


if __name__ == '__main__':
    main()
