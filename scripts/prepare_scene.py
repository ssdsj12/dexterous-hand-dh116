"""Validate a measured scene bundle before native Reset or SDF use."""
import argparse, json
from pathlib import Path
from dh116.env.scene import load_scene_inputs

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    a = p.parse_args()
    data = load_scene_inputs(a.manifest, a.sha256)
    print(json.dumps(dict(grasp=data['grasp'], scene_sha256=data['manifest_sha256'], mesh_count=len(data['archive']['mesh_plan']['entries']), qualification=data['qualification'])))
if __name__ == '__main__':
    main()
