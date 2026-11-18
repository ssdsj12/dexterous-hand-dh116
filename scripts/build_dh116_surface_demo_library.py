"""Build a provenance-preserving two/three-finger demonstration library."""
import argparse,json
from pathlib import Path
from go2_pvcnn.control.dh116.tabletop_surface_demo_library import merge_datasets

p=argparse.ArgumentParser()
p.add_argument('--pinch',type=Path,required=True)
p.add_argument('--tripod',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
data=merge_datasets([('pinch_thumb_index',args.pinch),
                     ('tripod_thumb_index_middle',args.tripod)],args.output)
print('SURFACE_DEMO_LIBRARY',json.dumps(dict(samples=len(data['observations']),
    episodes=len(data['episode_style_labels']),qualification=data['qualification'],
    episode_style_labels=data['episode_style_labels'],
    reset_family=[dict(label=x['label'],initialization_sha256=x['contract']['grasp_initialization_sha256'])
                  for x in data['reset_family']])),flush=True)
