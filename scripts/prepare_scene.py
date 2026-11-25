"""Validate a scene bundle or materialize an anatomical placement from its pinned source."""

import argparse
import hashlib
import json
from pathlib import Path

from dh116.env.contracts import ProjectPaths
from dh116.env.scene import (
    ANATOMICAL_SCENE_VERSION,
    anatomical_hand_axes,
    anatomical_scene_inputs,
    load_anatomical_reference,
    load_scene_inputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--anatomical-output', type=Path)
    parser.add_argument('--source-stations', nargs=2, type=int, metavar=('LEFT', 'RIGHT'))
    parser.add_argument('--reference-sha256')
    args = parser.parse_args()
    paths = ProjectPaths()
    source = paths.file(args.manifest)
    data = load_scene_inputs(source, args.sha256)
    result = dict(grasp=data['grasp'], scene_sha256=data['manifest_sha256'],
                  mesh_count=len(data['archive']['mesh_plan']['entries']),
                  qualification=data['qualification'])
    if args.anatomical_output is None:
        if args.source_stations is not None or args.reference_sha256 is not None:
            parser.error('placement inputs require --anatomical-output')
    else:
        if args.source_stations is None or args.reference_sha256 is None:
            parser.error('anatomical placement requires --source-stations and --reference-sha256')
        spec = dict(schema=ANATOMICAL_SCENE_VERSION, grasp=data['grasp'],
                    source_scene=dict(path=str(source.relative_to(paths.root)), sha256=args.sha256),
                    source_station_nodes=args.source_stations, reference_sha256=args.reference_sha256)
        scene = anatomical_scene_inputs(data, spec, root=paths.root)
        reference = load_anatomical_reference(data['grasp'], args.reference_sha256, root=paths.root)
        axes = {}
        for i, side in enumerate(('left', 'right')):
            asset = paths.file(scene['assets'][side])
            poses = (scene['authored_seed']['reset_root_pose_w'][i], reference['root_pose_wxyz'][i])
            axes[side] = {}
            for name, pose in zip(('reset', 'goal'), poses, strict=True):
                longitudinal, thumb = anatomical_hand_axes(asset, pose)
                if longitudinal[0] >= 0 or (thumb[1] <= 0 if side == 'left' else thumb[1] >= 0):
                    raise ValueError(f'original {side} {name} pose has incompatible anatomical axes')
                axes[side][name] = dict(longitudinal=longitudinal.tolist(), thumb_side=thumb.tolist())
        output = paths.output(args.anatomical_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(spec, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
        with output.open('xb') as stream:
            stream.write(raw)
        result.update(scene_sha256=hashlib.sha256(raw).hexdigest(),
                      authored_seed_sha256=scene['authored_seed']['sha256'],
                      anatomical_placement=scene['anatomical_placement'], anatomy=axes)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
