"""Read-only source-surface audit. No contact qualification or simulation steps."""
import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
from go2_pvcnn.control.dh116.usd_surface import layer_pins, read_surface


BODIES = ('base_link', 'finger14_link', 'finger23_link', 'finger33_link',
          'finger43_link', 'finger53_link')


def audit(asset, output):
    asset, output = Path(asset), Path(output)
    if output.exists():
        raise ValueError('refusing existing output directory')
    stage = Usd.Stage.Open(str(asset.resolve(strict=True)))
    if stage is None:
        raise ValueError('cannot open asset')
    pins = layer_pins(stage)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output.mkdir(parents=True)
    report = dict(asset=str(asset.resolve()), layers=pins, bodies=[],
                  calibration_qualified=False, physics_stepped=False,
                  pose='authored USD transforms, not an actuated grasp')
    cache = UsdGeom.XformCache()
    for side in ('left', 'right'):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection='3d')
        bx = fig.add_subplot(122, projection='3d')
        base = stage.GetPrimAtPath(f'/FixedDualPandaDH116/{side}_arm/dh116/base_link')
        to_base = cache.GetLocalToWorldTransform(base).GetInverse()
        all_points = []
        for i, body in enumerate(BODIES):
            body_path = f'/FixedDualPandaDH116/{side}_arm/dh116/{body}'
            collision = read_surface(stage, body_path)
            visual = read_surface(stage, body_path, 'visuals')
            points = np.asarray(collision['points_body'])
            faces = np.asarray(collision['triangles'])
            same = (collision['geometry_sha256'] == visual['geometry_sha256']
                    and np.allclose(points, visual['points_body'], atol=1e-12, rtol=0.))
            if not same:
                raise ValueError(f'visual/collision mismatch: {body_path}')
            brief = {k: v for k, v in collision.items()
                     if k not in ('points_mesh', 'points_body', 'triangles')}
            brief.update(side=side, body=body, vertex_count=len(points),
                         face_count=len(faces), visual_collision_equal=same,
                         bounds_body=[points.min(0).tolist(), points.max(0).tolist()])
            report['bodies'].append(brief)
            matrix = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)) * to_base).T
            transformed = points @ matrix[:3, :3].T + matrix[:3, 3]
            all_points.append(transformed)
            stride = max(1, len(faces) // 4500)
            colour = plt.get_cmap('tab10')(i)
            for view in (ax, bx):
                view.add_collection3d(Poly3DCollection(transformed[faces[::stride]],
                                        facecolor=colour, linewidth=.05, edgecolor=colour, alpha=.6))
                center = transformed.mean(0)
                view.text(*center, body, fontsize=7)
            if body in ('finger14_link', 'finger23_link'):
                local_fig = plt.figure(figsize=(13, 4))
                for j, (elev, azim) in enumerate(((15, 0), (15, 90), (80, 0))):
                    la = local_fig.add_subplot(1, 3, j + 1, projection='3d')
                    la.add_collection3d(Poly3DCollection(points[faces], facecolor='lightsteelblue',
                                             edgecolor='slategray', linewidth=.12, alpha=.9))
                    la.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])
                    la.set_box_aspect(np.maximum(np.ptp(points, axis=0), .001))
                    la.view_init(elev=elev, azim=azim)
                    la.set(xlabel='local X (m)', ylabel='local Y (m)', zlabel='local Z (m)')
                local_fig.suptitle(f'{side} {body}: source surface, no pad selected')
                local_fig.tight_layout()
                local_fig.savefig(output / f'{side}_{body}.png', dpi=140)
                plt.close(local_fig)
        cloud = np.concatenate(all_points)
        for view, azim in ((ax, -60), (bx, 120)):
            view.auto_scale_xyz(cloud[:, 0], cloud[:, 1], cloud[:, 2])
            view.set_box_aspect(np.maximum(np.ptp(cloud, axis=0), .001))
            view.view_init(elev=20, azim=azim)
            view.set(xlabel='base X (m)', ylabel='base Y (m)', zlabel='base Z (m)')
        fig.suptitle(f'{side}: palm and distal bodies, authored pose; sparse palm triangles')
        fig.tight_layout()
        fig.savefig(output / f'{side}_overview.png', dpi=140)
        plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.asset, args.output)
    print(json.dumps(dict(bodies=len(result['bodies']), calibration_qualified=False)))
