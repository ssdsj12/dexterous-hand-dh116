"""Read-only USD collision-plane audit. Run with scripts/python.sh; no Isaac app."""
import glob
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom


def audit(root):
    rows = []
    for side in ('left', 'right'):
        source = Path(glob.glob(str(Path(root) / 'assets/fixed_dual_panda_dh116_v1/dependencies' / ('*DH116_' + side + '.usd')))[0])
        stage = Usd.Stage.Open(str(source))
        cache = UsdGeom.XformCache()
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if '/collisions/' not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh):
                continue
            body = prim.GetParent()
            while body and body.GetName() not in ('finger14_link', 'finger23_link', 'finger33_link', 'finger43_link', 'finger53_link'):
                body = body.GetParent()
            if not body:
                continue
            p = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            relative = np.asarray(cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse())
            p = np.unique((np.c_[p, np.ones(len(p))] @ relative)[:, :3], axis=0)
            hull = ConvexHull(p)
            tri = p[hull.simplices]
            area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=-1)/2
            planes = {}
            for eq, a, t in zip(hull.equations, area, tri):
                key = tuple(np.round(eq, 6))
                row = planes.setdefault(key, dict(area=0., triangles=[], equation=eq.tolist()))
                row['area'] += float(a)
                row['triangles'].append(t.tolist())
            rows.append(dict(side=side, body=body.GetName(), source=str(source),
                             sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             mesh=str(prim.GetPath()), lower=p.min(0).tolist(), upper=p.max(0).tolist(),
                             planes=sorted(planes.values(), key=lambda v: -v['area'])))
    return rows


if __name__ == '__main__':
    import sys
    rows = audit(sys.argv[1])
    if '--summary' in sys.argv:
        refs = {'left': [0., -.01045, -.03394], 'right': [0., .03219, -.01556]}
        for row in rows:
            print(row['side'], row['body'])
            thumb = row['body'] == 'finger14_link'
            ref = np.array(refs[row['side']] if thumb else [0., row['lower'][1]+.0015, row['upper'][2]-.006])
            candidates = sorted(row['planes'], key=lambda p: np.linalg.norm(np.array(p['triangles']).reshape(-1, 3).mean(0)-ref))
            for p in candidates[:6]:
                vs = np.array(p['triangles']).reshape(-1, 3)
                print('area', round(p['area']*1e6, 2), 'eq', np.round(p['equation'], 6),
                      'center', np.round(vs.mean(0), 6), 'lo', np.round(vs.min(0), 6), 'hi', np.round(vs.max(0), 6))
    else:
        print(json.dumps(rows))
