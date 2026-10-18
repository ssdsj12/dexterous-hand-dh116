"""Diagnostic view of pinned simulation reference points, no physics stepping."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from pxr import Usd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from go2_pvcnn.control.dh116.usd_surface import read_surface


def render(reference_file,output):
    r=json.loads(Path(reference_file).read_text());output=Path(output)
    if output.exists():raise ValueError('refusing existing plot directory')
    for path,sha in r['layers'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('layer identity mismatch')
    stage=Usd.Stage.Open(r['asset']);output.mkdir(parents=True)
    for side in ('left','right'):
        fig=plt.figure(figsize=(13,9))
        refs=[p for p in r['references'] if p['side']==side]
        for i,ref in enumerate(refs):
            mesh=read_surface(stage,ref['body_path'])
            if mesh['geometry_sha256']!=ref['geometry_sha256']:raise ValueError('geometry changed')
            pts=np.array(mesh['points_body']);faces=np.array(mesh['triangles'])
            ax=fig.add_subplot(2,3,i+1,projection='3d')
            stride=max(1,len(faces)//5000)
            ax.add_collection3d(Poly3DCollection(pts[faces[::stride]],facecolor='lightsteelblue',alpha=.55))
            point=np.array(ref['point']);normal=np.array(ref['normal'])
            ax.scatter(*point,c='red',s=28,depthshade=False)
            ax.quiver(*point,*normal,length=.007,color='red')
            if ref['digit']=='palm':
                frame=np.array(r['palm_frames'][side])
                for k,c in enumerate(('r','g','b')):ax.quiver(*point,*frame[:3,k],length=.015,color=c)
            ax.auto_scale_xyz(pts[:,0],pts[:,1],pts[:,2]);ax.set_box_aspect(np.maximum(np.ptp(pts,axis=0),.001))
            ax.view_init(elev=float(np.rad2deg(np.arcsin(normal[2]))),azim=float(np.rad2deg(np.arctan2(normal[1],normal[0]))))
            ax.set_title(f"{ref['digit']} face {ref['triangle_index']}")
            ax.set(xlabel='X m',ylabel='Y m',zlabel='Z m')
        fig.suptitle(side+' simulation references: red point/normal; palm RGB axes (not a contact pad)')
        fig.tight_layout();fig.savefig(output/f'{side}_references.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--references',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();render(a.references,a.output)
