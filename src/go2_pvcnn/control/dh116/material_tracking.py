"""Fixed rod material anchors in measured palm frames; never reselect on slip."""
import numpy as np


def material_in_palm(nodes,poses,material_ids):
    nodes=np.asarray(nodes,dtype=float)
    if (nodes.ndim!=2 or nodes.shape[1]!=3 or len(nodes)<2 or not np.isfinite(nodes).all()
            or set(poses)!={'left','right'} or len(material_ids)!=6
            or any(type(i) is not int or not 0<=i<len(nodes) for i in material_ids)):
        raise ValueError('invalid fixed material anchors')
    result=[]
    for index,side in enumerate(('left','left','left','right','right','right')):
        t=np.asarray(poses[side],dtype=float)
        if (t.shape!=(4,4) or not np.isfinite(t).all()
                or not np.allclose(t[3],[0,0,0,1],atol=1e-8,rtol=0)
                or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6,rtol=0)
                or not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-6,rtol=0)):
            raise ValueError('invalid measured palm frame')
        result.append((nodes[material_ids[index]]-t[:3,3])@t[:3,:3])
    return np.asarray(result)
