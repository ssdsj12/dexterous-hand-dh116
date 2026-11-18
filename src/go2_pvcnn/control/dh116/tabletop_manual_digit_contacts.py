"""Conservative per-digit raw contact manifold diagnostics, manual mode only."""
import math
import numpy as np


def summarize(records,loads,center,axis):
    center=np.asarray(center,dtype=float);axis=np.asarray(axis,dtype=float)
    if center.shape!=(3,) or axis.shape!=(3,) or not np.isfinite([center,axis]).all() or np.linalg.norm(axis)==0:
        raise ValueError('finite center and nonzero object axis required')
    axis=axis/np.linalg.norm(axis)
    result={}
    for hand in ('left','right'):
        for digit in ('thumb','index'):
            selected=[r for r in records if r['hand']==hand and r['digit']==digit]
            radial=[];depth=[];bad=False
            for r in selected:
                p=np.asarray(r['position_w'],dtype=float)-center
                if p.shape!=(3,) or not np.isfinite(p).all():bad=True;continue
                radial.append(float(p[2]-np.dot(p,axis)*axis[2]))
                s=r.get('separation_m')
                if s is not None and math.isfinite(s):depth.append(max(0.,-s))
            load=float(loads[hand][digit])
            valid=math.isfinite(load) and load>=.05 and bool(radial) and not bad
            below=valid and max(radial)<-1e-6
            result[hand+'_'+digit]=dict(load_n=load,contact_point_count=len(radial),
                all_points_below_axis=bool(below),has_loaded_contact=bool(valid),
                min_radial_z_mm=min(radial)*1000 if radial else None,
                max_radial_z_mm=max(radial)*1000 if radial else None,
                mean_radial_z_mm=float(np.mean(radial))*1000 if radial else None,
                max_penetration_mm=max(depth)*1000 if depth else None)
    return dict(digits=result,all_four_digits_below_axis=all(r['all_points_below_axis'] for r in result.values()),
        source='all_raw_manifold_points_of_object_loaded_links; no_averaging_for_pass_fail',
        missing_or_unloaded_finger_passes=False)


def collect(env):
    from .tabletop_object_contact import link_groups
    w=env.world;legacy=w.w
    if w.n!=1:raise ValueError('manual raw-digit view requires one environment')
    groups=link_groups(legacy.object_contact_links)
    link_loads=env.task.m['object_link_loads'][0].detach().cpu().numpy()
    records=[];loads={}
    for hi,hand in enumerate(('left','right')):
        loads[hand]={}
        for gi,digit in enumerate(('thumb','index')):
            loads[hand][digit]=float(link_loads[hi,groups==gi].sum())
            for li in np.nonzero(groups==gi)[0]:
                if link_loads[hi,li]<.05:continue
                sensor=legacy.sensors[hi*len(groups)+int(li)]
                force,point,normal,separation,count,start=sensor.contact_physx_view.get_contact_data(dt=w.dt)
                number=int(count.reshape(-1)[0]);begin=int(start.reshape(-1)[0])
                pts=point[begin:begin+number].detach().cpu().tolist()
                sep=separation[begin:begin+number].reshape(-1).detach().cpu().tolist()
                for p,s in zip(pts,sep):records.append(dict(hand=hand,digit=digit,position_w=p,separation_m=s))
    center=w.wp[0].detach().cpu().numpy()
    qw,qx,qy,qz=w.wq[0].detach().cpu().tolist()
    axis=np.array([2*(qx*qy-qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz+qw*qx)])
    result=summarize(records,loads,center,axis)
    result['points']=records
    return result,center,axis


class Visuals:
    def __init__(self):
        from pxr import UsdGeom
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        self.points=UsdGeom.Points.Define(stage,'/World/ManualDiagnostics/DigitContacts')
        self.points.MakeInvisible()
        self.lines=UsdGeom.BasisCurves.Define(stage,'/World/ManualDiagnostics/AxisHeightReferences')
        self.lines.CreateTypeAttr('linear');self.lines.CreateWrapAttr('nonperiodic')
        self.lines.CreateCurveVertexCountsAttr([2,2]);self.lines.CreateWidthsAttr([.0004])
        self.lines.CreatePointsAttr([(0.,0.,0.)]*4)
        self.lines.MakeInvisible()
        self.lines.SetWidthsInterpolation('constant')
        self.lines.CreateDisplayColorAttr([(.05,.9,.15)])

    def update(self,row,center,axis):
        from pxr import Gf,UsdGeom
        colors={('left','thumb'):(1.,.25,.05),('left','index'):(.05,.7,1.),
                ('right','thumb'):(1.,.15,.8),('right','index'):(1.,.85,.05)}
        points=row['points']
        if points:
            self.points.MakeVisible()
            self.points.CreatePointsAttr([Gf.Vec3f(*r['position_w']) for r in points])
            self.points.CreateWidthsAttr([.0015])
            self.points.SetWidthsInterpolation('constant')
            self.points.CreateDisplayColorAttr([Gf.Vec3f(*colors[(r['hand'],r['digit'])]) for r in points])
            UsdGeom.PrimvarsAPI(self.points.GetPrim()).GetPrimvar('displayColor').SetInterpolation('vertex')
        else:
            self.points.MakeInvisible()
        # Parallel rails show the actual axis HEIGHT outside the opaque rod.
        side=np.cross(axis,[0.,0.,1.]);norm=np.linalg.norm(side)
        if norm<1e-8:side=np.array([1.,0.,0.])
        else:side/=norm
        verts=[center+sign*.0105*side+end*.25*axis for sign in (-1,1) for end in (-1,1)]
        self.lines.CreatePointsAttr([Gf.Vec3f(*p) for p in verts])
        self.lines.MakeVisible()
