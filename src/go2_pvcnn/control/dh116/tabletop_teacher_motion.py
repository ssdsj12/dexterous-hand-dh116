"""Observable teacher motion; visual markers never participate in physics."""
import torch
import math


def _finite_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,list):return [_finite_json(x) for x in value]
    if isinstance(value,dict):return {k:_finite_json(v) for k,v in value.items()}
    return value


def motion_snapshot(env):
    t,w=env.task,env.world
    center=t.m['contact_center_relative_m']
    valid=t.m['contact_valid']&torch.isfinite(center).all(-1)
    points=center.cpu().tolist();valid_cpu=valid.cpu().tolist()
    for i,row in enumerate(points):
        for h in range(2):
            if not valid_cpu[i][h]:row[h]=None
    row=dict(phase=t.phase.cpu().tolist(),episode=env.episodes.cpu().tolist(),
        elapsed_s=t.elapsed.cpu().tolist(),
        lift_command_m=(t.offset-w.initial_offset).cpu().tolist(),
        palm_actual_z_m=(w.roots[:,:,2]-w.origins[:,None,2]).cpu().tolist(),
        palm_target_z_m=(w.root_targets[:,:,2]-w.origins[:,None,2]).cpu().tolist(),
        contact_center_relative_m=points,contact_valid=valid_cpu,
        hand_load_n=t.m['hand_object_loads'].cpu().tolist(),
        normal_z_n=t.m['vertical_hand_force_n'].cpu().tolist(),
        force_target_n=t.f_target.cpu().tolist(),relative_slip_m=t.slip.cpu().tolist())
    if hasattr(t,'lift_diagnostic_masks'):
        row['lift_gates']=t.lift_diagnostic_masks.cpu().tolist()
    if hasattr(t,'close_grasp_hold'):
        row.update(close_holding=t.close_grasp_hold.active.cpu().tolist(),
                   close_ready=t.close_grasp_hold.ready.cpu().tolist())
    if hasattr(t,'held_safety_correction'):
        row['held_safety_correction']=t.held_safety_correction.cpu().tolist()
    if hasattr(t,'lift_grasp_hold'):
        for name in ('recovering','contact_loss_s','recovery_stable_s'):
            if hasattr(t.lift_grasp_hold,name):
                row[name]=getattr(t.lift_grasp_hold,name).cpu().tolist()
    return _finite_json(row)


class TeacherMotionView:
    """Show environment zero's force-weighted contact centers and live state."""
    def __init__(self):
        import omni.usd
        import omni.ui as ui
        from pxr import UsdGeom,Gf
        stage=omni.usd.get_context().get_stage()
        self.markers=[]
        for h,color in enumerate(((0.,1.,1.),(1.,0.,1.))):
            sphere=UsdGeom.Sphere.Define(stage,f'/World/TeacherContactCenter{h}')
            sphere.CreateRadiusAttr(.002)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            self.markers.append((sphere,sphere.AddTranslateOp()))
            sphere.MakeInvisible()
        self.window=ui.Window('Teacher grasp status - env 0',width=450,height=210)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label('Measured contact centers: cyan LEFT / magenta RIGHT',word_wrap=True)
                ui.Label('Markers are force-weighted centers, not attachment points.',word_wrap=True)
                self.status=ui.Label('',word_wrap=True)

    def update(self,env,row):
        from pxr import Gf
        positions=env.world.contact_center_w[0].cpu().tolist()
        for h,(sphere,translation) in enumerate(self.markers):
            if row['contact_valid'][0][h]:
                translation.Set(Gf.Vec3d(*positions[h]));sphere.MakeVisible()
            else:sphere.MakeInvisible()
        phase={1:'CLOSE',2:'HOLD',3:'LIFT'}.get(row['phase'][0],str(row['phase'][0]))
        if row['phase'][0]==3 and row.get('recovering',[False])[0]:
            phase+=' / GRIP RECOVERY'
        load=row['hand_load_n'][0];actual=row['palm_actual_z_m'][0]
        self.status.text=(f'Env 0 | Episode {row["episode"][0]+1} | {phase}\n'
            f'Sim {row["elapsed_s"][0]:.2f} s\n'
            f'Lift command: {row["lift_command_m"][0]*1000:.1f} mm\n'
            f'Actual palm Z: L {actual[0]*1000:.1f} / R {actual[1]*1000:.1f} mm\n'
            f'Contact load: L {load[0]:.2f} / R {load[1]:.2f} N\n'
            f'Object/hand relative slip: {row["relative_slip_m"][0]*1000:.1f} mm')

    def close(self):
        self.window.destroy()
