"""Explicit multi-finger pressure sharing, nonslip acquisition and lift/hold.

Uses original position drives and a requested per-hand total force band.
No object attachment, external object force, modified physics or PPO dispatch.
"""
from pathlib import Path
import argparse,hashlib,runpy,sys,json
import torch
from go2_pvcnn.control.dh116 import tabletop_grip_lift as program
from go2_pvcnn.control.dh116 import tabletop_tensor_world as adapter
from go2_pvcnn.control.dh116 import tabletop_tensor_training as training
from go2_pvcnn.control.dh116.tabletop_multifinger_runtime import MultifingerLiftControl

options_parser=argparse.ArgumentParser(add_help=False)
options_parser.add_argument('--hand-force-range',type=float,nargs=2,default=(2.,3.))
options_parser.add_argument('--native-pad-compliance',action='store_true',
    help='experimental series-compliance feedback using actual native pad stiffness')
options,remaining=options_parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]
ROOT=Path(__file__).resolve().parents[1];state={}
OUTPUT=Path(sys.argv[sys.argv.index('--output')+1])
original_init=adapter.TensorWorld.__init__;original_advance=adapter.TensorWorld.advance
original_reset=adapter.TensorWorld.reset;original_write=training.write_json
OriginalDwell=program.LiftDwell
class FiveSecondDwell(OriginalDwell):
    def __init__(self,n,device,**kwargs):super().__init__(n,device,hold_s=5.,**kwargs)

def initialize(self,world):
    original_init(self,world);state['control']=MultifingerLiftControl(self,hand_force_range=options.hand_force_range,
        include_contact_compliance=options.native_pad_compliance)
    from torch.utils.tensorboard import SummaryWriter
    state['writer']=SummaryWriter(str(OUTPUT/'tensorboard'))
    state['writer'].add_text('configuration/hand_force_range_n',str(list(options.hand_force_range)),0)
def reset(self,ids):
    measured=original_reset(self,ids)
    if 'control' in state:state['control'].reset(ids)
    return measured
def advance(self,*args,**kwargs):
    measured=original_advance(self,*args,**kwargs);state['control'].after_physics(measured)
    control=state['control']
    if control.tick%24==1:
        writer=state['writer'];step=control.tick
        for case in range(control.n):
            for h,side in enumerate(('left','right')):
                prefix=f'case{case}/force/{side}'
                writer.add_scalar(prefix+'_total_n',float(measured['hand_object_loads'][case,h]),step)
                writer.add_scalar(prefix+'_requested_n',float(control.pressure[case]),step)
                writer.add_scalar(prefix+'_window_min_n',float(control.monitor.hand_force_min_n[case,h]),step)
                writer.add_scalar(prefix+'_window_max_n',float(control.monitor.hand_force_max_n[case,h]),step)
                for d,name in enumerate(('thumb','index','middle','ring','little')):
                    writer.add_scalar(f'case{case}/digits/{side}_{name}_n',float(measured['grouped_object_loads'][case,h,d]),step)
            writer.add_scalar(f'case{case}/state/stable',int(control.ready[case]),step)
            writer.add_scalar(f'case{case}/state/rod_lift_mm',float((self.bottom[case]-.30)*1000),step)
    if control.tick%240==1:
        status=control.summary();status.update(physics_steps=control.tick,phase=control.phase.cpu().tolist(),
            actual_digit_loads_n=measured['grouped_object_loads'][:,:,:5].cpu().tolist(),
            actual_hand_total_load_n=measured['hand_object_loads'].cpu().tolist(),
            requested_digit_loads_n=control.last['desired'].cpu().tolist(),
            maximum_raw_cv=control.monitor.force_cv.amax((1,2)).cpu().tolist(),
            bottom_lift_mm=((self.bottom-.30)*1000).cpu().tolist())
        original_write(OUTPUT/'multifinger_status.json',status)
        state['writer'].flush()
        print('MULTIFINGER',json.dumps(status),flush=True)
    return measured
def targets(command,seed,closed,elapsed,dt,**kwargs):return state['control'].targets(command,seed,dt)
def write(path,data):
    path=Path(path);control=state.get('control')
    if path.name=='resolved_grip_lift.json':
        files=['scripts/run_dh116_multifinger_lift.py']+['src/go2_pvcnn/control/dh116/'+name for name in
            ('tabletop_multifinger_control.py','tabletop_multifinger_monitor.py','tabletop_multifinger_runtime.py')]
        data.update(mode='multifinger_balanced_nonslip_lift',force_controller_executed=True,force_goal='per_hand_total_normal_band',
            native_pad_compliance=bool(getattr(options,'native_pad_compliance',False)),
            force_limits_used_as_success_gate=True,
            hand_force_range_n=list(options.hand_force_range),initial_hand_force_setpoint_n=sum(options.hand_force_range)/2,
            force_allocation='bearing_digits_share_budget;unloaded_eligible_digits_seek_separately;palm_load_subtracted',
            active_fingers='any_five;not_thumb_index_only',lift_start_s=None,lift_speed_m_s=.005,lift_goal_m=.03,
            acceptance='actual_rod_bottom30mm;stable_balanced_native_digit_contact;anchor_slip<=0.5mm;continuous5s',
            controller_source_lf_sha256={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in files},
            geometry_sampling_hz=60,raw_force_and_slip_sampling_hz=round(1/control.dt) if control is not None else 480,
            stable_window_raw_steps=control.monitor.window_steps if control is not None else 144,
            source_depth_retreat_m=.0015,native_depth_retreat_m=.0008,
            maximum_raw_loaded_digit_ratio=1.5,
            retreat_is_trigger_not_hard_penetration_bound=True)
    elif path.name.startswith('lift_case_'):
        # The shared driver's first event is height-only. Do not publish it as
        # a nonslip success until the same raw five-second window is audited.
        return None
    elif path.name=='grip_lift_result.json':
        if control.samples:
            torch.save({k:torch.stack([s[k] for s in control.samples]).cpu() for k in control.samples[0]},path.parent/'multifinger_trace.pt')
        if control.models:
            torch.save({k:torch.stack([s[k] for s in control.models]).cpu() for k in control.models[0]},path.parent/'contact_models.pt')
        if 'writer' in state:state['writer'].flush();state['writer'].close()
        data['height_only_summary']={k:data[k] for k in ('hold_s','qualified','ever_qualified','successful_cases','last_good','longest_hold_s')}
        data.update(control.summary())
        data.update(last_good=(control.hold>0).cpu().tolist(),longest_hold_s=control.longest.cpu().tolist(),
            teacher_export='deferred_until_independent_same_window_force_slip_geometry_audit')
    return original_write(path,data)

adapter.TensorWorld.__init__=initialize;adapter.TensorWorld.advance=advance;adapter.TensorWorld.reset=reset
program.advance_program=targets;program.LiftDwell=FiveSecondDwell;training.write_json=write
runpy.run_path(str(ROOT/'scripts/run_dh116_grip_lift.py'),run_name='__main__')
