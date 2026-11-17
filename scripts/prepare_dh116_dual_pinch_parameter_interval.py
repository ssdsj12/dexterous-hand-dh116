"""Parameter provenance added to the qualified director owner, with no new steps."""
from scripts import prepare_dh116_dual_pinch_director_interval as director
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

VERSION="dh116-dual-pinch-parameter-readback-v1"
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,16,18
PROFILE_SHA=None
action_schedule=director.action_schedule
sdk_clock_sample=director.sdk_clock_sample
run_director=director.run_existing_app
inspect_director=director.inspect_interval

def make_reader(nodes,device):
    from go2_pvcnn.control.dh116.tabletop_material_director_readback import MaterialDirectorReadback
    from go2_pvcnn.control.dh116.tabletop_flexible_parameter_readback import capture_parameters
    class ParameterDirectorReader(MaterialDirectorReadback):
        def _read(self,row,*,rope,backend,sim,read_clock):
            runtime=None
            try:
                runtime=getattr(read_clock,"__self__",None)
                if runtime is None or runtime.rope is not rope:
                    raise ValueError("same actual runtime/rope bound clock required for parameter readback")
                from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import _clock
                parameters=capture_parameters(runtime,sim=sim,backend=backend,
                    expected_clock=_clock(lambda:row),device=self.device)
                packet=super()._read(row,rope=rope,backend=backend,sim=sim,read_clock=read_clock)
                packet["parameter_readback"]=parameters
                return packet
            finally:
                self=row=rope=backend=sim=read_clock=runtime=None
    return ParameterDirectorReader(nodes,device=device)

def inspect_interval(path,*,complete=True):
    from scripts.validate_dh116_dual_pinch_parameter_interval import inspect_parameters
    return dict(inspect_director(path,complete=complete),
                parameter_audit=inspect_parameters(path,complete=complete))

def run_existing_app(*args,**kwargs):
    try:
        with bindings((director,"VERSION",VERSION),(director,"PROFILE_SHA",PROFILE_SHA),
                      (director,"make_reader",make_reader),(director,"inspect_interval",inspect_interval)):
            return run_director(*args,**kwargs)
    finally:
        args=kwargs=None
