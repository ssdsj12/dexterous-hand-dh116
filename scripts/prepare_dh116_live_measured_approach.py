"""Opt-in same-clock measured approach for the existing post-reset R0 owner.

No SDK imports, actions, clock advances, reward coefficients or legacy defaults
are changed. This connector does not establish native or Teacher qualification.
"""
from functools import partial
from scripts import prepare_dh116_live_r0_native as operation
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116.tabletop_r0_measurement_origin import OriginLiveR0Observer
from go2_pvcnn.control.dh116.tabletop_effective_right_r0_approach import (
    EffectiveRightMeasuredApproach,VERSION as SOURCE_VERSION,GEOMETRY_VERSION)
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _copy,_digest

VERSION="dh116-live-measured-approach-v1"

class MeasuredApproachObserver(OriginLiveR0Observer):
    """Uses actual completed reset and same-step geometry; rejects injections."""
    def __init__(self,*args,project,parameter_packet,**kwargs):
        super().__init__(*args,**kwargs)
        self.approach=EffectiveRightMeasuredApproach(project)
        self.parameter_packet=_copy(parameter_packet)
        self.reset_distance=None

    def complete_physical_reset(self,event,*,reset_approach):
        try:
            if reset_approach is not None:
                raise ValueError("caller reset approach cannot replace measured source")
            measured=self.approach.capture_reset(event["reset"],self.parameter_packet,grasp=self._grasp)
            super().complete_physical_reset(event,reset_approach=measured["reset_approach"])
            contract=dict(version=VERSION,source_version=SOURCE_VERSION,
                geometry_version=GEOMETRY_VERSION,reset_baseline_available=measured["reset_baseline_available"],
                actual_reset_measurement=measured,parameter_packet_sha256=_digest(self.parameter_packet),
                native_qualification=False,teacher_admission=False)
            self.initialization["measured_approach_contract"]=contract
            self._reset_audit=_copy(self.initialization)
            self.reset_distance=_copy(measured)
            self.parameter_packet=None
            return _copy(self.initialization)
        except Exception:
            self._state="failed"
            raise

    def record(self,event,*,auxiliary=None):
        try:
            if auxiliary is not None:
                raise ValueError("caller auxiliary cannot replace measured source")
            if self.reset_distance is None or event.get("kind")!="physical":
                raise ValueError("actual completed reset then physical event required")
            measured=self.approach.capture(event["exchange"],grasp=self._grasp)
            super().record(event,auxiliary=measured["auxiliary"])
            self._rows[-1]["pair_diagnostics"]["measured_approach"]=_copy(measured)
        except Exception:
            self._state="failed"
            raise

    def end(self,control_result):
        out=super().end(control_result)
        # This field describes the actual reset source, never infers contact.
        out["approach_first_sample_is_unavailable"]=not self.reset_distance["reset_baseline_available"]
        self._controls[-1]["approach_first_sample_is_unavailable"]=out["approach_first_sample_is_unavailable"]
        return out

class MeasuredApproachSession(operation.LiveSession):
    """Delegate original initialization and I/O; bind owned parameter packet."""
    def initialize(self,packet,error,identity,write_original):
        factory=partial(MeasuredApproachObserver,project=self.project,parameter_packet=packet)
        with bindings((operation,"LiveR0Observer",factory)):
            return super().initialize(packet,error,identity,write_original)
