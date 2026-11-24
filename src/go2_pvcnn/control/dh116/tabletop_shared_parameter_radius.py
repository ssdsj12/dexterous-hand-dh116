"""One same-SDK full-parent getter for the unchanged per-lane geometry math."""
from .tabletop_shared_teacher_source import SharedTeacherSource, FAMILY
from .tabletop_native_teacher_context import ContextIdentity


class SharedParameterRadius:
    def __init__(self, source):
        if type(source) is not SharedTeacherSource:
            raise ValueError("actual shared Teacher source required")
        self.source=source
        self._boundary=self._value=None

    def capture(self):
        before=self.source._boundary()
        raw=self.source.runtime.solver.parameter_snapshot()
        self.source._parameters(raw,before)
        if before != self.source._boundary():
            raise ValueError("shared material capture changed physical boundary")
        self._boundary=before
        self._value=float(raw["contact_radius_m"])

    def read(self, exchange):
        b=self.source._boundary()
        if b != self._boundary or self._value is None:
            raise ValueError("fresh same-SDK full-parent radius capture required")
        rod=exchange["rod"]
        i=rod["env_id"]
        if type(i) is not int or not 0 <= i < b.parent_num_envs:
            raise ValueError("actual shared env row required")
        current=ContextIdentity.from_clock(rod)
        expected=ContextIdentity(b.run_uuid,i,b.episode_uids[i],b.reset_ids[i],
            b.rod_uids[i],b.sdk_step,b.rod_ticks[i],b.policy_time_s,b.solver_time_s)
        if current != expected:
            raise ValueError("radius source has a different row/clock/Reset epoch")
        return dict(value=self._value,source=dict(api="RopeBatch.parameter_snapshot.contact_radius_m",
            source_family=FAMILY,parent_solver_id=b.parent_solver_id,parent_num_envs=b.parent_num_envs,
            env_row=i,backend_sha256=self.source.profile.binary_sha256,
            run_uuid=b.run_uuid,episode_uid=b.episode_uids[i],reset_id=b.reset_ids[i],
            physics_step=b.sdk_step,rod_tick=b.rod_ticks[i],solver_time_s=b.solver_time_s,
            uncertainty=None,hardware_measured=False,task_physics_qualified=False))


from .tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchGeometry


from .tabletop_owned_memory_geometry import OwnedGeometryMode


class SharedPinchGeometry(OwnedGeometryMode,PolicyOwnedPinchGeometry):
    """Only replaces the radius source; all original guard math is inherited."""
    def __init__(self, project, *, scope, radius_source,fast_memory=False):
        if type(radius_source) is not SharedParameterRadius or scope != radius_source.source.scope:
            raise ValueError("same-scope actual shared radius source required")
        self.radius_source=radius_source
        self.fast_memory=bool(fast_memory);self._frozen_sources=None
        super().__init__(project,scope=scope)

    def _radius(self, exchange):
        return self.radius_source.read(exchange)

from .tabletop_native_teacher_policy_tripod_distance import PolicyOwnedTripodGeometry


class SharedTripodGeometry(OwnedGeometryMode,PolicyOwnedTripodGeometry):
    """Original Tripod17 guards with an explicit full-parent radius source."""
    def __init__(self,project,*,scope,radius_source,fast_memory=False,**kwargs):
        if type(radius_source) is not SharedParameterRadius or scope != radius_source.source.scope:
            raise ValueError("same-scope actual shared radius source required")
        self.radius_source=radius_source
        self.fast_memory=bool(fast_memory);self._frozen_sources=None
        super().__init__(project,scope=scope,**kwargs)

    def _radius(self,exchange):
        return self.radius_source.read(exchange)
