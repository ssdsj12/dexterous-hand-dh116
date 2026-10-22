"""Measured-load feedback for the hand MPC's radial contact reference.

This changes commanded closure, not contact stiffness or measured forces.
"""
import numpy as np


class PreloadTargetHold:
    """Pause target chasing after measured preload, not a grasp qualifier.

    A hand resumes regulation on load/contact loss. The independent contact
    observer must still establish separation, dwell, slip and lifting success.
    """
    def __init__(self,target_loads,*,active_mask):
        self.targets=np.asarray(target_loads,dtype=float).reshape(2,3)
        self.active=np.asarray(active_mask,dtype=bool).reshape(2,3)
        self.held=np.zeros(2,dtype=bool)
        self._window_count=0
        self.last_window=None

    def observe(self,loads_n,gaps_m,known):
        """Retain actual physics feedback without making a motor decision."""
        loads=np.asarray(loads_n,dtype=float).reshape(2,3)
        gaps=np.asarray(gaps_m,dtype=float).reshape(2,3)
        measured=np.asarray(known,dtype=bool).reshape(2,3)
        if self._window_count==0:
            self._window_loads=loads.copy()
            self._window_gaps=gaps.copy()
            self._window_known=measured.copy()
        else:
            self._window_loads=np.minimum(self._window_loads,loads)
            self._window_gaps=np.maximum(self._window_gaps,gaps)
            self._window_known&=measured
        self._window_count+=1

    def update(self,loads_n,gaps_m,known):
        loads=np.asarray(loads_n,dtype=float).reshape(2,3)
        gaps=np.asarray(gaps_m,dtype=float).reshape(2,3)
        measured=np.asarray(known,dtype=bool).reshape(2,3)
        if self._window_count:
            loads=np.minimum(loads,self._window_loads)
            gaps=np.maximum(gaps,self._window_gaps)
            measured=measured & self._window_known
        self.last_window=dict(sample_count=self._window_count,
            minimum_loads_n=loads.tolist(),maximum_gaps_m=gaps.tolist(),known=measured.tolist())
        contact=(measured & (gaps<=1e-7)) | ~self.active
        # Use the existing physical grasp load requirement. A separate95%
        # capture gate keeps chasing a target even when measured preload is
        # already sufficient for the independent stable-contact observer.
        self.held=np.all(contact & ((loads>=.9*self.targets) | ~self.active),axis=1)
        self._window_count=0
        return np.repeat(self.held,6)


class GripLoadReference:
    def __init__(self, target_loads, *, active_mask=None, contact_stiffness=1e4,
                 gain_m_per_ns=.005, maximum_penetration_m=.0006,initial_depth_m=None):
        self.targets = np.asarray(target_loads, dtype=float).reshape(2, 3)
        self.active = (np.ones((2, 3), dtype=bool) if active_mask is None else
                       np.asarray(active_mask, dtype=bool).reshape(2, 3))
        self.nominal = np.where(self.active, self.targets / contact_stiffness, 0.)
        self.depth = self.nominal.copy()
        self.gain = gain_m_per_ns
        self.maximum = maximum_penetration_m
        if initial_depth_m is not None:
            initial=np.asarray(initial_depth_m,dtype=float)
            if (initial.shape!=(2,3) or not np.isfinite(initial).all() or np.any(initial<0.)
                    or np.any(initial>self.maximum+1e-12) or np.any(initial[~self.active]!=0.)):
                raise ValueError('initial load reference must match active channels and depth bounds')
            self.depth=initial.copy()

    def update(self, loads_n, gaps_m, dt,*,held_hands=None):
        loads = np.asarray(loads_n, dtype=float).reshape(2, 3)
        gaps = np.asarray(gaps_m, dtype=float).reshape(2, 3)
        contacting = self.active & (gaps <= 0.)
        underloaded=contacting & (loads<self.targets)
        self.depth[underloaded]=np.maximum(self.depth[underloaded],-gaps[underloaded])
        reached=contacting & (loads>=self.targets)
        self.depth[reached]=np.minimum(self.depth[reached],-gaps[reached])
        self.depth[contacting] += self.gain * (self.targets - loads)[contacting] * dt
        # An open gap is an approach target, not a reason to wind up force.
        self.depth[self.active & (gaps > .0002)] = self.nominal[self.active & (gaps > .0002)]
        self.depth = np.where(self.active, np.clip(self.depth, 0., self.maximum), 0.)
        if held_hands is not None:
            held=np.repeat(np.asarray(held_hands,dtype=bool).reshape(2,1),3,axis=1) & self.active
            self.depth[held]=np.clip(-gaps[held],0.,self.maximum)
        return self.depth.copy()
