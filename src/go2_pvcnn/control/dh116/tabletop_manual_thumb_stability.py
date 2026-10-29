"""Stability of both measured thumb loads, in simulation time, not hand totals."""
from collections import deque
import numpy as np

CONTRACT=dict(window_s=.3,minimum_thumb_load_n=.5,absolute_range_n=.1,
    relative_range_of_mean=.05,scope='both_thumbs_individually',
    zero_force_is_not_grasp=True)


class ThumbStability:
    def __init__(self):self.reset()

    def reset(self):
        self.samples=deque();self.duration=0.
        self.result=dict(CONTRACT,thumb_force_stable=[False,False],
                         observed_window_s=0.,thumb_range_n=[0.,0.],thumb_mean_n=[0.,0.])

    def update(self,digit_loads,dt):
        loads=np.asarray(digit_loads,dtype=float)
        if loads.shape!=(2,2) or not np.isfinite(loads).all() or (loads<0).any() or not np.isfinite(dt) or dt<=0:
            raise ValueError('finite nonnegative measured digit loads and positive simulation dt required')
        self.samples.append((dt,loads[:,0].copy()));self.duration+=dt
        while len(self.samples)>1 and self.duration-self.samples[0][0]>=CONTRACT['window_s']-1e-12:
            old,_=self.samples.popleft();self.duration-=old
        values=np.stack([sample[1] for sample in self.samples])
        span=np.ptp(values,axis=0);mean=values.mean(0)
        stable=(values.min(0)>=CONTRACT['minimum_thumb_load_n'])
        stable &= span<=np.maximum(CONTRACT['absolute_range_n'],CONTRACT['relative_range_of_mean']*mean)
        stable &= self.duration>=CONTRACT['window_s']-1e-12
        self.result=dict(CONTRACT,thumb_force_stable=stable.tolist(),
                         observed_window_s=self.duration,thumb_range_n=span.tolist(),thumb_mean_n=mean.tolist())
        return self.result
