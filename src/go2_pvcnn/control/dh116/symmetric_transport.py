"""Stage-local measured-height coordinator, not a generic task constraint.

Instantiate on entry to synchronized lift (or explicit return demo). Other
tasks bypass this object and use independent bilateral transport commands.
All constants are simulation tuning, not manufacturer hardware ratings.
"""
import numpy as np


class SynchronizedTransport:
    def __init__(self,*,max_speed_m_s=.005):
        if not np.isfinite(max_speed_m_s) or not 0<max_speed_m_s<=.01:
            raise ValueError('simulation lift speed must be in (0, .01] m/s')
        self.max_speed_m_s=float(max_speed_m_s)
        self.last_time=None
        self.started=None
        self.direction=None
        self.target=None
        self.waiting_side=None

    def update(self,time_s,*,rises_m,contacts_valid,direction):
        h=np.asarray(rises_m,dtype=float)
        valid=np.asarray(contacts_valid)
        if (h.shape!=(2,) or not np.isfinite(h).all()
                or valid.shape!=(2,) or valid.dtype.kind!='b' or not valid.all()
                or type(direction) is not int or direction not in (-1,1)
                or not np.isfinite(time_s) or time_s<0):
            raise ValueError('finite measured heights, bilateral contacts and direction required')
        if self.last_time is not None and not 0<time_s-self.last_time<=.01+1e-12:
            raise ValueError('transport evidence must advance contiguously')
        difference=float(h[0]-h[1])
        if abs(difference)>=.002-1e-12:
            raise ValueError('bilateral height difference reached 2 mm')
        progress=direction*h
        fresh=self.direction!=direction
        started=float(time_s) if fresh else self.started
        dt=0. if fresh else float(time_s-self.last_time)
        target=float(progress.min()) if fresh else self.target
        waiting=None if fresh else self.waiting_side
        gap=abs(difference)
        if gap>=.001-1e-12:
            waiting=int(np.argmax(progress))
        elif gap<=.0005+1e-12:
            waiting=None
        speed=self.max_speed_m_s*float(np.clip((time_s-started)/.2,0.,1.))
        target=min(target+speed*dt,float(progress.min())+.001)
        velocity=np.zeros((2,3))
        if dt:
            # Positive progress only; never reverse a leading hand to correct.
            remaining=np.maximum(target-progress,0.)/dt
            error=progress.mean()-progress
            correction=np.where(np.abs(error)>.00025,np.clip(2.*error,-.001,.001),0.)
            signed=np.minimum(np.clip(speed+correction,0.,self.max_speed_m_s),remaining)
            if waiting is not None:signed[waiting]=0.
            velocity[:,2]=direction*signed
        self.last_time=float(time_s)
        self.started=started;self.direction=direction
        self.target=target;self.waiting_side=waiting
        return dict(velocity_w=velocity,target_progress_m=target,
                    height_difference_m=difference,waiting_side=waiting,
                    direction=direction)
