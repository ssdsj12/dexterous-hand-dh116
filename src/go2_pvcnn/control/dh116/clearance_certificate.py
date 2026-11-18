"""Conservative transport of a previously verified geometric clearance.

Hausdorff distance between corresponding convex hulls is bounded by the
maximum vertex displacement. Likewise, corresponding piecewise-linear wire
segments move no farther than their endpoints. Subtract BOTH body shifts for
robot pairs, and wire/radius changes as well. This deliberately overestimates
the loss for every pair. Never update the reference frame after reuse.
"""
import numpy as np


class ClearanceCertificate:
    def __init__(self,world,wire,*,lower_bound,wire_radius,wire_motion):
        self.world={k:np.array(v,dtype=float,copy=True) for k,v in world.items()}
        self.wire=np.array(wire,dtype=float,copy=True)
        self.bound=float(lower_bound);self.radius=float(wire_radius);self.motion=float(wire_motion)
        if not np.isfinite(self.bound):raise ValueError('invalid certificate bound')
        self.lower_bound(self.world,self.wire,wire_radius=wire_radius,wire_motion=wire_motion)

    def lower_bound(self,world,wire,*,wire_radius,wire_motion):
        if set(world)!=set(self.world) or not world:
            raise ValueError('certificate body topology changed')
        shift=0.
        for key,old in self.world.items():
            new=np.asarray(world[key],dtype=float)
            if (new.shape!=old.shape or new.ndim!=2 or new.shape[1]!=3 or not len(new)
                    or not np.isfinite(new).all()):raise ValueError('invalid certificate vertices')
            shift=max(shift,float(np.linalg.norm(new-old,axis=1).max()))
        wire=np.asarray(wire,dtype=float)
        if (wire.shape!=self.wire.shape or wire.ndim!=2 or wire.shape[1]!=3 or len(wire)<2
                or not np.isfinite(wire).all() or not np.isfinite([wire_radius,wire_motion]).all()
                or wire_radius<=0 or wire_motion<0):raise ValueError('invalid certificate wire')
        loss=(2*shift+float(np.linalg.norm(wire-self.wire,axis=1).max())
              +max(0.,wire_radius-self.radius)+max(0.,wire_motion-self.motion))
        return self.bound-loss
