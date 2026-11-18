"""Opt-in constrained repair of the existing local pad MPC's soft solution.

This is a kinematic feasibility calculation, not force/contact/object dynamics.
The original soft trajectory is retained independently for every failed hand.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch

from .linear_coupling import active_limits
from .tabletop_surface_mpc import SurfaceGripMPC


class GeometricConstrainedMPC(SurfaceGripMPC):
    def __init__(self, horizon=4, *, solver_maxiter=150, **kwargs):
        super().__init__(horizon, **kwargs)
        if type(solver_maxiter) is not int or solver_maxiter < 1:
            raise ValueError('solver_maxiter must be a positive integer')
        self.solver_maxiter = solver_maxiter
        self.geometry_context = None

    def set_geometry_context(self, context):
        """Set this call's context; inputs are read only, never consumed or edited."""
        if context is not None and not isinstance(context, dict):
            raise ValueError('geometry context must be a dictionary or None')
        self.geometry_context = context

    def _context(self, q, points, weights):
        ctx = self.geometry_context
        if not isinstance(ctx,dict): raise ValueError('geometry context must be a dictionary')
        n,d = q.shape[0],points.shape[2]
        for name,shape,boolean in (
                ('geometry_mask',(n,2,d),True), ('object_pose',(n,7),False),
                ('cad_outward',(n,2,d,3),False), ('pressure_inward',(n,2,d,3),False),
                ('pressure_protected',(n,2,d),True)):
            value=ctx.get(name)
            if (not isinstance(value,torch.Tensor) or value.shape!=shape
                    or value.device!=q.device or value.dtype!=(torch.bool if boolean else q.dtype)):
                raise ValueError(name+' requires matching shape/dtype/device')
        for name in ('cylinder_radius_m','cylinder_length_m'):
            value=ctx.get(name)
            if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or value<=0:
                raise ValueError(name+' must be explicit, finite and positive')
        active=ctx['geometry_mask']&(weights>0)
        active_env=active.any(dim=(1,2))
        norm=ctx['object_pose'][:,3:].norm(dim=-1)
        if not bool((torch.isfinite(ctx['object_pose']).all(-1)&torch.isfinite(norm)&(norm>0))[active_env].all()):
            raise ValueError('object quaternion must have finite positive norm')
        for name,mask in (('cad_outward',active),
                ('pressure_inward',ctx['pressure_protected']&active.any(-1,keepdim=True))):
            norm=ctx[name].norm(dim=-1)
            if not bool((~mask | (torch.isfinite(norm)&(norm>0))).all()):
                raise ValueError(name+' must have a usable direction on constrained pads')
        return ctx

    def _evaluate(self, u, kw):
        q,command=kw['q'],kw['command'];response=float(kw.get('response',.5))
        delta=torch.zeros_like(q);states=[]
        for future in u.unbind(1):
            delta=(1-response)*delta+response*(future-command);states.append(delta)
        states=torch.stack(states,1)
        pred=kw['points'][:,None]+torch.einsum('nsdcj,nhsj->nhsdc',kw['jacobian'],states)+kw['wrist_motion']
        largest=kw['weights'].amax(-1,keepdim=True)
        relative=kw['weights']/torch.where(largest>0,largest,torch.ones_like(largest))
        normalized=relative/relative.sum(-1,keepdim=True).clamp_min(1.)
        geometry=(((pred-kw['target_points'])/self.point_scale).square().sum(-1)*normalized[:,None]).sum(-1).mean(1)
        tracking=((u-kw['nominal'])/self.max_command_step).square().mean((1,3))
        changes=torch.cat((u[:,:1]-command[:,None],u[:,1:]-u[:,:-1]),1)
        costs=geometry+self.nominal_weight*tracking+self.smoothness_weight*(changes/self.max_command_step).square().mean((1,3))
        return costs,q[:,None]+states,pred

    @staticmethod
    def _axis(pose):
        q=pose[3:]/np.linalg.norm(pose[3:]);w,x,y,z=q
        return np.array((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)))

    def _check_hand(self, kw, ctx, mask, env, hand, candidate):
        """Check original constraints in the issued dtype, without solve margins."""
        response=float(kw.get('response',.5));step=self.max_command_step
        radius=float(ctx['cylinder_radius_m']);half=float(ctx['cylinder_length_m'])/2
        delta=torch.zeros_like(kw['q'][env,hand]);pred=[];relative=[]
        for future,wrist in zip(candidate,kw['wrist_motion'][env,:,hand]):
            delta=(1-response)*delta+response*(future-kw['command'][env,hand])
            move=torch.einsum('dcj,j->dc',kw['jacobian'][env,hand],delta)
            relative.append(move);pred.append(kw['points'][env,hand]+move+wrist)
        pred=torch.stack(pred);relative=torch.stack(relative)
        axis=self._axis(ctx['object_pose'][env].detach().cpu().double().numpy())
        axis_t=kw['q'].new_tensor(axis);off=pred-ctx['object_pose'][env,:3]
        axial_t=(off*axis_t).sum(-1);radial_t=off-axial_t[...,None]*axis_t
        length_t=radial_t.norm(dim=-1);gap_t=length_t-radius
        normal_t=ctx['cad_outward'][env,hand]
        normal_t=normal_t/normal_t.norm(dim=-1,keepdim=True).clamp_min(1e-30)
        facing_t=(-radial_t/length_t.clamp_min(1e-30)[...,None]*normal_t).sum(-1)
        initial_t=kw['points'][env,hand]-ctx['object_pose'][env,:3]
        initial_t=initial_t-(initial_t*axis_t).sum(-1,keepdim=True)*axis_t
        original_t=initial_t.norm(dim=-1)-radius;active=mask[env,hand]
        geom_ok=(torch.isfinite(gap_t)&(gap_t>=0)&(gap_t<original_t[None]-1e-10)
            &(facing_t>=.8)&(axial_t.abs()<half))[:,active].all()
        protected=ctx['pressure_protected'][env,hand]
        pressure=(relative*ctx['pressure_inward'][env,hand]).sum(-1)
        pressure_ok=(torch.isfinite(pressure[:,protected])&(pressure[:,protected]<=1e-10)).all()
        physical_lo,physical_hi=active_limits();q64=kw['q'][env,hand].double()
        low=torch.maximum(torch.maximum(kw['lower'][hand].double(),
            torch.as_tensor(physical_lo,dtype=torch.float64,device=candidate.device)),q64-.1)
        high=torch.minimum(torch.minimum(kw['upper'][hand].double(),
            torch.as_tensor(physical_hi,dtype=torch.float64,device=candidate.device)),q64+.1)
        if kw.get('reference_radius') is not None:
            low=torch.maximum(low,kw['nominal'][env,:,hand].double()-kw['reference_radius'])
            high=torch.minimum(high,kw['nominal'][env,:,hand].double()+kw['reference_radius'])
        increments=torch.cat((candidate[:1]-kw['command'][env,hand],candidate[1:]-candidate[:-1]))
        bounded=(candidate.double()>=low).all()&(candidate.double()<=high).all()&(increments.abs()<=step).all()
        ok=bool(torch.isfinite(candidate).all()&geom_ok&pressure_ok&bounded)
        detail=dict(exact_geometry_pass=bool(geom_ok),exact_pressure_pass=bool(pressure_ok),original_bounds_pass=bool(bounded),
            minimum_geometry_gap_m=float(gap_t[:,active].min()),
            minimum_geometry_improvement_m=float((original_t[None]-gap_t)[:,active].min()),
            maximum_slew_rad=float(increments.abs().max()),
            maximum_protected_relative_inward_m=float(pressure[:,protected].max()) if bool(protected.any()) else None)
        return ok,detail

    def _repair_hand(self, kw, ctx, mask, env, hand, soft):
        from scipy.optimize import Bounds, minimize
        as_np=lambda t: t.detach().cpu().double().numpy().copy()
        command=as_np(kw['command'][env,hand]);q=as_np(kw['q'][env,hand])
        nominal=as_np(kw['nominal'][env,:,hand]);points=as_np(kw['points'][env,hand])
        jac=as_np(kw['jacobian'][env,hand]);wrist=as_np(kw['wrist_motion'][env,:,hand])
        targets=as_np(kw['target_points'][env,:,hand]);weights=as_np(kw['weights'][env,hand])
        pose=as_np(ctx['object_pose'][env]);axis=self._axis(pose)
        outward=as_np(ctx['cad_outward'][env,hand]);inward=as_np(ctx['pressure_inward'][env,hand])
        protected=as_np(ctx['pressure_protected'][env,hand]).astype(bool)
        geom=as_np(mask[env,hand]).astype(bool);digits=np.flatnonzero(geom)
        H=self.horizon;V=H*6;step=self.max_command_step;scale=self.point_scale
        response=float(kw.get('response',.5));radius=float(ctx['cylinder_radius_m']);half=float(ctx['cylinder_length_m'])/2
        lag=np.arange(H)[:,None]-np.arange(H)[None,:]
        sensitivity=np.where(lag>=0,response*(1-response)**np.maximum(lag,0),0.)
        A=(sensitivity[:,None,None,:,None]*jac[None,:,:,None,:]*step).reshape(H,len(points),3,V)
        base=points[None]+wrist
        difference=np.eye(V)
        for t in range(1,H):difference[t*6:(t+1)*6,(t-1)*6:t*6]=-np.eye(6)
        nominal_x=((nominal-command)/step).reshape(V)
        normalized=weights/weights.max();normalized/=max(normalized.sum(),1.)
        def objective(x):
            residual=base+np.einsum('tdcv,v->tdc',A,x)-targets
            weighted=residual*normalized[None,:,None]/scale**2
            cost=np.sum(residual*weighted)/H+self.nominal_weight*np.mean((x-nominal_x)**2)+self.smoothness_weight*np.mean((difference@x)**2)
            grad=2/H*np.einsum('tdc,tdcv->v',weighted,A)
            grad+=2*self.nominal_weight/V*(x-nominal_x)+2*self.smoothness_weight/V*difference.T@(difference@x)
            return float(cost),grad
        physical_lo,physical_hi=active_limits()
        lower=np.maximum(np.maximum(as_np(kw['lower'][hand]),physical_lo),q-.1)
        upper=np.minimum(np.minimum(as_np(kw['upper'][hand]),physical_hi),q+.1)
        low=np.broadcast_to(lower,(H,6)).copy();high=np.broadcast_to(upper,(H,6)).copy()
        if kw.get('reference_radius') is not None:
            low=np.maximum(low,nominal-kw['reference_radius']);high=np.minimum(high,nominal+kw['reference_radius'])
        lo_x=((low-command)/step).reshape(V);hi_x=((high-command)/step).reshape(V)
        # Tighten numerical solves, never relax authoritative physical/slew bounds.
        qmargin=8*torch.finfo(kw['q'].dtype).eps*max(1.,float(np.abs(command).max()))
        rate_limit=1-qmargin/step
        projection=np.eye(3)-np.outer(axis,axis)
        initial_radial=(points-pose[:3])@projection
        initial_length=np.linalg.norm(initial_radial,axis=-1)
        original_gap=initial_length-radius
        detail=dict(attempted=True,accepted=False,solver='scipy_SLSQP',maxiter=self.solver_maxiter,
            variables=V,status=None,message='',iterations=0,
            objective='original weighted geometry plus nominal and smoothness',
            constraint_scope='full horizon; exact radial upper/facing/sidewall and conservative radial tangent lower; relative protected pressure')
        if np.any(original_gap[digits]<=1e-10):
            detail.update(status=-1,message='initial geometry has no strictly improving nonnegative gap')
            return None,detail
        normals=outward[digits]/np.linalg.norm(outward[digits],axis=-1,keepdims=True)
        radial_base=(base[:,digits]-pose[:3])@projection
        radial_A=np.einsum('ab,tdbv->tdav',projection,A[:,digits])
        x0=((as_np(soft)-command)/step).reshape(V)
        soft_radial=radial_base+np.einsum('tdcv,v->tdc',radial_A,x0)
        soft_radius=np.linalg.norm(soft_radial,axis=-1)
        current_out=soft_radial/np.maximum(soft_radius,1e-30)[...,None]
        # Any unit radial tangent defines a conservative outer halfspace.
        # Use each future soft point's direction, so real tangential wrist
        # transport does not make a nanometre annulus artificially empty.
        current_out=np.where((soft_radius>0)[...,None],current_out,
            (initial_radial[digits]/initial_length[digits,None])[None])
        axial_base=np.einsum('tdc,c->td',base[:,digits]-pose[:3],axis)
        axial_A=np.einsum('tdcv,c->tdv',A[:,digits],axis)
        # Absolute metre margins remain smaller than the available initial gap.
        quantization_scale=(np.abs(points[digits]).max(axis=-1)
            +np.abs(jac[digits]).sum((1,2))*max(1.,float(np.abs(command).max())))
        margin=np.minimum(original_gap[digits]/4,
            np.maximum(1e-9,8*torch.finfo(kw['q'].dtype).eps*quantization_scale))
        improve=np.maximum(margin,2e-10)
        detail['numerical_margins']=dict(radial_lower_m=margin.tolist(),improvement_m=improve.tolist(),
            facing_cosine=1e-9,sidewall_m=1e-9,slew_rad=qmargin,protected_relative_m=2e-11)
        pressure_A=np.einsum('tdcv,dc->tdv',A[:,protected],inward[protected])
        def constraints(x):
            radial=radial_base+np.einsum('tdcv,v->tdc',radial_A,x)
            length=np.linalg.norm(radial,axis=-1)
            unit=radial/np.maximum(length,1e-30)[...,None]
            tangent=np.einsum('tdc,tdc->td',radial,current_out)
            facing_numerator=np.einsum('tdc,dc->td',radial,-normals)
            axial=axial_base+np.einsum('tdv,v->td',axial_A,x)
            v=[rate_limit-difference@x,rate_limit+difference@x,
                ((tangent-radius-margin)/scale).ravel(),
                ((radius+original_gap[digits]-improve-length)/scale).ravel(),
                ((facing_numerator-(.8+1e-9)*length)/scale).ravel(),
                ((half-1e-9-axial)/scale).ravel(),((half-1e-9+axial)/scale).ravel()]
            g=[-difference,difference,
                (np.einsum('tdc,tdcv->tdv',current_out,radial_A)/scale).reshape(-1,V),
                (-np.einsum('tdc,tdcv->tdv',unit,radial_A)/scale).reshape(-1,V),
                (np.einsum('tdc,tdcv->tdv',-normals[None]-(.8+1e-9)*unit,radial_A)/scale).reshape(-1,V),
                (-axial_A/scale).reshape(-1,V),(axial_A/scale).reshape(-1,V)]
            if protected.any():
                v.append(((8e-11-np.einsum('tdv,v->td',pressure_A,x))/scale).ravel())
                g.append((-pressure_A/scale).reshape(-1,V))
            return np.concatenate(v),np.concatenate(g)
        try:
            result=minimize(objective,np.clip(x0,lo_x,hi_x),jac=True,method='SLSQP',bounds=Bounds(lo_x,hi_x),
                constraints=[dict(type='ineq',fun=lambda x:constraints(x)[0],jac=lambda x:constraints(x)[1])],
                options=dict(maxiter=self.solver_maxiter,ftol=1e-12,disp=False))
            detail.update(status=int(result.status),message=str(result.message),iterations=int(result.nit),solver_success=bool(result.success),
                scaled_min_constraint_residual=float(constraints(result.x)[0].min()))
            candidate=kw['q'].new_tensor(command+step*result.x.reshape(H,6))
            # Re-evaluate with the caller's actual dtype/device and original
            # recurrence; numerical convergence alone never grants admission.
            ok,checks=self._check_hand(kw,ctx,mask,env,hand,candidate)
            detail.update(accepted=ok,**checks)
            return (candidate if ok else None),detail
        except (ValueError,RuntimeError,FloatingPointError) as error:
            detail.update(status=-2,message=str(error))
            return None,detail

    @torch.no_grad()
    def plan(self, **kwargs):
        first,soft=super().plan(**kwargs)
        if self.geometry_context is None:return first,soft
        ctx=self._context(kwargs['q'],kwargs['points'],kwargs['weights'])
        mask=ctx['geometry_mask']&(kwargs['weights']>0)
        if not bool(mask.any()):return first,soft
        trajectory=soft['trajectory'].clone()
        repair=torch.zeros_like(mask[...,0]);results=[]
        baseline_checked=mask.any(-1)
        baseline_feasible=torch.zeros_like(baseline_checked)
        for env in range(len(trajectory)):
            hands=[]
            for hand in range(2):
                if not bool(mask[env,hand].any()):
                    hands.append(dict(attempted=False,accepted=False,reason='no_active_geometry'));continue
                feasible,checks=self._check_hand(kwargs,ctx,mask,env,hand,trajectory[env,:,hand])
                baseline_feasible[env,hand]=feasible
                if feasible:
                    hands.append(dict(attempted=False,accepted=False,reason='already_feasible_soft',baseline_checks=checks));continue
                candidate,detail=self._repair_hand(kwargs,ctx,mask,env,hand,trajectory[env,:,hand])
                if candidate is not None:trajectory[env,:,hand]=candidate;repair[env,hand]=True
                detail['baseline_checks']=checks
                hands.append(detail)
            results.append(hands)
        costs,joints,points=self._evaluate(trajectory,kwargs)
        baseline=soft['optimized_cost_per_hand'].clone()
        # Preserve every original diagnostic tensor for unrepaired hands too.
        costs=torch.where(repair,costs,baseline)
        joints=torch.where(repair[:,None,:,None],joints,soft['predicted_joints'])
        points=torch.where(repair[:,None,:,None,None],points,soft['predicted_points'])
        diag=dict(soft,trajectory=trajectory,predicted_joints=joints,predicted_points=points,
            optimized_cost=costs.sum(-1),optimized_cost_per_hand=costs,
            baseline_soft_cost=baseline.sum(-1),baseline_soft_cost_per_hand=baseline,
            baseline_soft_feasible=baseline_feasible,baseline_soft_feasibility_checked=baseline_checked,
            feasibility_repair=repair,repair_cost_delta_per_hand=costs-baseline,
            repair_cost_delta=(costs-baseline).sum(-1),geometry_constraint_results=results,
            geometry_constraint_solver='scipy_SLSQP_per_environment_hand',
            feasibility_repair_may_increase_soft_cost=True,force_prediction=False)
        return trajectory[:,0].clone(),diag
