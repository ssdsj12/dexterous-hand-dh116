"""Compare independent physics trials; nonzero computed force alone is insufficient."""
import numpy as np


def compare_pair(baseline,reacting,baseline_rows,reacting_rows):
    for field in ('asset_sha256','run_config','initial_state'):
        if baseline[field]!=reacting[field]:raise ValueError(f'pair {field} mismatch')
    if baseline['contact']['reaction_applied'] is not False or reacting['contact']['reaction_applied'] is not True:
        raise ValueError('expected disabled baseline and enabled reaction')
    for summary,rows in ((baseline,baseline_rows),(reacting,reacting_rows)):
        if (summary['passed'] is not True or len(rows)!=summary['run_config']['steps'] or
            len(rows)!=summary['steps'] or summary['contact']['peak_force_n']<=1e-8):
            raise ValueError('incomplete or no-contact trial')
    qdiff=0.;vdiff=0.
    for step,(b,a) in enumerate(zip(baseline_rows,reacting_rows)):
        if b['step']!=step or a['step']!=step or not b['passed'] or not a['passed']:
            raise ValueError('invalid step sequence')
        for side in ('left','right'):
            x,y=b['sides'][side],a['sides'][side]
            if x['joint_names']!=y['joint_names']:raise ValueError('joint names mismatch')
            deltas=[]
            for field in ('actual_joint_positions','actual_joint_velocities'):
                p=np.asarray(x[field],dtype=float);r=np.asarray(y[field],dtype=float)
                if p.shape!=(len(x['joint_names']),) or r.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(r).all():
                    raise ValueError('invalid measured joint state')
                deltas.append(float(np.max(np.abs(p-r))))
            qdiff=max(qdiff,deltas[0]);vdiff=max(vdiff,deltas[1])
    return dict(passed=qdiff>1e-7 and vdiff>1e-6,max_joint_delta_rad=qdiff,
        max_velocity_delta_rad_s=vdiff,grasp_qualified=False,
        interpretation='bounded contact reaction evidence, not sustained grip or lift')
