# Right Pinch inactive-direction proposal: exact nonlinear rejection retained

Basea5b8ca52cec1f6da1f1d5bdd19e5860d5b6590da. Independent CPU geometry work only;0newnative,0PPO,no newnative recipe. Existing controller,limits,source/cooked hulls,SDF,rod nodes and historicaldata remain unchanged.

## Root cause isolated in saved Jacobians

The previous relative-Reset batch stopped with relative Jacobian discrepancy0.325979. Its opposition row alone had two-scale derivative difference norm12.594715, despite actual residual+0.510373. Other largest row difference0.055222. This identifies a local SDF-direction derivative discontinuity; it does not prove contact feasibility or global infeasibility.

## Actual code

- solve_dh116_right_pinch_reset_direction.local_pass adds opt-in proposal_only_rows. Only distinct existing integer rows with baseline margin>=0.25 are allowed. Their derivatives are zeroed solely inside the linear candidate proposal and its derivative consistency check.
- Raw two-scale Jacobians and discrepancy remain recorded. Every trial still evaluates ALL original nonlinear constraints; best_violation and any final gate use the complete original vector. No safety row is removed from acceptance. Default behavior is unchanged.
- solve_dh116_right_pinch_relative_orientation adds --inactive-opposition-proposal. Only the uniquely named final opposition row is selected. It pins the exact last failed9D source result SHA532c6461b1ed30dd59b316b5e2718ef57a1ad9ba7ed3f47d17f50620fed86f86 and keeps the same9Dphysical domain,trust0.1,6passes/246endpoint ceiling,161path-only-if-valid,120sCPU.
- New tests prove that an adverse nonlinear value in the proposal-only row cannot become accepted. Active/small-margin rows,duplicates,badindices are rejected. The original default still stops on its original nonsmooth threshold.

## Actual bounded result

Output/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-inactive-proposal-20261113-4gNAksDp/run01.
Result SHAccd88f8ff95aab82f638d85f7ab022697979360fd377030facc568927c316ad1.
41/246endpoint measurements,0path samples,6.113557s,diagnostic exit0.

Proposal derivative discrepancy falls to0.002804; the raw discrepancy remains0.325979. LP predicts violation0.101129, but all four complete nonlinear trials are worse than the initial0.140321:

| Step fraction | Actual maximum violation | Limiting actual constraint |
|---:|---:|---|
|1|0.202811|thumb contact margin|
|0.5|0.163886|thumb contact margin|
|0.25|0.148283|thumb contact margin|
|0.125|0.141457|thumb contact margin|

The solver correctly keeps the original candidate and stops without another batch. The smallest tested proposal improves finger13 separation bound but loses thumb contact margin. The obstacle is therefore not only the derivative guard: this local direction trades incompatible actual geometry constraints. No global infeasibility claim.

Selected candidate remains exactly the previousfailed pose: thumb gap+0.040321mm,indexgap-0.075421mm,sourcefinger13 lowerbound-0.035878mm. All four initialpad/table clearances and their zero initialtablepenetration remain as in the previous report; this does not establish fullpath safety.

## Verification / publication

RED10failed,1originaldefaultpassed; afterimplementation122passed,0failed,7.98s. The11new tests plus111oldregressions use:
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_nonsmooth_reset_proposal.py tests/test_dh116_right_pinch_relative_orientation.py tests/test_dh116_right_pinch_reset_direction.py tests/test_dh116_right_pinch_measured_direction.py tests/test_dh116_polyline_closest_support.py tests/test_dh116_polyline_support_certificate.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_pinch_rod_axis_clearance.py tests/test_dh116_pinch_segment_geometry.py
```
Log/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/nonsmooth-proposal-full-20261113.log SHA2316710addf438f2aef59a6b37a5ebe9431889ec3ac5a0c81121522208cf51df.
Experiment command: python -m scripts.solve_dh116_right_pinch_relative_orientation --inactive-opposition-proposal --output <new exclusive directory>.

CPU software verification passes; rightgeometry andTeacher remain blocked. No consumednative recipe rerun. Next geometry work must change the actual feasible approach/Reset construction based on sourcepadgeometry and independent rightFK, rather than repeatedly running this unchanged local linear model. Independent R0 stability/materialslip and Tripod/Power qualification continue separately. Publish viaWindows,verify528protectedfiles+2checkpoints,PR2Draft.
