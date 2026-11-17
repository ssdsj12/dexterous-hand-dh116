# Right-owned stations and constrained joint/root endpoint

Base8fe74cf793fd77f50cc3415da7334695c318aad7. CPU0native/0PPO. Original controls, assets, physical coefficients, coupling and limits unchanged.

## Changes and RED/GREEN
- solve_dh116_right_pinch_owned_surfaces.py: right USD/SDF patch locations at actual SDK234 root/q select own material stations4/3, rather than imposing left reflected5/6. Three radial references are explicitly proposed geometry, not measured normal forces.
- solve_dh116_right_pinch_joint_root_constraints.py: nine independently bounded joint/root coordinates. Existing all17 source+5cooked geometry is included in proposal residual and independently gated after every candidate. Root properrotation, legal original coupledq and missing-SDF rejection enforced.
- Two new test modules cover ownstation order, ambiguousside/node/frame rejection, root transforms, retention of non-target collision and bothcontact residuals. Initial missing-module RED logs retained. 102 relatedCPU tests pass,0fail.
- Formal native qualification remains false, even if geometric endpoint passes.

## Preserved radial-reference failure
Result /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-owned-surfaces-20261113-ArN16oy2/run01/result.json
SHA b702f1a267b6ace0a5db2e5300351a2b19ce365b24726f35ac948f6bfd417f2a.
242FK,5.346677s.0/3endpoints pass. Horizontalreference rootstaysabove but finger22 penetrates2.740555mm. Downwardreferences stillfliprootbelowtable. No attempt executedphysically.

## New constrained endpoint
Result /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-joint-root-20261113-VJTnWwEu/run01/result.json
SHA c53e0085a3e20aa0742460f3e0d51fbd08a5b14e4ee80554dfd4cf8f62d79ed6.
315/1200evaluations,6.732235s/120s,2starts;1staticendpoint passes.

Selected case1:
q=[0.49641164125344606,0.010309715109042952,0.7251197645241905,1.15,1.15,1.15].
Root xyz=[0.5351562922835009,-0.23666665853788135,0.3948096523566933].
Root quaternionwxyz=[0.2324822392860856,0.6544949886855543,-0.4027341505331337,-0.5961489094147354].

|Right shape|Source tableclearance mm|Cooked tableclearance mm|Penetration mm|
|---|---:|---:|---:|
|finger14|3.225379|3.271593|0|
|finger23|2.668488|2.923656|0|
|finger43|47.495525|47.530303|0|
|finger53|61.324720|61.577637|0|

Across22 shapes minimumtableclearance2.199504mm atsourcefinger22; minimum certified non-target rodseparation0.099392mm atsourcefinger13. Requiredthumb/index lockedSDF gaps-0.149766/-0.150060mm, normaldot-0.882762. Original2.1mm/-0.3mm/-0.1mm/-0.2gate unchanged.
Originalsource-only17shape evidence doesnot magically become all-livecooked evidence.

## Limits and next gate
No fullapproachpath, actualReset tracking, selfcollision or actual contact is qualified yet. Finger12endpoint is only2.397092mrad above original coupled lowerlimit0.007912623rad. Opening by a generic20mrad would violate the unchanged limit; must solve a legal three-joint approach and certify allpath samples/bounds first.
Start0 remained effectively unchanged. Current least_squares relative finite-difference nearzero coordinates interacts withfloat32SDF; its local stall is not an infeasibility proof. The passing endpoint is independently recomputed and frozen; future direction/path derivatives must use explicit absolute steps and reject missing/ill-conditioned neighborhoods. This result is not an optimizer convergence or native tracking claim.

TeacherPinch/Tripod/Power remain0/3. R0 stable/materialslip stillunqualified. No native recipe emitted; no new native authorization budget consumed.

## Commands and protection
RunCPU:
bash scripts/python.sh -m scripts.solve_dh116_right_pinch_owned_surfaces --output NEW_DIR
bash scripts/python.sh -m scripts.solve_dh116_right_pinch_joint_root_constraints --output DIFFERENT_NEW_DIR
Tests: test_dh116_joint_root_proposal.py,test_dh116_owned_surface_targets.py,test_dh116_surface_registration.py plus measured-direction/polyline-support/closure/rod-axis/segmentgeometry regressions.
Logs resideunder /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/, notGit.
Publication checks528protectedfiles+2checkpoints. Existing234stepnaturalexit0 andTensorViewownership scope unchanged. WireSeg remains historical128seg/128depthsteps; no newtraining. Next: legalopeningdirection,continuousclosureclearance,selfcollision/pairedReset,thennewboundednativeonlyifallprechecks pass.
