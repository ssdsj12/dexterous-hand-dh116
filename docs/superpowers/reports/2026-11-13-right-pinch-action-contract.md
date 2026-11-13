# Right-owned Reset, original24D schedule and scoped projection validator

Baseceb14527e17bddecc5a47fb7e70b1d8066ab1cd6.0newIsaac,0PPO.

## Implementation
- prepare_dh116_right_pinch_owned_contact.py derives independentright Reset using originalprepare_reset_state and originalaffinecoupling. Leftinitialpose and freerodbytes unchanged.
- flexible_right_pinch_owned_approach_v1.json is new configuration, not an overwrite. Onlyright roots/q andexplicitproposalidentity changed. Sourcesretain ownright USD/SDF hashes.
- action_schedule compiles29×24float64 actions:6quietcontrols,20approachcontrols,3holdcontrols. Onlyindices6/7/8/17 change. Originalrates .3rad/s/.01m/s/.5rad/s preserved.
- audit_dh116_right_approach_action_contract.py reuses originalfinger_pre_guard,surface_final_guard,wrist_requested_targets,project_policy_wrist_targets andMountViewlocal_translation.232CPUidealtrackedsteps, notnativeSDKsteps.
- tabletop_right_pinch_approach_geometry.py adds opt-in RightApproachGeometry. Every480Hz identity, scheduled24D, requested/projected/submittedrootandangles, projectionflags/source mustmatchfrozenpath. Onlythen oldzero-wrist-offsetassertion is replaced. Allsource/cookedtable,non-targetrod,depth,joint/mimic,tracking/speedguards remain active. Oldgeometryclass/default unchanged. Independentrecomputation preservesrawpayloads.

## Actual CPU evidence
Result /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-action-contract-20261113-FfkDi9o6/run01/result.json
SHA9689d66dbcac7bf5e7c769e9027322501e685aadd98ccc9fe2c4c6488c5d0ef9.
Sourceexplicitly CPU_ideal_tracking_not_native.
232rows,4.107269s,maximumqerror5.773159728050814e-15rad,rooterror5.551115123125783e-17m,world/localroundtriperror0,projectionclippedrows0.
The result's pending-scoped-validator note predates the subsequentlyimplemented13CPUtests; rawresult wasnotrewritten.
ConfigSHA e88f2a24408f84fcd8b6e3e24bda6c5d6633e3a6dc9a705271dee73fa1c8db9d.
ResetseedSHA c9ef9bf12a65fb89d597ff6f4ad3e64dce09bdf3d74ff8b58717fa8fb95e3b69.
StaticpathSHA28f4fdfc138daf800d372d7f3882c60e3953b16779c4fcb3032801cfff5ed8da.

Initialtestfound implicitfloat32 tensor construction beforeassignmenttofloat64:integratedqerror4.27e-9rad. Corrected source to explicitaction.dtype; originaltolerance retained, no testrelaxation.
Frozenaction nonzeros duringapproach:
rightfinger11=-0.7157957980;
rightfinger12=0.01397092891;
rightfinger21=0.06920831659;
rightworldZ=-0.15.
Remaining20dimensions zero.

## Validation
Missingmodule RED anddtypefailure logs preserved.54relatedregressions passed(123.35s),13additionalprojection-scope casespassed;67uniqueCPUtests total. Casescover physical49/208/232boundaries,zero-windowretention,right-onlytargets,leftmotionrejection,rotationrejection,safe-zclip/source rejection,wrongaction/substep rejection, immutableoldguards andconfigresetidentity. Publicationrerunscombined67.
Old528protectedfiles and2checkpoints verifiedbyreleaseworkflow; nooldcontroller/model/dataoverwrite.

## Native status / next package
No recipe executed and noactualSDKtrackingorcontactqualification. Next: register separatelyfrozenright worker/scheduler/postexitvalidator, include newconfig+geometrysourcesinexactfingerprint, bindactualqualifiedparameterparentreceipt, confirmGPU3/oneIsaac/resourceavailability, then once-only234step/1env/0PPO≤180s diagnostic understandingapproval. PreserveGDB stack/maps andactualouterexit; no retry orbudgetextension.
Currentoldzero-wristvalidator mustnotbeusedfornewnonzerowristscope. Newgeometryaccepts onlyspecified-0.5mmrightZtrajectory; it doesnotdisableclearanceprotection.
TeacherPinch/Tripod/Power still0/3. Existingleftbriefcontact/234stepnaturalexit0 stayshistoricalnarrowevidence. R0stable/slip/fullReset gatesremain. WireSeg unchanged128seg/128depthsteps. No C5rerun.
