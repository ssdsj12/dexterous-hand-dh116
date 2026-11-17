# Right Pinch legal precontact opening and continuous approach

Base493fb8d6a2d1a89a69973744da185f66c5180063. This package is CPU qualification only:0newnative,0PPO.

## Actual changes
solve_dh116_right_pinch_owned_approach.py freezes the independently passing endpoint(c53e0085a3e20aa0742460f3e0d51fbd08a5b14e4ee80554dfd4cf8f62d79ed6). Absolute central finite differences at1e-4/5e-5rad avoid the zero-relative-step issue. Bounded minimum-norm opening enumerates27active-bound patterns, rather than accepting an arbitrary underdetermined TRF solution.
Opt-in --raised-start keeps a1mrad extra planning margin above original coupled jointminimum, raises only initialroot0.5mm, and linearly returns to unchangedfrozenendpoint. Originallimits/material/controller remainunchanged.
All17source+5cooked initialshapes require50um rodseparation and2.1mm tableclearance.161pathsamples include angularlever+linearroot remainder; requireddepth uses a bound on actualtrilinearSDF gradients, not assumedunitgradient.

## Failure retained
Original fixedroot attempt42endpointcalls,0/3initialpasses,0pathsamples.
Result /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-owned-approach-20261113-4GSJT6Gf/run01/result.json
SHA6338a32ae3db02f3b69e4284c056dc6f89d575e4cf5a0640cb8ce047a3e5056e.
Discrete nodeSDF gaps werepositive but sourcefinger23 continuousrodsegment separation lowerbounds were-0.394506/-0.294364/-0.145799mm. Negative lowerbounds are not a measured penetration claim. Largestopening also loweredfinger22tableclearance below2.1mm.
Exactoriginalsource archivedbesidefailure as source_exact_v1.py, SHA d378799d14bc63bdc906706120d25824ba55f42eee8288e7602874d22c340581 verifiedagainstsavedresult.

## Raised-start result
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-owned-approach-v2-20261113-sQz6EbUf/run01/result.json
SHA28f4fdfc138daf800d372d7f3882c60e3953b16779c4fcb3032801cfff5ed8da.
ScriptSHA d354703477a9f0e7050b964c90391b0013150a553484097dc1845c265cc2d9ca.
41/48endpointcalls,161/161pathsamples,2.444867s/120s.0.6mm/0.8mmopening targets pass;1mmtarget rejected byoriginaljoint/displacementbounds. Minimalpassingstart selected.

Startq=[0.5679912210577621,0.008912622218025697,0.7181989328650555,1.15,1.15,1.15].
Endminusstartq=[-0.07157957980431606,0.0013970928910172556,0.0069208316591349295,0,0,0].
Required160step/480Hz velocities=[-0.2147387394,0.0041912787,0.0207624950,0,0,0]rad/s, allbelowunchanged0.3rad/s.
Roottranslation0.5mmdown over160steps, proposedvelocity-1.5mm/s. ActualSDKactionnormalization andsafetyprojection must be verifiedseparately.

Initialallshapetablemin2.581584mm,initialallshaperodgaplower0.107811mm.
Wholepathminimumtablelower2.195228mm, non-targetrodlower0.049080mm, deepestrequiredSDFbound-0.189913mm. Finalthumb/indexgaps remain-0.149766/-0.150060mm, opposednormaldot-0.882762.

|Rightlink|Initialsource tablelower mm|Finalsource tablelower mm|Initial/finalcooked lower mm|Tablepenetration|
|---|---:|---:|---|---:|
|finger14|5.724835|3.202191|5.724835 /3.248405|0|
|finger23|3.032694|2.664215|3.283401 /2.919383|0|
|finger43|47.993962|47.493962|48.028741 /47.528741|0|
|finger53|61.823157|61.323157|62.076074 /61.576075|0|

Valuesinclude conservativecontinuous-pathremainder,not just samplepositions. Fullpathstaticgates pass; native_ready=false,teacher_admission=false. ExactResetrod is fixedonlyforofflineanalysis; physicalrodremainsfreeinanyfutureexperiment.

## Tests, limits, next work
InitialREDmissingmodule plusminimum-normmissingfunction preserved.107CPUregressions pass,0fail,includingabsoluteFD,boundneighborhood,nonlocalderivative,actualSDFLipschitz,minimum-normboundpatterns,andearlierallshape/geometryregressions.
Command: bash scripts/python.sh -m pytest -q tests/test_dh116_owned_approach.py plusnineexistingowned-station/joint-root/surface/closure/rod-axis/segmentregressionmodules.
Nativeclaim:none. RuntimeSDKrootandjointmapping, actualResettracking, self/interhandcollision policy, actualcontactpair/wrench andnaturalexit remain gates. No newrecipe emittedhere. Pinch/Tripod/PowerTeacher0/3, R0stability/slipunqualified.
528protectedfiles+2checkpoints checked onpublication; no historicaldata/modeloverwrite. Existing234-step exit0 andTensorViewownershipevidence unchanged; no C5rerun. WireSeghistorical128seg/128depthsteps unchanged.
Next: compiletheverifiedstaticpath into an independentlyfrozenrightnative recipe onlyafteractionmapping,pairedinitialgeometry andresource checks. Newrawledger must proveactualReset,contactandexit; CPUcannot substitute.
