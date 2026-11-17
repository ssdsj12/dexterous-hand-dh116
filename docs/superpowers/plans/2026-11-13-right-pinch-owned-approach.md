# Right endpoint to legal pre-contact approach
Base493fb8d6a2d1a89a69973744da185f66c5180063. Freeze passing static endpoint resultSHA c53e0085a3e20aa0742460f3e0d51fbd08a5b14e4ee80554dfd4cf8f62d79ed6 case1.
Rootfixed. Solve three-active-joint opening with originalcoupledlimits and max0.1rad total displacement perjoint (=160steps at480Hz/0.3rad/s). Use absolute centralFD1e-4 and5e-5rad, compare derivatives; zero-coordinate float32 relativeFD not permitted.
Three targetprecontactgaps0.15/0.25/0.4mm. Atmost48endpointFK calls,161path samples,120sCPU,0Isaac/0PPO. Every initialshape mustclearrod50um andtable2.1mm. Pathcertifies17source+5cookedtable/non-targetrod with angulartravel remainder,requireddepth with actualSDF Lipschitz bound. Preserve allfailedcandidates. PassingCPU pathdoesnotqualify selfcollision, actualReset/contact orPPO.

Distinctv2afterfixedrootfailure:27active-pattern minimum-norm solver;initialroot+0.5mm,originalcoupledlimit+1mradplanningmargin;targets0.6/0.8/1mm;48endpoint/161path/120sCPUbudgetunchanged. Wholepathincludeslinearrootremainder and actualSDFLipschitz. Originalv1source/evidence preserved,nonewIsaac.
