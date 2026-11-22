# Tripod index continuation after thumb load recovery

The measured combined-close trial lost 13 opportunities for index advance because the thumb had already recovered in a combined command window. On the sealed pre-command snapshots of all 13, positive thumb normal load covered all 12 solver substeps. The same-dose index-only candidates passed the original full-envelope check with a minimum conservative margin of 7.230 µm. This replay contains no counterfactual contact forces.

The separate `--resume-index-after-reseat` mode retains the original control window, causal force readback and nominal index increment. It allows the physical exchange to remove only the wrist increment from a predeclared left index-plus-wrist proposal. It records the authorized index-only action separately from the original proposal and issued readback. It rechecks the measured geometry before advancing and still holds after the first index reaction. All existing collision, joint, timing and issued-step bounds apply.

CPU regression passed 163 tests, including rejection of unapproved substitution, index dose changes, unrelated hand motion and conflicting hold markers. The frozen 9 mm native trial in the preceding report remains the measured baseline, with zero index load and no Lift. This continuation has no native result and grants no Teacher or grasp qualification.
