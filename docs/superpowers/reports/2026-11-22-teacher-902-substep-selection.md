# 902 flexible substep selection — 2026-11-22

The measured SDK475 failure showed wrist following error while fingers followed their targets.
An opt-in selection hook in FlexiblePhysicalExchange.control() can now retain targets for a real
physics substep. It may issue only the original 24D proposal or exact zeros. The original default
path, 480Hz/60Hz rates, geometry/joint guards and next-step rod reaction writes remain unchanged.
Waiting does not freeze either physical body or the flexible rod.

Every selected step records its pre-step physical identity/clock, proposal, actual issued action and
decision. Callback clock mutation, malformed actions, nonboolean pause flags, empty reasons, and
nonfinite/unserializable decision data reject before native writes. Exceptions close the owner.
This hook is not an integrated measured tracking controller: source-window accounting and a
same-clock measured wrist/velocity gate are still required before enabling it in a native recipe.

## Validation and retained failures

- New tests first reproduced 3 missing-interface failures; evidence tests then reproduced 2 pre-write
  validation omissions. Final focused run: 25 passed. Active related regression: **93 passed, 0 failed**.
- Expanded legacy suite: **95 passed, 12 failed, 1 setup error**. Twelve failures arise from old V2
  guard-source pins; setup also rejects the obsolete retained Pinch builder SHA. Git bytes at baseline
  53ca7fff already disagree with these pins (three guard files and the retained builder). Baseline SHA
  comparison was performed; a pristine baseline test run was not. No pin, archived receipt or validation
  threshold was silently replaced. This legacy path remains blocked.
- Antigravity R1–R4 read the same Ubuntu tree and final exchange SHA. Codex corrected its pre-step
  clock off-by-one suggestion and its claim that fixed wrist targets imply zero physical error.
- No Isaac run, solver step, Teacher query or PPO update. No new grasp/Teacher qualification.
- 528 protected files, three historical best.pt checkpoints and fixed stereo configurations retained.
- Source/config/tests/reports only; raw logs remain in the private evidence directory.

CPU command and log SHA are in substep-selector-active-regression.json and readiness; the private
evidence directory is /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/ref902.

Next: qualify the independently screened open-hand Reset at the original final wrist, then measure
fixed-wrist closure and actual opposed forces. If transit remains necessary, add bounded source-action
accounting before using this hook. Old V2 source requalification is a separate blocked legacy task.
