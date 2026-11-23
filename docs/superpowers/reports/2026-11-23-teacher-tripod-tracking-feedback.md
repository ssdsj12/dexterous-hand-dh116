# Tripod feedback with a measured target-lead limit

The opt-in \`--bilateral-feedback configs/dh116_tasks/flexible_tripod_bilateral_tracking_load_feedback_v1.json\` diagnostic reads the completed physical exchange before the next 60 Hz decision. It uses that exchange’s 12 measured solver samples and its actual PhysX joint position and backend target readback. The post-SDK source must agree with the rod’s run, environment, episode, Reset, rod identity and clocks. The reader requires the eighth SDK substep of the control window and the correct named, radian articulation channels.

An unqualified hand can request positive closure only on an eligible force-deficit channel. The controller ranks those channels by measured normal-load deficit after applying the original remaining travel budget and the signed lead limit. The lead is backend target minus measured joint position. The requested increase can bring the decision-time lead up to one nominal window of travel, 0.15 mrad at the original configured step. The existing force-only variant, force and joint limits, 24D action path, wrist holds and collision guards run under their established rules.

## Playback of original native measurements

The CPU replay verified each input against its outer receipt. It reproduced all 64 logged commands and hand decision reports from the original bilateral feedback run. The opt-in result was zero correction requests at all selected boundaries.

| Archived physical cohort | Control boundaries | Original force-only nonzero proposals | Tracking-limited nonzero proposals | Measured left-index target lead |
| --- | ---: | ---: | ---: | ---: |
| Original bilateral feedback | 64 | 12 | 0 | 0.864–1.691 mrad |
| Late fixed-wrist interval before lateral pulse | 16 | 4 | 0 | 1.761–1.814 mrad |
| Late hold after lateral pulse | 16 | 4 | 0 | 1.744–1.893 mrad |

The force-only proposals in the last two rows are CPU queries of a historical trajectory. The zero proposals in 96 boundaries establish that these recorded states lack headroom for this positive-closure correction. They give no physical response estimate for an alternate command. The prior coordinated native run also showed a 0.589 mrad increase in the left-index backend target while measured q moved −0.026 mrad. A standalone rerun of the tracking gate on the same replay prefix has no evidence of restoring left-index contact.

The real-source CPU freeze bound the independent request and implementation to the supported Tripod prefix through SDK 1296 and the original 64 decision windows. The full frozen diagnostic budgets 1808 SDK ticks, 196 controls, and 21672 solver substeps. The package executed zero new Isaac steps and zero updates.

A next correction needs independent evidence that it can recover left index tracking and maintain opposing reaction on both hands for the fixed-wrist hold. Physical qualification requires a 30 mm unsupported lift, a continuous five-second hold and slip at most 0.5 mm. The [readiness record](../../contracts/dh116_teacher_tripod_tracking_feedback_v1.json) binds the CPU suite, playback and freeze artifacts.

The combined CPU suite passed 181 tests. It exercised the live diagnostic entry with a CPU physical driver and covered tracking headroom, exhausted travel, and rejection of stale, mismatched or unavailable readbacks.
