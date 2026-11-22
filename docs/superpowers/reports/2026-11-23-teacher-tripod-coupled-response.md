# Tripod bilateral force response around the original wrist correction

The CPU audit re-read 256 individually hashed native physics records from the nominal 10 mm run. The 128 baseline ticks end at SDK2056 and the later 128 begin at SDK2065, after the eight ticks used for the left-wrist command. The tool applies the existing Tripod contact qualifier to each record, preserves the side and pad of every rejection and gives quantiles of the per-tick minimum load across 12 solver substeps. It requires the same run, Reset and rod in both windows. No simulator ran for this analysis.

| Measured result | 128 ticks before | 128 ticks after |
| --- | ---: | ---: |
| Left index ≤0.05 N, of 1536 substeps | 989 | 989 |
| Left index minimum load, median across ticks | 0.000 N | 0.000 N |
| Left total minimum load, median across ticks | 0.147 N | 0.149 N |
| Right middle ≤0.05 N, of 1536 substeps | 0 | 51 |
| Right middle minimum load, fifth percentile across ticks | 0.0563 N | 0.0495 N |
| Both hands qualified across a complete SDK tick | 0/128 | 2/128 |

Even after the left wrist reached the commanded offset, the left index continued to lose load. The right middle lost some of its small force margin while the right hand remained fixed. This suggests measuring both sides during every correction, then holding the wrist only after the actual wire load has stabilized. One archive provides no generalization or causal strength estimate for changing either wrist or individual fingers.

Antigravity independently read the four current control and feature sources. The existing force callback receives the previously completed SDK sample at the 60 Hz boundary, while the associated pose is added only to the emitted row. A new controller must bind the actual SDK hand poses and live rod parameters at that boundary. The existing closed-loop controller is limited to Tripod finger closing with fixed wrists. A generalized branch needs separately sourced palm-frame geometry and real per-pad force on each hand. Only Power may use its existing palm-plus-five-finger opposition criterion.

The next contact experiment should use the original action and safety guards with small, screened corrections. Require consecutive bilateral reactions before transitioning from a fixed-wrist hold to Lift. Use train wire specifications for choosing gains and evaluate held-out specifications without retuning. Keep the 30 mm net Lift, 5 s Hold, and 0.5 mm fixed-anchor slip limits. Neither this replay nor the 9 mm Reset supplies a qualified contact reference or a training success.

The CPU regression passed 84 tests. The readiness contract records the exact audit command, source commit, receipt SHA and output.
