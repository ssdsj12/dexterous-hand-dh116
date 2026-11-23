# Tripod 902 live continuation interlock

The opt-in continuation extends the existing supported approach by one right-hand 24D window. It SHA-binds the naturally exited 568-step native run, its final measured control row, the unchanged 41 prior windows and the separate CPU proof from that final state. The chosen 1/16-scale correction has 73.056 µm modeled rod margin and improves all six modeled gap measurements by at least 28.965 µm. They remain positive.

The same physical exchange handles all 42 windows. The SDK 568 observation verifies the rod and registered bodies, measured link geometry, joint positions and velocities, submitted targets, wrist projection, support pairs and issued reaction. Any mismatch blocks the new command before its SDK 569 substep. The original 240 and 552 interlocks and 528 distal probe continue to run. The continuation requires both supports to retain load at every solver substep and aborts on hand or table contact. The new action cannot initiate a Lift or acquire Teacher credit. A same-task replay hook now also permits the Pinch launcher to bind its own archived handoff.

Antigravity R10 read the original fixed runtime and pointed out that returning a nonempty stop at a successful boundary closes the existing control instance. The current implementation independently verifies the measured and historical identities and only stops on a real mismatch. A successful interlock preserves the previous measured reaction in the current exchange.

## Offline verification

New tests cover the measured-anchor rejection, loss of precontact scope and the receipt-bound command prefix. The initial red run failed on the missing continuation implementation. The verification suite passed 125 CPU tests. The actual Python/GDB worker preflight exited 0 with 2,261 frozen input files on the integration checkout.

The recipe uses 1 environment, 1 Reset, 2 initialization SDK steps, 238 parked SDK steps, at most 42 control windows and 576 total SDK steps, with 6,888 solver substeps. It has a 420-second soft wall limit and 0 camera captures, model queries or PPO updates. This change does not authorize a native launch.

| Evidence | SHA-256 |
| --- | --- |
| Prior live outer receipt | `3f404d60345a420e2d0e48141344f25b3eaa23c2531f871b4a8cc0c7faa385ca` |
| Prior live endpoint CPU proof | `208aca8047a5f2240a6abd4864ea2cade452b2982e4d93b0a95e1823807afc8f` |
| Integration freeze | `4260fbb45c6db456cb3ece45650c989e2ba8b2f90bedcfa7644ec0810eba1f81` |
| GDB CPU verification log | `a45a8585d8c74d9f369310cfb4645d4e752ace18b43542fe44876d55479615ee` |
| CPU suite | `96b197ac34d64dc910a3d840c8a8977890fd3c6f3d64ec22d5cbcbd3b1b9549f` |
