# Pinch fixed-wrist index seating native result

The GPU 2 experiment reproduced the sealed right contact at SDK 656, then issued a total additional right index target of 0.0012 rad. Neither wrist moved by command. The independent decoder checked all 24 seating decisions against the recorded solver state and verified the subsequent contact reactions. The run completed 848 SDK steps and 10152 solver substeps in 160.2 seconds, with no authored external force. Both process and GDB returned naturally with code 0.

| Minimum load across the 12 solver substeps (N) | First | Best across steps | Final |
| --- | ---: | ---: | ---: |
| Right thumb | 0.029 | 0.061 | 0.031 |
| Right index | 0.022 | 0.082 | 0.036 |
| Combined right hand | 0.051 | 0.144 | 0.067 |

The original threshold remains more than 0.30 N combined and more than 0.05 N per required pad, with opposing normals at every solver substep. Zero SDK steps passed that test. Both supports continued receiving load and the left hand did not touch. The index closure changed loads transiently, then they relaxed. This diagnostic rules out taking the peak index contact alone as stable closure. The next route should coordinate both Pinch pads at the measured right station and retain the staged left geometry plan. The result provides neither grasp qualification nor Teacher admission.

217 CPU regression tests passed before the native run. No PPO updates, new assets or physics changes were part of the experiment.
