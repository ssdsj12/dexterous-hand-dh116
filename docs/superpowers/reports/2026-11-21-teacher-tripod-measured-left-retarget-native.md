# Tripod retarget native contact result

A single environment on GPU 2 completed 1456 SDK steps, 152 control windows and 17448 DeformX solver substeps in 405.8 seconds. There were zero PPO updates. The SDK 1136 replay matched the sealed hand, rod, contact and reaction data. The replacement approach passed SDK 1178, where the old path stopped. There were no geometry guard stops, out-of-station pairs, or table contacts. Both GDB and the worker returned 0 after normal SDK shutdown.

| Left link | First actual solver contact SDK | First subsequent PhysX composer delivery SDK |
| --- | ---: | ---: |
| Middle distal | 1255 | 1256 |
| Thumb distal | 1264 | 1265 |
| Index distal | 1282 | 1283 |

The independently reduced 704 rows after the right contact boundary all recorded actual reaction delivery. Under the original required-pad force and opposition checks, the left hand passed all 12 substeps on 33 rows and the right on 46 rows. No row passed for both hands. All 46 loaded right rows preceded the retarget. Both end supports continued to carry the rod. No Lift or Hold qualification was performed.

At SDK 1296 the left thumb, index and middle minimum substep loads were 0.319, 0.235 and 0.133 N. Right minimum loads were 0.109, 0.056 and 0.033 N. Subsequent left motion changed the shared rod load distribution, and the right middle lost contact before the final window. The final left index also lost contact. The next diagnostic should hold both wrist targets at a measured full control boundary and check small finger-only changes against actual opposition and load. Use neither the maximum pair force nor the number of contacting links to label a grasp successful.

The frozen launcher was validated by 208 CPU tests before the native run. A first local launch command used an incorrect plan filename and stopped before GDB or Isaac. The corrected launch used a fresh output directory and completed the one native run reported here.
