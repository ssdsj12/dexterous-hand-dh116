# Real128-lane no-camera Teacher PPO: two completed updates

GPU3,128 deformable rods,2 regex Articulations,canonical24D,32windows/update,
2 accepted real optimizer updates and8192 on-policy samples. GlobalSDK514
(2 initialization+512 measured),12 solver substeps/SDK/lane=786432 lane-substeps.
Timeout Reset advances no physics; final reset counters are3 for all128 lanes.

This is the unchanged finger-goal/R0 wiring diagnostic. It is not a successful
grasp expert, full Lift/Hold/slip evaluation or formal three-grasp qualification.
No camera,Student,BC/DAgger,R1/R2 orForceProxy admission is granted.

| Update | End SDK | Samples | Rollout s | Optimizer s | Total incl. checkpoint s | Post-step exact KL | Accepted LR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 258 | 4096 | 9.847213 | 0.124367 | 10.142763 | 0.0013332939706742764 | 1e-05 |
| 2 | 514 | 4096 | 9.583548 | 0.055583 | 9.789208 | 0.001229075132869184 | 1e-05 |

Configured LR remains1e-4,gamma.995,targetKL.02,3 rollback steps.
At1e-4 actual full4096-sample KL was.1333377063/.1229954809.
The original KL Guard restored policy/Adam/RNG and accepted1e-5;
post-step KL is shown above. The threshold was not increased. Both updates
accepted1 full4096-row minibatch; top-level frozen/pre-step KL0 is not passed
off as post-update KL. Tanh differential entropy-21.633734/-21.636242 and24D
std groups are recorded in raw JSON. Maximum old/recomputed logprob error
7.6293945e-6 is below unchanged2e-5 absolute/relative tolerances.

Warm second iteration mean window.299187s,min.288731,max.371857.
Whole GPU sampled peak3488MiB;Torch peak647681536 bytes.
Mean second-iteration phases,seconds:
{
  "observation_policy_sample": 0.01949161537049804,
  "reward_window_begin": 0.006732422931236215,
  "project_reaction_submit": 0.053412641362228896,
  "physx_step": 0.06350054568611085,
  "sdk_readback": 0.00898631817108253,
  "raw_source_guard_mesh_submit": 0.01861681955779204,
  "rod_flat_export_reduce": 0.025261949078412727,
  "geometry_r0_and_post_guards": 0.08440097229322419,
  "value_and_reward_settlement": 0.01609561904479051,
  "window_reset": 0.002028820978011936
}

## Limits and evidence

6–8s/update is NOT met. First full iteration10.142763s exceeds10s; second
9.789208s is under10s. No checkpoint time is excluded to manufacture a pass.
Cold total408.096023s includes USD authoring/startup/close,not per-update cost.
The independent N128 probe located371.14s before hand-ownership registration;
read-only source review suggests profiling repeated surface extraction/material/
availability checks and O(N²) collision-filter targets next. No current source
or physical isolation was modified to hide that startup cost.

App.close returned after the original stat-cache/clear/drop-alias/stop_commit/
manager order. LocalTimeline weakref is dead before close;source-freeze has
admissible=true anddrift=[]. Outer natural exit0,signal=null,forced_exit=false.
No os._exit,SIGTERM/SIGKILL,physics/1µm/joint/30N protection change was used.

Source publication eb3914bb58546c5e7ac9e503b1d8280f25227510 was imported and
pushed by Windows,with52-file list and treeadf5b98a33f62293b8b45fed18460ebf687af30f
verified. The run froze those actual bytes before publication; no old committed
tree is invented for earlier uncommitted source.

Receipt outside Git: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/tensor-common-ppo2-128-gpu3-xc6n1p7x/outer.json
SHA256 18007ea09f339e879d72a94e4e1bcf57cc302639dd41adc1c59c59f9d11cd395
Command and source inventory: siblingattempt.json,SHA256 0bc2f93bc039c9dae188046700da531b515b3058b2b338f6b706d7de64112276

## Checkpoint/recovery

Two exclusive v2 checkpoints remain outside Git,not in a successful expert pool:
- /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/tensor-common-ppo2-128-gpu3-xc6n1p7x/native/tensor_pinch_update_0001.pt SHA256 3606bfab81930ae877dbc9cbc17f8f2b81e936e1e905a347b138236401f0cbf6
- /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/tensor-common-ppo2-128-gpu3-xc6n1p7x/native/tensor_pinch_update_0002.pt SHA256 6d9f61eaab9d4f1baeb77524a62859931d77d6036bd1c07525f220bc019f6757

Policy/Adam/counters/Torch CPU+CUDA/Python/NumPy RNG are saved.
The separate actual GPU2 continuation test matches subsequent actions/rollout/
metrics/policy/Adam/RNG exactly. Physical state is not saved: restart requires
fresh verified Reset and cleared history. The bounded entry saves v2, but an
end-to-end native resume was not exercised and is not claimed.

CPU603pass/15CUDA-skips(96.42s),Registry/source/receipt120pass(17.94s),
actual CUDA continuation1pass(6.40s). N2 real-contact/raw-cache/selected-Reset
and N128 probe/update workers all naturally exit0.528 protection hashes remain
unchanged;2 historical/tmp checkpoints were already absent. No assets,models,
.so binaries,raw logs or private data uploaded;PR#2 staysDraft/open/unmerged.

Next: preserve per-SDK guards while reducing remaining numerical validation,
reward and source-clearance costs; profile cold authoring separately. Tripod's
extra guard is CPU-tested but not connected/qualified in this tensor entry.
Power has not run this new CABI proof. Three-grasp training remains pending.
Student reuses the common physical/reset/lifecycle interface, but native
TiledStereoCapture and Student training are not claimed complete.
