# Tripod force-guided seating native result

One GPU 2 Isaac instance completed 1520 SDK steps, 160 controls and 18216 rod solver substeps in 404.4 seconds with zero PPO updates. The exact SDK 1296 force boundary matched live replay. The independent reducer verified 768 consecutive reaction transfers across the bilateral phase and all 28 recorded decisions against the completed solver samples.

The right index accumulated 0.0012 rad of issued target travel and the middle 0.0018 rad. Both wrist action groups and the left finger actions stayed at zero through the seating and settling windows. Over the 224 SDK steps after the boundary, left contact qualified for 31 steps and right contact for 167, with 24 steps qualifying on both hands. The longest uninterrupted right sequence was 60 steps. The longest bilateral sequence was one SDK step, about 2.08 ms. Qualification here means the original load and opposition test observed in each of the 12 corresponding solver substeps.

The supported rod remained on both supports, and no table or unassigned hand contact appeared. No geometry guard tripped. The process and GDB exited naturally with code 0. The earlier same-cohort run produced zero simultaneous qualified rows. The new contact remains too intermittent for a stable grasp or a lift. Next, inspect the left index SDF and measured force on the same rod state and validate a bounded bilateral seating command under the same wrist and success constraints.

The recipe passed 216 CPU tests before the native run. Teacher admission and PPO training remain closed.
