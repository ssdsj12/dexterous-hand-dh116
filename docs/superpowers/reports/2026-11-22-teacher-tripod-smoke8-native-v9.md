# Tripod v9唯一原生诊断：失败保留

实测fcb22947；8控制、66SDK、768内部子步，1次真实PPO更新。更新后父墙钟校验拒绝，外层自然exit1，无signal。
stat cache清理后0项、tracked weakrefs全失活；未获native release、自然exit0或抓取资格。失败Checkpoint不恢复、不入成功池。
[完整证据报告](2026-11-22-tripod-native-v9-and-wall-rejection.md)。原指纹已消耗，不重跑、不扩预算。
