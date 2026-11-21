# Tripod Reset 原生入口的进程隔离修复

基线29a63e4723a4939dae0730438257ea63394523fb，保留旧生命周期和启动边界。

## 实际失败与根因
新17槽入口的第一次外层启动指纹 ebe5ffeacde86cd2ca1eca1aa2269ef8c1e8d43c35ebe6a34d955b59e139444b，
12.860s自然退出1，无SIGSEGV、无强制结束。Isaac App尚未创建，实际SDK/控制/求解器/PPO均0。
真实栈表明 freeze_plan -> scene_preparation -> usd_surface 提前导入pxr，
已有assert_clean_import_boundary正确拒绝。此前CPU测试只验证freeze数据与构造，遗漏了同一进程从freeze进入App的边界。
首次非canonical任务路径在CPU冻结阶段也被正确拒绝；改用resolve后的真实路径，保留失败日志，不放松路径校验。

## 实际修复
record_dh116_tripod_supported_reset.main将完整来源重建核验放到独立同解释器CPU子进程。
子进程必须比较完整冻结配方；父进程随后逐项重读SHA并比较plan，再进入原R06 execute。
保留原pre-App检查，不卸载USD模块、不跳过SHA、不修改SDK依赖。
新增真实子进程测试：App父进程拒绝pxr/PhysxSchema/isaaclab/isaacsim/ropesim的提前导入，
仍可经实际CPU核验流程到达launch边界。红灯复现原native失败，绿灯7/7，42.66s。

本包不把前一次失败当成Reset资格；新指纹尚需真实17槽Reset与自然退出。
后续独立Tripod NativeSources/session/preflight和原PPO更新/恢复仍待完成。
固定双RGB、左右身份、物理、24D/371D、原关节保护及30mm/5s/0.5mm标准均不变，R2关闭。
原始日志和外层回执保存在Git外ref902/fresh-supported-tripod-reset-native-v1；SHA见readiness。

Registry：60通过、0失败，15.33s。
