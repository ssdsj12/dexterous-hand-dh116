# Pinch Teacher 启动握手修复

基线 `f497dee1a4bcd6c431d9ae2bf70b4860818f6f68`，分支 `11-17`。

旧双目预览已经进入 App.close。收到已批准的一次 SIGTERM 后仍持有单实例锁。再次核验 PID、用户、起动时刻、命令和持锁文件后，按独立批准仅向旧预览发送一次 SIGKILL。锁已释放，旧预览记录为退出失败。

首个 Pinch 调度回执 `8069fad23d4be9bd5981560dd9c948f48030ed855e64c84499bc5573f7667fc9`。实际执行标志为 false，Isaac 未启动，物理、求解器和 PPO 均为零。标准 CLI 将原生 worker 限定在 PR 的开发分支。临时调用 Repository 接口虽然到达预约阶段，却不满足 ActualLaunchServer 要求的真实 worker 进程 argv，握手未写 binding，也没有外层退出记录。空原生日志仍保留。该指纹已登记失败。

`Repository.verify_native_target()` 允许同一项目 origin 下的开发分支和 `11-17` 运行标准 worker。发布、远端和 Draft PR 检查沿用原开发分支。握手仍核对真实 CLI、PID、源闭包、预约和锁。启动异常另附类型与原始错误文本在私有回执。旧原生配方和回执保留。

重新冻结 v2 的 core 与 companion，范围仍为 GPU3、单实例、1 环境、32 控制步、258 SDK 步、最多 3072 求解器子步、1 PPO 更新、180/240 秒。新尝试指纹 `6de6894074051f972aeeea3ac234f780761a4c6290aa92530d2a06335fdce347`。使用原 training JSON、模型来源、场景和物理预算。等待该指纹的原生批准。

152 项 CPU 回归通过，覆盖分支与 origin 验证、发布隔离、真实 worker 启动合同和证据校验。命令和日志 SHA 见 `docs/contracts/dh116_teacher_pinch_native_worker_branch_v1.json`。本轮没有获得原生 PPO 或抓取资格。

核验了 528 个保护文件及三个保留的可用模型。Registry 指向的两个历史 128 步诊断 Checkpoint 位于旧 /tmp 路径，当前文件缺失，无法重读 SHA。两条历史 SHA 和路径仍保留。
