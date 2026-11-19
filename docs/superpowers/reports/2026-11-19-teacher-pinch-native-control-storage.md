# Pinch 原生 worker 私有会话目录修复

基线 `3314206116c4ccc1810e79d14774692e6db7cc99`。已批准的 v2 实际启动了 GDB 下的 Python worker，并观察到它持有单实例锁。父进程完成绑定，但在启动授予前拒绝实际会话目录。外层与 GDB inferior 均退出 2，耗时 45.81 秒，无 SIGSEGV。不会重跑该指纹。Isaac、物理和 PPO 均未开始。

父端回执明确记录 `actual launch grant rejected: private current attempt session directory required`。子端的 `missing or oversized launch packet` 来自握手被关闭。实际控制目录位于大盘 Fuse 挂载，尽管调用 mkdir 指定 0700，内核读回 UID 0、0777。现有所有者与权限校验 在发放授权前按合同拒绝。回执 SHA `dd37906d2a7add2f30e27b74c215ffc8d9fa3c080e9ceb325eab66b829eef1b3`，详细日志及绑定 SHA 见 readiness。

调度器新增 `native_control_directory_blocker`，检查目录和其新建子目录的实际所有者与权限。资源预检查和实际注册均调用该验证。Ubuntu 上已实测 Linux 用户运行目录支持当前 UID 和 0700，大盘控制目录会在注册前被拒绝。后续 scheduler intent、Unix socket 授权和外层回执使用私有运行目录，日志与结果密封后保存副本到大盘；体积较大的物理输出仍在独立大盘路径。保持原会话权限和启动身份校验。

153 项 CPU 回归通过。v3 冻结指纹 `6d5cb9cc4248dedc105035c63bf590a1ca659ca5e37ef0c031a554188186a0cc`。预算保持 GPU3、1 环境、32 控制步、258 SDK 步、最多 3072 求解器子步、1 PPO 更新，180 秒内层、240 秒外层。无重试、无强制结束。尚待对该新指纹的授权，未获得 Teacher PPO 或抓取资格。
