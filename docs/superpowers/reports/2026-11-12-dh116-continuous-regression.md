# DH116 有界原生并行回归

首次组合 CPU 回归与 V0 原生并行时，4 个 CPU 子进程 fixture 误用生产调度租约，返回 `waiting_resource`。81 项通过，4 项失败，日志 SHA256 `6005622c4309bf06bf043a14a4bfc90e7a8b516e52bc80872191b286b8543632`。将 fixture 的调度租约和 Isaac 锁改为 pytest 私有临时路径。原生执行器与实际共享锁逻辑保持原合同。

在 Ubuntu 经 `bash scripts/python.sh -m pytest -q -p no:cacheprovider` 运行调度、F7 依赖、单指、单刚体生命周期、相机、WireSeg、指腹 SDF 和凸包相关 11 个测试文件，85 项通过，日志 SHA256 `fefaf58b408cd8b1eac0cf8fecb54d4e40ec02891316bf9b42e970f3da81585c`。

相机 V0 首轮在默认 SDK stage 路径取得 close marker，外层 SIGSEGV（Python 子进程退出码 −11）。随后只关闭 SDK 默认 stage，读回验证 `sdk_default_stage_present=false`、`effective_create_new_stage=false`，证据 SHA256 `866e2e8a464ab8802f16f58a0f6de7c77d2d34ccf8823655ae891470fceb0926`。第二次也在 close marker 后外层 −11。两次都观测到共享 Isaac 锁。此次范围可排除默认 stage 是触发崩溃的必要条件，原生栈仍待独立定位。V1 至 V4 按前置退出资格停留待验证。

当前可恢复入口为左拇指 v2 有界配方。该配方仅改变资源释放顺序，保留首轮接触账本和退出失败。待新的精确批次授权与资源核验后执行一次 240 步。右侧随后需要左侧外层退出 0 和同一账本身份。
