# Teacher WebRTC 显式关闭链与首次 GDB 崩溃栈

本轮按已批准方案，在现有 WebRTC 预览追加阶段账本。退出时先冻结物理入口，再同步实际 CUDA 设备，释放只读显示副本和运行时引用。随后通过已绑定安装源码的现有清理函数释放拥有的 context，发出 STOP 并核验物理视图，再调用 `app.close(wait_for_replicator=False)`。阶段账本在每次 API 前后写入并 fsync。未使用硬退出或替换 PhysX 属性。

## 结果

- CPU 定向回归 54 项通过。新增测试验证阶段顺序、同步错误和最终 App 关闭仍有各自的回执。
- GPU 2 单实例诊断执行一次，三个姿态各 60 个控制步，SDK 到 step 482，姿态拒绝 0，PPO 0。冻结预算、命令和日志 SHA 见 [readiness](../../contracts/teacher_webrtc_shutdown_order_v1.json)。
- GPU 2 同步返回。已绑定的 callback 解除和 context 清理返回。STOP 返回时 `SimulationManager` 的物理视图为 None。此后的 `app_close` 只有 started 回执。
- GDB 首次捕获 `present thread` 的 SIGSEGV。栈顶三帧均来自 `libomni.kit.livestream.plugin.so`，下接 `libcarb.events.plugin.so` 和 `libomni.kit.renderer.plugin.so`。已保存 C++ 栈、共享库和内存映射。定位到的是这次 WebRTC 关闭中的原生故障位置，尚未证明具体对象生命周期或线程竞争。
- 传递第一次故障信号后，GDB 读取寄存器时报告进程已消失，未取得正式 inferior 最终退出状态。GDB 自身返回 1。结果仍为退出失败。完整原始日志留在服务器 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-webrtc-shutdown-order-20261118`，没有自动重跑。

这批结果不授予 Teacher 抓取及训练数据资格。下一次关闭对照应以 livestream 生命周期为单变量，保留自然退出确证，并与独立的无流媒体相机关闭问题分开诊断。
