# PhysX 标定资源生命周期

独立 v2 左拇指标定完成 240 个 480 Hz 物理步。实际 Kit 读回 `/app/fastShutdown=false`，`app.close()` 返回，回调和所属 `SimulationContext` 已清理。子进程约 0.487 秒后收到 SIGSEGV，外层验收未通过，第二轮未启动。现有试验记录没有可用 core 或崩溃堆栈，具体故障对象尚未定位。

原标定函数在 Kit 关闭及插件卸载时仍持有本轮 Tensor、PhysX 报告与 USD 包装对象。新版按所有权释放本轮订阅和 Tensor 子视图，在 Kit 可用时用已安装 SDK 的公开 `SimulationView.invalidate()` 使仍有效的所属视图失效，然后释放视图、报告接口、设置和场景包装对象，再调用既有关闭流程。若失效调用抛异常，仍释放这些引用并尝试关闭，错误继续交给外层失败处理。新增 Python `faulthandler` 在 AppLauncher 启动后接入继承的 stderr，随进程生命周期保留诊断能力。实际关闭返回和自然退出码 0 仍是验收条件。

场景和采样段在归一化句柄归属后 AST 一致。未改变物理参数、240 步预算、原始采样字段、回调证据语义、原生验收或力学资格。此次提交没有启动 Isaac，也没有修改此前冻结源码或原生证据。

Ubuntu CPU 回归使用 `python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_shutdown_validation.py tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_physx_polled_contact_audit.py`。修复前新增回归 5 项失败，修复后 65 项通过，0 项失败。CPU 测试覆盖了有效、失效、缺失和异常视图四种释放场景，以及 AppLauncher 后开启诊断。修复后日志 SHA256 为 `f215a7978d732db687df2e3a9cd24507b1126b58dcc7ae832bcbc920bc10cfdf`。修复效果仍待单实例原生复验。
