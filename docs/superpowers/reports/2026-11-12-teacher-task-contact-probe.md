# 现有刚性 Teacher 任务逐接触读回探针

`collect_dh116_teacher_task_contact_probe.py` 复用现有 `solid_cylinder` 环境和冻结 Teacher mean，2 环境最多 32 策略步，0 次 PPO 更新。通过现有 `EvaluationTelemetry` 旁路读取每个 480 Hz 物理步的 PhysX poll。保留 body A/B、各侧 collider、原生缓冲区范围、normal/contact 与 friction anchor 两套向量、world/SDK tick、env、episode UID 和 policy tick。每个物体独立匹配 `RigidObject.root_physx_view.prim_paths`，十个指腹通过实际 contact sensor body view 路径对应到各环境。未知对象接触、跨环境身份和缺失来源保持未解析，物体侧力为 NA。台面、外力、附着与各关节 relationship 的场景清单用于追查来源，当前完整性为 false。

## 原生初次探针

Ubuntu GPU 3 单实例 05:00:13–05:01:22 UTC，外层自然退出码 0。冻结快照使用预检的原手 USD SHA，独立副本仅重定位资产来源。采用已知 update-1 Teacher SHA `c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999`。探针当时在 scene reset 之后切换读回。两物体和 20 个指腹 actor 都具有报告 API，记录了 256 个连续全局物理步，SDK 19–274；同批有 506 条原生有效环境级行，其中 394 条含非零有效指腹载荷，结构化 poll header 共 0。首回合终止 1，原生成功 0。启用 readback 之后读回为 false，现有证据无法仅凭设置值确认首次建场采集缓冲已经启用。

| 原始证据 | SHA256 |
|---|---|
| 逐步接触探针 | `0e6d8c485a8e43fd00e7b57e236597ed59bb06aaca29cb72f7fb87d8015b4788` |
| 接触场景清单 | `2b506cc40c4333ba5df5d7519a3f30dd6181b9adea5285543cfdbbf01b4856ea` |
| 原生 Teacher 物理 JSONL | `8e83c57d30514bcd06384648e103558b27c84160a8d8ab22b7fdf0dbfdc02b1a` |
| 保护入口 scene summary | `586c9aa49a9e7144266e22028046d5ae5effa660fb493cd1e02eac0f9da72346` |

保护入口在关闭前写完 scene summary 和探针 fsync，实测没有返回到外层 Python 继续写原先的 completion。新版在独立回调内写 fsync 完成回执，外层退出证据另行核对。新版把读回开关前移到 `GpuWorld` 构造前，保持原物理设置、控制合同和原入口源码。首次 reset 后验证配置，单独保留后续原生验证资格。初次运行不能证明逐指与外部支撑来源。

## 复核

CPU 回归 `python -m pytest -q -p no:cacheprovider tests/test_dh116_teacher_task_contact.py tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_physx_polled_contact_audit.py tests/test_dh116_stereo_shutdown_validation.py`，71 通过，0 失败。覆盖原数组归属、未知或跨环境 partner、读回设置时序与标定回归。新版首次 reset 前的配置时序尚待排队复验。R1、R2、Force Proxy 和成功专家准入保持原门槛。
