# 柔性双手控制步的物理交换

新增 `FlexiblePhysicalExchange`，复用原双手 24D SDK 连接、同杆运行时、桌面腕部投影及原物理步时钟检查。只接收调用方已经构造并重置的单环境。校验实际 PhysX articulation 和每个指腹 link 对应同一份杆槽位绑定。每个 60 Hz action 最多推进八个 480 Hz 子步；每个子步依次核验物理时钟、计算安全目标、提交上一有效杆步的反作用力及双手 SDK 目标、推进一次 PhysX、读回两手、推进同一根杆、核对实际目标、交付一行观测。首步反作用力明确为 reset 零值；桌面反作用力不提交到手。

复用原关节限位与 mimic 保护，保留既有 30 N 单指腹反作用力上限。已知终止只交付一次，下一步不提交终止帧的新反作用力。记录各子步请求、投影、SDK 目标和执行前后状态。外部乱走物理时钟、反作用力时钟错位和不完整控制步预算均拒绝；已推进后出错关闭当前实例，禁止重放。持久化和退出归调用者；这个类不构造 App、不会自动 reset、不会执行 PPO。

Ubuntu CPU API double 回归 **117 passed**。新增六项对照覆盖 8 子步、反力时差、终止不追加 no-op、预算、异常、原关节及载荷保护。其中保护测试在补保护前确实失败。原生资格保持 false，训练奖励保持 NA。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_flexible_physical_exchange.py tests/test_dual_hand_rod_runtime.py \
  tests/test_dh116_flexible_wrist_projection.py tests/test_dh116_deformable_bimanual_step.py \
  tests/test_dh116_deformable_control_step.py tests/test_dh116_deformable_grasp_task.py
~~~

| 私有测试证据 | SHA256 |
| --- | --- |
| 初次缺模块失败 | `8fb046784d530e3f57f0aaccd08c836d4777c5865ed26d111b5340ebcf769d26` |
| 保护回归失败 | `cc982771346802a3c3b86ca5a6f4681b87e83204c5dc872881509c5694c5356a` |
| 最终 117 回归 | `87cee20b08f6edd7ffc5adcc07e29dbb580d2bd94e3ec5cc3e2c808821b1a481` |

使用既有自由腕策略的零公共 lift offset，不引入额外高度命令。实测全手净空和 active limits 还需要由真实场景逐项核对；初始候选、双手力学和材料滑移仍未通过原生资格。当前只能组成单环境物理交换，2 环境 rollout、完整 371D、正式 R0 和独立原生关闭仍各有阻断。
