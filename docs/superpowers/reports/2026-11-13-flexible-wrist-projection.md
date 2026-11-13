# 柔性 Teacher 腕部原桌面安全投影

双手 24D SDK 边界此前要求调用方给出原桌面投影后的腕部目标。新增纯计算入口 `project_policy_wrist_targets`，复用原 `palm_transport`、`per_hand_safe_z` 和 `compose_rotation`。沿用 xyz 策略默认 2.1 mm 净空、480 Hz 子步、0.01 m/s 向量范数约束、每手独立安全 Z 及内禀 XYZ 顺序。保留原逻辑中安全抬升可优先撤回已排队下压目标的行为。返回投影前目标、左右安全高度、裁剪标志和原因位。

调用者必须提供相同物理时刻的实测 base_link、全碰撞几何桌面净空及原 reset root seed。此入口不从指腹采样猜测全手净空，不写 SDK 或推进时钟。终止环境只保留当前 offset/角度，调用方仍负责禁止写入终止行。投影输出要经过原双手 SDK 接口的 carriage 行程检查。当前未建立原生全手净空来源；此包不授予 Teacher 资格，旧控制路径保持不变。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_flexible_wrist_projection.py tests/test_dh116_deformable_bimanual_step.py \
  tests/test_free_wrist_grasp.py tests/test_free_wrist_mount.py tests/test_surface_contract.py
~~~

先新增失败测试，缺模块时收集失败，日志 SHA256 `18f896b0bd225a6bdd942685556bdb340a337243996b1cfc8141848c828ef50f`。随后 CPU 测试验证不同左右 seed/实测净空、禁止动作时撤回旧下压目标、live mask、原内禀角限制和同设备数值。新增原任务默认净空核对，最终 104 项相关回归。新增原生物理步、优化更新均为 0。
