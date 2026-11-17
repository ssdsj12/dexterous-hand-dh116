# 双手 PhysX 到 DeformX 的位姿交接

基线 `9bfeee711a0bfa3f02532774862c0ab424ab1bbd`。此前的 PhysX 几何读回要求两套时钟相等。接入双手交换时，PhysX 必须先完成一个子步，DeformX 才能取得更新后的碰撞位姿。旧接口在这里会拒绝正常的 SDK 读回。

- `LiveGeometryReader.read_after_sdk_step()` 仅允许 SDK 比给定的 rod 边界领先一个 480 Hz 步。读回前后验证 SDK 时间、rod 的完整身份和时钟、双手 link 路径。返回新 SDK 步号与前一 rod 步号，保持此时的位姿与同步完成后的 rod 数据可区分。
- `FlexiblePhysicalExchange` 在 SDK 步前保存 rod 时钟，在下一次 rod 求解前核对交接阶段、步号和未推进的 rod 身份。调用方可显式传入这个读回方法。既有同钟读取及独立测量回调保留原行为。
- 动作、力矩限制、推进次数、反作用力滞后一拍和关节保护均沿用已有路径。交接失败后关闭该交换实例并保留已推进的 SDK 状态。原生资格及训练奖励仍未授予。

## CPU 验证

新增测试先复现 3 个失败，均为交接 API 缺失。最终 7 个相关测试文件 **92 passed**，新增 6 项覆盖准确一拍、相同或跳步、身份变化、读回期间时钟变化、失败后禁止重放和双手八步完整交换。物理替身对每一步写入右手指腹的不同位置，并检查实际交给 rod 的位姿随 SDK 步变化。

在 Ubuntu 使用 `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider`，测试文件为 `test_dh116_dual_hand_clock_handoff`、`test_dh116_flexible_live_geometry`、`test_dh116_flexible_physical_exchange`、`test_dual_hand_rod_runtime`、`test_dh116_deformable_bimanual_step`、`test_dual_hand_reset`、`test_dh116_flexible_rod_boundary`。

CPU 日志 SHA256 `42fd3ce2bba446930187358d97d7e870c8664e44c60fa5e58243ea17e03ee034`，原始日志保存在 Ubuntu 私有证据目录。该测试运行 0 个原生步，0 次 PPO 更新。下一步把实际 Reset 与交换连到同一双手生命周期，并绑定真实物理证据。
