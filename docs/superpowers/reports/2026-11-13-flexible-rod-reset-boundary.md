# 柔性杆 Reset 与测量状态边界

新增 `RodBoundaryState`，将双手同杆运行时的已拥有数据交给既有 `DeformableRodStateProvider` 校验。要求 run、环境、episode、reset、rod、SDK 及求解时钟与当前边界一致。Reset 后可读取真实位置、速度、中心线、切向、曲率和静止段长，接触保持全 NaN 与 false mask。已完成物理步读取原 solver 节点合力并保留其作用侧和 SI 单位。检查现有 provider 的后端文件 SHA、有限性、材料顺序、应变和数据所有权。

原版本只在测量后提供节点状态；新入口允许首个动作之前绑定 reset 杆状态。它不运行 solver，也不授予完整 371D、有效接触、原生或 Teacher 资格。调用者负责由实际加载模块绑定后端文件，原冻结模型、奖励和任务合同未被替换。

Ubuntu CPU 64 passed，新增 4 项。首次缺模块日志 SHA256 `aad1320375027cf0a9b8c4b666f240191ff3cfaff65fca06017545edecc24262`，最终回归 `e079511d9ecda6aa7279d51d312f91c584f197ac0ad62f91f84cf1331bc751b3`。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_flexible_rod_boundary.py tests/test_dual_hand_rod_runtime.py \
  tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_privileged_teacher.py
~~~

新增原生物理步和优化更新均为 0。后续由真实双手 reset 和测量主循环调用；新后端指纹与原模型不混用。
