# 柔性物理交换的逐步几何读回

把现有 USD 场景初次核对的 link/shape 映射接入控制交换的 `read_geometry` 参数。每次读取实际 PhysX `get_link_transforms` 和 `get_coms`。使用同一 link 姿态换算世界 COM，原始 XYZW 转原控制 WXYZ，复用旧 `collision_clearance` 核计算每手全包络净空。保留指腹的位置和原 SDF 所需 XYZW。读取前后核对 SDK 时钟、原实例、实际 link 名称与路径。每步无需重新打开 USD 或读取所有哈希。

CPU double 回归 **43 passed**，新增 2。已验证非单位旋转对应的局部 COM 平移转换、双手实际不同高度、只读数组复制、link 替换和读取期间时钟变化。测试首次因模块不存在而失败。实现后测试对 float32 Z 的期望误用了实数 0.36，换成 SDK 表示后的结果通过；运行逻辑与原约束未放宽。私有日志 SHA256 最终 `8eefee8cc9ebcc91678c765fa6cb89026b9a7de88d63834fe93a440e565b95a4`。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_flexible_live_geometry.py tests/test_dh116_flexible_physical_exchange.py \
  tests/test_dh116_flexible_wrist_projection.py tests/test_dh116_deformable_bimanual_step.py
~~~

实际新原生物理步和 PPO 更新均为 0。此路径的净空沿用源 USD 全手几何包络；仍缺非指腹 live cooked 校验和本任务原生接触资格。
