# 双手柔性杆逐指接触诊断

对同一物理步的 node/mesh/substep 接触对，先对账独立保留的节点合力和每个 mesh wrench，再按左右手和原指腹归档。每个指腹输出平均法向力、作用于杆的世界冲量、有效子步数、加载节点与其离散 rest station。分别统计各手必需指腹和双手必需指腹在同一个求解子步的同时加载，避免跨子步拼接。保留桌面 SDF 载荷标记；未知捕获保留 NA。这里的 1e-5 N 是已有力证据审计的数值噪声门，不是有效夹持/成功阈值。

离散节点不具备亚节点材料锚点精度。加载不等于有效对向接触，桌面 SDF 无载荷不能独立证明无外部支撑。R0/R2、滑移和原生资格仍不赋值。原模型、物理参数、成功标准和 pair 校准配方保持原样。

Ubuntu CPU 回归 73 passed，新增 8 项。首次缺模块失败日志 SHA256 `b24dc2b56a58c05e54a0ebce7fb5ce9efb0b77c056d9122a787ec258dc25565c`，最终回归 `83f794a21a369c63687b7ccb5af809eaed516f8b2677df4919debfb00c3b23f1`。无新增 Isaac、真实 CUDA solver 步或 PPO 更新。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_flexible_pair_contact_diagnostics.py tests/test_dual_hand_rod_runtime.py \
  tests/test_dh116_deformx_pair_capture.py tests/test_dh116_deformable_grasp_task.py
~~~
