# DeformX 单精度时钟与长窗口

锁定的 CUDA 源码 `YarnBall.h` 将 `MetaData::time` 定义为 float，`csrc/batch.cu` 的 `SimAccess::finish` 每个外部物理步累加一次 float dt。现有 Python 路径假设绝对秒数接近实数相加。CPU 重放显示第 242 步单步误差超原 1e-8 秒，第 2400 步累计误差约 -54.84 µs。前一假设会截断 32 控制步 smoke，后一假设会截断 5 秒 Hold。

双手运行时改为检查每次 SDK 整数步、杆整数子步与实测 solver 秒数。保存上一已完成秒数，下一步先核对真实读回未被外部推进，然后比较 float32 一步累加；保留 CPU 测试的高精度表示路径。拒绝停滞、倒退、非有限、漏步和分辨率不足的时钟。reset 不推进 solver 秒数，整段连续性逐步建立。现有 60/480 Hz、成功时间标准、物理与 CUDA backend 均未修改。

Ubuntu 屏蔽 GPU 的回归 **71 passed**，新增 11。包括 256 步 smoke、2401 步原 Hold 端点范围、2880 步独立预算、已有非零 solver time，及实测格式的单精度累加贯穿双手适配器。测试不运行真实 solver 或训练。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_flexible_solver_clock.py tests/test_dual_hand_rod_runtime.py \
  tests/test_dh116_deformable_bimanual_step.py tests/test_dh116_flexible_physical_exchange.py
~~~

缺模块失败日志 SHA256 `c4f9c5a319fb647943ef4bdbc8dada844a042ec714a5ca719a4f61858a692554`，71 项通过日志 `0a204a815da30c83b30d4f3a62bcd482c58c3e3a9498ccad1ebd8b64580275d8`。CPU 重放不授予原生时钟或 PPO 资格。未改已经批准的 contact-pair 或 lifecycle 配方指纹。
