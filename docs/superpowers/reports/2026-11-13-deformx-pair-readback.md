# DeformX 逐接触对读回与有界 GPU 校准

## 实现与构建

现有 DeformX 节点合力及逐 mesh wrench 无法恢复每个节点和碰撞体的对应关系。在单独 CUDA 源码工作区，对锁定提交 `96b8514c7c55c724f1abba1fbf6ed6642b7db72b` 增加可选读回。记录子步索引、节点索引、mesh 索引、世界坐标接触点、外法向、杆侧力、mesh 原点力矩。开关默认关闭；补丁沿用 GPL-3.0-only，不提交外部源码、二进制或原始数据。写入有容量上限，溢出拒绝。入口限制一根杆、仅 mesh SDF、显式子步 `step_free`。读回在同一互斥区和 CUDA stream 同步后返回接触对及原聚合值。reset 后必须重新完成一次 step。

重放生成器前校验源码锚点，重放后的补丁与构建补丁字节一致。锁定原仓库保持干净。CPU-only 构建进程隐藏 GPU，使用已有 Python 3.10、CUDA Toolkit 12.8.61 与 SM89。加载 cp310 binding、检查 sm_89 和新接口通过。未构造原生 RopeBatch，也没有运行 CUDA kernel 或 Isaac。

| 证据 | SHA256 |
| --- | --- |
| 生成补丁 | `dad186360e8a9c31e3ace4c6ec6f7641e1d1f8a17ff26c42a267767a730d75` |
| 独立扩展 | `92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0` |
| 构建日志 | `0e0a203aa0ac186cf3e85e9f79bd01ea514f3cb621df4623a3a4227ec56018b3` |
| sm_89 检查日志 | `16ebb67f2a7fa9a1972c4f1b21a403108aa6096f25ac94da448412647b94e47d` |
| CPU 回归日志 | `cfcc8339139973a18af6fa7a507c5302ac2d6a3624a344524781aab5c0d36ad3` |

## CPU 回归与校准边界

使用 Ubuntu 现有 launcher、屏蔽 GPU。62 passed，包含 12 个接触对专项用例：

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_deformx_pair_capture.py tests/test_dh116_deformable_force_evidence.py \
  tests/test_deformable_grasp_runtime.py tests/test_dh116_deformable_grasp_task.py
~~~

校验唯一身份、法向、冲量换算、每节点力以及每 mesh 反力和力矩的步均值。重复、非有限值和容量溢出均拒绝。配方核对实际加载的扩展路径与 SHA。外层等待自然退出后重新读取 raw packet，核对聚合、时钟、接触正载荷和 reset 失效。缺少注册审批时入口在 solver 构造前拒绝。

新预算仅用于已知平面和 3 节点自由杆，GPU 3，单次 90 秒，1 CUDA solver step 内 12 子步，0 Isaac 物理步、0 PPO。初始穿入 1 mm 用于标定，不能计为抓取。此前没有运行该配方。时钟目前仅在该单步标定边界核对；通用环境接入还需绑定生产时钟，验证双向传递、真实手桌接触、材料 anchor 滑移和外部支撑。Teacher PPO、R2 和三个抓法的原生资格保持阻断。
