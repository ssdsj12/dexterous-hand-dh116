# F4 原始 DH116 指腹 CPU／GPU PhysX cooked hull 一致性

在 Ubuntu Isaac Sim 4.5 上先运行 CPU，再运行 CUDA:0，分别创建单独进程、同一原始左右手 USD 和非对称参考凸体。两次 reset 后各显式推进 2 个 480 Hz 物理步，实际读回时钟、源资产 SHA 和两种局部坐标系校准均匹配。两次 App 自然退出，外层退出码分别为 0 和 0。GPU 进程以 `CUDA_VISIBLE_DEVICES=3` 启动，CUDA:0 为该进程可见的第一个设备。

| 核验 | CPU 实测 | CUDA:0 实测 |
|---|---|---|
| SimulationManager 实际设备 | `cpu` | `cuda:0` |
| 活动 Tensor SimulationView device ordinal | −1 | 0 |
| PhysX scene GPU dynamics / broadphase | 关闭 / `MBP` | 开启 / `GPU` |
| PhysX 设备 override | −1（scene 决定） | −1（scene 决定） |
| 参考体返回坐标系 | mesh-local | mesh-local |
| 指腹 owner 读回 | 10/10 | 10/10 |

全部 10 个 active owner 凸包通过拓扑比较。CPU/GPU 之间先做顶点一一对应，再比较各 polygon 的有向环，容许顶点排列、polygon 排列和环起点变化。**全部对应顶点的最大距离为 0.0 m，有向拓扑完全相等。** 坐标基底与各自的 link-local 一致，矩阵误差不超过 `2.69e-16`。本次 parity 未使用上一步未记录设备身份的 CPU 归档充当 GPU 输入。对该旧 CPU 归档的负向自检虽然几何完全相同，仍因没有两份独立设备证据返回 1。

这项资格覆盖同一原始资产、相同 probe 组成和时钟下的 CPU／GPU active PhysX cooked hull 一致性。SDF bake、同 clock 的单指真实接触、反作用力和后续双手抓取仍待原生验证。

## 复现

使用同一环境，串行执行两个命令。两进程均应持有各自的单实例锁直至自然退出，再执行离线比较。输出路径须独立。

```bash
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
export CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 DH116_NUM_THREADS=2
bash scripts/python.sh scripts/verify_dh116_cooked_convex_hulls_native.py \
  --headless --device cpu --steps 2 --output "$cpu_run" --execute-native
bash scripts/python.sh scripts/verify_dh116_cooked_convex_hulls_native.py \
  --headless --device cuda:0 --steps 2 --output "$gpu_run" --execute-native
bash scripts/python.sh scripts/audit_dh116_cooked_hull_device_parity.py \
  --cpu-run "$cpu_run" --gpu-run "$gpu_run" --output "$parity_output"
```

实际独立证据 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-f4-parity-prep-Vu9H0f`。其中配对 launcher 记录了两次退出码和入口 SHA。

| 文件 | SHA256 |
|---|---|
| parity 对比 JSON，退出码 0 | `aa424c5c444ad4e1020d744f1941a20380b105e8155e81f9af1510e7d7d1b7c4` |
| CPU 原生摘要 | `43cfd60d73b97520b3caa9d0b871f45a38b144f72554d165ebc5e273eada6bb3` |
| GPU 原生摘要 | `2de331289145d4be824243db0eeb64b796f24a92cee7b93ee61a37244e9bbde5` |
| launcher | `67cae5efa7ab4faa4400c7827a2f549a1ec5526f6ebec2cc5026b4f0f86a9aa4` |
| OpenUSD CPU 回归 | `7169fd14394853df63bcf41d6abae909fb4832b8b70fd621ebc6172dc07ffaa7` |

OpenUSD CPU 回归 16 项通过。裸解释器 15 通过、1 跳过（缺 `pxr`）。原始资产、528 个保护文件及 2 个历史 Checkpoint 通过 SHA 核验。
