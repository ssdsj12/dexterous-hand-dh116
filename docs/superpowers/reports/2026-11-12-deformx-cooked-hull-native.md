# F4 原始 DH116 实时 PhysX 凸包读回

单实例 Isaac Sim 4.5，以只读引用打开摘要锁定的两套原手 USD 和一套独立非对称校准凸体。物理使用 CPU device 与 480 Hz。reset 后显式步进 2 次，采样时实际 SDK physics step 为 4。外层退出码 0，有 `simulation_context_cleared.json` 和 `shutdown_complete.json`。未启动 rod、未烘焙 SDF、未施加外力或进行抓取。

## SDK 通道

- 10/10 指腹的原 rigid body 由 `get_rigidbody_transformation` 返回当前位置和 xyzw 四元数。
- 10/10 指腹的 **原碰撞 owner Xform 路径**通过实时 `get_nb_convex_mesh_data` / `get_convex_mesh_data` 各返回 1 个活动 shape 凸包。
- 后代 Mesh 路径的实时查询均返回 0。独立同步 cooking request 在 owner 路径返回 `RESULT_ERROR_COOKING_FAILED`，后代 Mesh 路径返回 `RESULT_ERROR_INVALID_PARSING`。这两个通道及错误保留在同一指腹记录里。
- 校准凸体的 mesh-to-link 和 link-to-world 均有独立非零平移与旋转。两条查询在校准凸体上均与 mesh-local 顶点唯一匹配，点集对称误差 `3.58e-9 m`。link-local 和 world 候选误差分别为 `0.196 m` 与 `1.164 m`。
- 在本次组成的原手场景中，所有指腹的 owner、子 Mesh 与 link 局部基准重合，最大变换矩阵差 `2.69e-16`。SDK 实际 link 世界位姿与 USD 默认位姿矩阵元素差最大 0.0518，后续姿态转换应使用同一物理时刻的 SDK world pose。按 SDK pose 做的数值往返由 JSON 记录。将原始凸包数据当作 world 位置会产生大于 1.55 m 的形状距离。

| 指腹 | 左侧 live 顶点／原生 polygon／三角面 | 右侧 live 顶点／原生 polygon／三角面 |
|---|---:|---:|
| 拇指 | 36 / 54 / 68 | 36 / 52 / 68 |
| 食指 | 34 / 60 / 64 | 34 / 60 / 64 |
| 中指 | 34 / 60 / 64 | 34 / 60 / 64 |
| 无名指 | 34 / 60 / 64 | 34 / 60 / 64 |
| 小指 | 28 / 47 / 52 | 28 / 47 / 52 |

原始碰撞网格与活动凸包的 192 点/方向采样双向最大点面距离为 **1.044–2.047 mm**，原 Qhull 估计与活动凸包距离为 **0.601–0.686 mm**。每个指腹的原生顶点、polygon、三角面、包围盒、双向 min/max/P95、0.5 mm 采样覆盖率及体积差均单独保存。体积、距离和框架数据由本次实际凸包与两种离线来源比较得到。

当前结果涵盖**原始资产在本次 CPU PhysX 组成与物理时刻**的实时凸包。GPU PhysX cooking/shape parity、原生正负 SDF、实际单指载荷与接触仍待验证。进入 GPU SDF 前，需用同一原资产和目标 PhysX device 重跑短时只读读回，比较实际 hull 顶点、拓扑、尺度与实时 link pose。比较工具只对局部基准重合且得到实际唯一凸包的指腹输出对比。无 GPU 实验或 SDF 结果混入本包。

## 命令与证据

```bash
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
export PYTHONDONTWRITEBYTECODE=1 DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=3
bash scripts/python.sh scripts/verify_dh116_cooked_convex_hulls_native.py \
  --headless --device cpu --steps 2 --output "$native_output" --execute-native
bash scripts/python.sh scripts/verify_dh116_cooked_convex_hulls_native.py \
  --compare-run "$native_output" --output "$compare_output"
```

独立证据目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-f4-native-giMQKH`。SDK 实时读回完成后添加并执行离线对比入口。只读预检打印摘要和预算，默认不打开 App。脚本仅在显式原生模式取得单实例锁，并限定 10 个 SDK 物理步以内。默认或离线对比模式无需关闭任何现有 App。

| 证据 | SHA256 |
|---|---|
| 外层退出码 | 0 |
| 原生摘要 | `cc223eca1a64f332eaace273f1a5b79050cbd2ef894d19a50e03e0bc60a43f13` |
| 非对称参考体 | `6432d8a25132dd85cfb4c4f88d3b294555eee6569a800bd281de1fb0dfd7fab5` |
| 全部原生对比 | `4ae7385f6a711da63aaa06e1db4d840b078e2e894141e596d88564280d33f07d` |
| OpenUSD CPU 回归日志，14 通过 | `c7d8aa0cb1bd682c0d8da110c6f3532328b15680cbc555f4cac4ef7ce3738156` |

裸解释器回归 13 通过、1 跳过（缺 `pxr`）。528 个保护文件和 2 个历史 Checkpoint 通过 SHA 验证。
