# F4 DH116 指腹碰撞体的 CPU 核验

原始左右 DH116 USD 独立核验，沿用资产清单的 SHA 和米制 Z-up。实际两侧原始 URDF 各含 6 个主动关节、5 个按原系数和偏置联动的关节，以及固定到各指的 5 个远端指腹。canonical action 依然按现有 24D 合同读取，首 12 维是左右手的主动关节速度，接触反力没有动作维度。

固定 DeformX CUDA 后端 `96b8514c7c55c724f1abba1fbf6ed6642b7db72b` 的编译包用于 CPU 审计。其 `body_geometry()` 在 10 个原指腹上均返回 `no collision geometry`。这组资产的 `CollisionAPI` 标在实例代理 Xform，三角 Mesh 在下一级。上游函数仅在自身带 `CollisionAPI` 的几何 prim 上提取顶点。DH116 入口复用现有 `usd_surface.read_surface()`，保留实例代理、核对 Mesh 的最近碰撞所有者与有效标志，并输出 PhysX link 原点坐标系内的米制顶点。试验性场景整体平移、旋转后，指腹本地顶点保持一致。

| 指腹 | 左侧焊接后顶点/三角形 | 右侧焊接后顶点/三角形 |
|---|---:|---:|
| 拇指 `finger14_link` | 939 / 1,874 | 937 / 1,870 |
| 食指 `finger23_link` | 1,989 / 3,974 | 1,989 / 3,974 |
| 中指 `finger33_link` | 1,989 / 3,974 | 1,989 / 3,974 |
| 无名指 `finger43_link` | 1,989 / 3,974 | 1,989 / 3,974 |
| 小指 `finger53_link` | 1,832 / 3,660 | 1,832 / 3,660 |

实际后端 `weld()` 与 `sdf.is_closed()` 验证 10/10 封闭。表中是**原始碰撞分支三角网格**。几何来自 `collisions/`，每个原始 `MeshCollisionAPI` 声明 `convexHull`。求解器尚未获得 PhysX 实际 cooked 凸包或已验证等价的凸包，禁止用表中网格直接代替该凸包进行正式资格验证。未烘焙 SDF、启动 Isaac 或执行接触。

`collider_plan()` 生成可由实际 `ColliderSpec` 构造的身份。单左拇指、单右拇指、双手 Pinch、Tripod 和 Power 每环境均指定 **一根连续柔性杆**。双手各有独立的 articulation key 与动态 link 列表，固定夹具如启用则单独使用 `dynamic=False` 的刚体 key。原刚性任务仍使用原场景。这里的 key 属于独立柔性场景绑定，原生步骤须用 articulation 的实际 `body_names` 和 `root_physx_view.link_paths` 重验。

## 单指下一步

1. F3 通过后，用独立 `deformable_rod` 场景绑定一个左拇指和一根杆。先读取或在原始网格、尺度和碰撞配置上重建原声明的 PhysX 凸包。离线重建需要对照 PhysX 的 cooked 几何、边界和采样距离，记录与原始三角面之间的差值。
2. 以核验后的凸包调用后端 `bake_body()`，核对网格采样分辨率和杆半径边界。由独立适配器显式调用 `RopeBatch.add_mesh()`，将实际同环境的 link 位姿和明确转换后的四元数用于注册。原版 `RopeColliders` 没有自定义网格注入口，直接传 `ColliderSpec` 会复现此次发现的错误。
3. 单实例原生先核对 link 原点、CoM、世界方向、单位和同一步真实 force/torque，再校验反作用力的应用位置、零载荷恢复和无重复施力。完成单左指后再验证单右指、双手同杆和三种抓取。此阶段不修改原 371D、24D、碰撞体或抓取成功标准。

## 验证记录

CPU 审计退出 0，私有 JSON SHA256 `3a68a5ba96e2ec7c96ddcf3e25d9e1bc7382b6d5808c5bcf72f9e89f26c8853e`。实际 `isaac_bridge.py` SHA256 `7a593ec2b56f4115e3c04ebe5cd6969294c7047ec8c577012bfcb3c1bf649eae`。证据目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-f4-pad-audit-WdzGIb`。

```bash
f1=/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-f1-20261112-883RS4
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
export LD_LIBRARY_PATH="$f1/sdk/lib64"
export PYTHONDONTWRITEBYTECODE=1 DH116_NUM_THREADS=2
bash scripts/python.sh scripts/audit_dh116_deformx_colliders.py \
  --ropesim-package-root "$f1/build-attempt3/python" --output /tmp/dh116-flex-f4-pad-audit.json
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_flexible_colliders.py tests/test_dh116_deformx_cuda_source.py
```

OpenUSD CPU 回归 9 通过、0 失败，日志 SHA256 `7f713cf6414cacde84296eff4fd35d05a94aabe49a3a551baa5d3ebc7fe3e2eb`。裸解释器缺 `pxr`，同样测试 8 通过、1 跳过。528 个受保护文件和 2 个历史 Checkpoint SHA 复核通过。
