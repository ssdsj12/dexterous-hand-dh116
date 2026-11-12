# F4 PhysX cooked convex hull 读回准备

Ubuntu 安装的 Isaac Sim 4.5 已验证有两条候选读回路径。此包完成源码和原 USD 的 CPU 核验，实际 10 个运行时凸包和坐标系待原生读回。未调用 SDF bake 或 `RopeBatch.add_mesh()`。

## 已安装接口

| 路径 | 4.5 安装证据 | 资格边界 |
|---|---|---|
| 实时凸包查询 | `get_nb_convex_mesh_data(meshPath)` 与 `get_convex_mesh_data(meshPath, convexIndex)`，安装绑定标注需要仿真运行 | 须确认原实例代理 owner 或后代 Mesh 路径是否返回同一活动 shape，以及实际返回坐标系 |
| 同步烘焙请求 | `request_convex_collision_representation(stage_id, collision_prim_id, run_asynchronously=False, on_result=...)`。安装的 `ConvexMeshDataDemo.py` 在不运行仿真时调用；结果含 vertices、indices 和 polygon index span | Demo 把自身带 CollisionAPI 的 Mesh 返回点乘其 local-to-world 矩阵。DH116 碰撞 owner 是实例代理 Xform，必须验证该差异。请求所得凸包独立记录，等价性待与运行时 shape 核验 |

本机 `isaacsim/extsPhysics/omni.physx/omni/physx/bindings/_physx.pyi` SHA256 `b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9`。本机 `ConvexMeshDataDemo.py` SHA256 `63fb4fc050c5e1797cda08ba71a442b3ad94b9636d7d8a63106eaf1b3c45944a`。PhysX 将凸包、shape 局部姿态和每实例 scale 分开管理，坐标系确认需与实际 shape 和 link 同时核对。[NVIDIA PhysX 几何说明](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/Geometry.html#convex-meshes)

`read_native_pad_hulls()` 对精确 owner 与 Mesh 路径分别读取两个通道。解码器校验顶点、索引和 polygon span，保留原生 polygon 数据与派生三角形。原生顶点坐标系预设为 `unconfirmed`。CPU 测试用合成记录验证读回分类、索引验证及距离报告格式。

## 10 个源碰撞体

全部 owner 均有 `PhysicsCollisionAPI`、`PhysicsMeshCollisionAPI` 和显式 `convexHull`。全部 owner 和子 Mesh 都没有已编写的 `PhysxConvexHullCollisionAPI` 覆盖。已安装的生成 schema 给出 `hullVertexLimit=64`、`minThickness=0.001`，保存为 **schema 默认值**，运行时实际 cooking 参数需原生核验。schema SHA256 `829db2f41e2558fd5797fb3d211f7ec327e7468f377ff1316051ca2da51e3176`。

源 Mesh、碰撞 owner、link 各自的 USD local-to-world 矩阵已记录。原源 Mesh 点的 link → world → link 最大数值误差小于 `4.3e-17 m`，属于离线矩阵自检。原生仿真的实时 link、CoM 和 shape pose 均待读回。

| 侧／指腹 | 源网格顶点／三角形 | Qhull 估计顶点／三角形 | 样本双向最大距离 mm | 估计体积减源体积 m³ |
|---|---:|---:|---:|---:|
| 左拇指 | 5,622 / 1,874 | 658 / 1,312 | 1.677 | 2.5054e-7 |
| 左食指 | 11,922 / 3,974 | 563 / 1,122 | 2.176 | 7.5580e-7 |
| 左中指 | 11,922 / 3,974 | 564 / 1,124 | 2.176 | 7.5580e-7 |
| 左无名指 | 11,922 / 3,974 | 562 / 1,120 | 2.176 | 7.5580e-7 |
| 左小指 | 10,980 / 3,660 | 596 / 1,188 | 1.172 | 3.5408e-7 |
| 右拇指 | 5,610 / 1,870 | 657 / 1,310 | 1.728 | 2.5053e-7 |
| 右食指 | 11,922 / 3,974 | 562 / 1,120 | 2.176 | 7.5580e-7 |
| 右中指 | 11,922 / 3,974 | 562 / 1,120 | 2.176 | 7.5580e-7 |
| 右无名指 | 11,922 / 3,974 | 564 / 1,124 | 2.178 | 7.5580e-7 |
| 右小指 | 10,980 / 3,660 | 596 / 1,188 | 1.172 | 3.5408e-7 |

距离每方向使用 192 个确定性面积采样点，逐点计算到目标三角面的最小距离。JSON 保留两个方向的 min/max/P95 及 0.5 mm 内覆盖率。原源到估计覆盖 61.98%–83.85%，反向覆盖 64.58%–89.58%。这是离散采样，不是严格的连续 Hausdorff 上界。Qhull 估计从原碰撞分支三角网格得出，未应用 PhysX cooking 参数。估计顶点数甚至超过本机 schema 的 64 顶点默认限制。**以上没有实时 PhysX cooked hull 顶点或对比结果。**

## F3 退出资格通过后的原生读回步骤

1. 使用同一台 Ubuntu 的 Isaac Sim 4.5，单独一台 App、一个环境。按只读原 USD 引用生成左右手，保持碰撞 approximation、scale、关节和物理配置，检查实际 body/link 路径和 USD stage cache ID。只读取几何，不接入 rod 或外力。
2. 先用非零 link-to-world 变换与非零 shape-to-link 变换的隔离校准凸体，核对两条 SDK 返回的顶点究竟在哪个坐标系。查询时刻记录 actual physics tick、SDK link/shape pose 和 USD 转换，不能把 USD 默认时刻当实时位姿。
3. 运行物理后，对每个实际指腹通过现有 `read_pad_triangles(stage, body_path)` 获取 owner 和 Mesh，再用 `read_native_pad_hulls(stage, collision_owner, mesh_path, simulation_running=True)` 逐个读取。在同一个停住的样本时刻复制全量数组，保存两条路径的成功数、零返回和 callback 状态。对解析失败或 SDK 路径缺失的记录保持缺失，先保存证据再研究绑定。
4. 用校准结果将实时读回转换到 link metres。对比 live 与同次请求结果、源三角面与 live 的距离、覆盖率、体积和边界，保留 raw 与转换后的数据及 frame round-trip。只有验证是运行时目标 shape 且确认坐标系的闭合凸包，才能进入单指 `bake_body()` 与显式 `RopeBatch.add_mesh()`。全程保留独立输出、原始物理保护和自然退出证据。

## 复现

```bash
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
export PYTHONDONTWRITEBYTECODE=1 DH116_NUM_THREADS=2
schema=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.usd.schema.physx/plugins/PhysxSchema/resources/generatedSchema.usda
bash scripts/python.sh scripts/audit_dh116_cooked_convex_hulls.py \
  --installed-schema "$schema" --output /tmp/dh116-f4-authored-hulls.json
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_cooked_convex_hulls.py tests/test_dh116_flexible_colliders.py \
  tests/test_dh116_deformx_cuda_source.py
```

CPU 审计退出 0，原始 JSON SHA256 `3e579900f9f530df4942c8e25b8c4839a19dd4a21e1931bcd48ea18689274274`。OpenUSD 回归 12 项通过，日志 SHA256 `df6ea1e5fb7a413e887dc5464efedff3af0a16337b9e65f4767bdbcfd923c62e`。私有证据位于 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-f4-cooked-S7EFnJ`。528 个保护文件和 2 个历史 Checkpoint 通过核验。
