# DH116 实际USD表面提取与实例代理审计

Baseline/Candidate：5e3bcc4 + dirty worktree；Inline Execution，源USD未改。父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。前置：[纯坐标基础](2026-10-15-dh116-contact-coordinate-core.md)。

## 交付

新增[usd_surface.py](../../Go2Pvcnn/go2_pvcnn/control/dh116/usd_surface.py)：显式body/visuals或collisions分支、实例代理遍历、唯一三角网格、单位/刚体/刚性变换/有限值/索引检查、向上查CollisionAPI与启用状态、几何SHA和已使用文件层SHA。忽略且只忽略空匿名session layer；源匿名层拒绝。输出保持原三角面编号，没有焊接或修改资产。Mesh-to-body明确转为列向量矩阵。

新增[audit_dh116_surfaces.py](../../Go2Pvcnn/scripts/audit_dh116_surfaces.py)：新目录输出，拒绝覆盖；左右各掌体+5末端刚体，视觉/碰撞源网格相等检查及图像。summary明确calibration_qualified=false、physics_stepped=false，姿态为USD authored transform，不是抓取动作。

## 真实资产结果

主仓库assets/dh116_v2/surface_audit_v1包含summary.json及6张PNG。成功退出0，12个刚体全部读到视觉/碰撞源网格且一致，5个加载文件层SHA固定。图像已查看左右拇指、左食指及左手概览；掌体为稀疏三角显示，仅用于定向，不据其空白判断网格缺面。

两个必须保留的下游问题：

1. 左右拇指局部坐标不同：左finger14_link主要沿负Z，右finger14_link主要沿正Y；左右网格面数1874/1870也不同，禁止简单复制/镜像局部点。应分别选面并用完整变换核验。
2. 12个碰撞owner的MeshCollision approximation均为convexHull。源显示/碰撞网格相同不证明PhysX cooked凸包与表面重合；后续接触选点需同时核验源面及有效碰撞面。不得取消壳体碰撞解决差异。

## 测试与命令

测试文件：[test_dh116_usd_surface.py](../../Go2Pvcnn/tests/test_dh116_usd_surface.py)。RED缺模块后13项通过；report覆盖拒绝测试RED缺脚本后14项通过（5.21s）。包括真实实例代理、实际文件层SHA，以及错误单位/缩放/碰撞禁用/索引/多网格拒绝。完整DH116专项105项通过，无skip；本轮无物理步或抓取。

与此前环境一致：go2 Python、pxr库路径、真实表格目录、taskset -c 0，OMP/OPENBLAS/MKL各1。运行：

```bash
python scripts/audit_dh116_surfaces.py \
 --asset /home/xk/coding/M1/assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd \
 --output /home/xk/coding/M1/assets/dh116_v2/surface_audit_v1
python -m pytest tests/test_dh116_no_load.py tests/test_dh116_usd_contract.py \
 tests/test_dh116_linear_coupling.py tests/test_dh116_table_core.py \
 tests/test_dh116_panda_mount.py tests/test_dh116_contact_frames.py \
 tests/test_dh116_usd_surface.py -q
```

## 状态

D1实际网格提取DONE；指腹/掌心选面及外法向语义标定仍OPEN，不将这些图当完成标定。下一步分别选左右指腹表面、验证接触面与凸包关系，再固定参考点和SHA，进入真实USD/PhysX Jacobian。D3桌面与D4闭指/夹持/抬升仍未执行。没有要求用户重选路线。
