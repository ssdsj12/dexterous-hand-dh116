# DH116 接触坐标与点Jacobian纯计算基础

Baseline Ref：5e3bcc4。Candidate Ref：同提交+dirty worktree，未提交代码；保留其他改动。
父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。用户已确认书面设计，沿用Inline Execution，不再等待抓取路线选择。

## 范围与方法

按[基础实施计划](../../docs/superpowers/plans/2026-10-15-dh116-contact-coordinate-core.md)新增纯函数[contact_frames.py](../../Go2Pvcnn/go2_pvcnn/control/dh116/contact_frames.py)及[测试](../../Go2Pvcnn/tests/test_dh116_contact_frames.py)。本轮实现D1/D2数学基础，不是实际网格标定或PhysX Jacobian资格。

surface_reference从三角面重心坐标求局部点和绕序法向；palm_frame构造+Z掌面外/+Y腕到中指投影的右手帧；transform_reference拒绝反射/缩放/坏法向；point_jacobian明确输入原点→接触点的世界系偏置，完成6×18空间Jacobian平移及按名称缩减为3×13。Panda7列保留，DH11611→6用现有线性耦合。输入必须有限，输出溢出拒绝。法向绕序不自动证明外表面，不能凭此批准指腹。

## TDD与回归证据

- 基线：52 passed in 0.39s。
- RED1：contact_frames模块不存在，pytest收集ImportError。
- GREEN1：23 passed in 0.10s，表面/坐标/异常输入。
- RED2：新增16个Jacobian案例因缺point_jacobian失败，原23通过。
- GREEN2：39 passed in 0.10s。
- 最终DH116专项：91 passed in 0.41s，无skip，含真实Excel与真实USD安装结构测试。
- 三个差分步长1e-4/1e-5/1e-6下合成耦合旋转点误差≤1e-8 m/rad。仅合成解析例，不是实际DH116几何差分结果。

从Go2Pvcnn执行，设置PYTHONPATH包含当前目录和Isaacsim的omni.usd.libs目录，LD_LIBRARY_PATH包含conda/lib、omni.usd.libs/bin和isaacsim/kit，DH116_TABLE_SOURCE_DIR指向下载目录；OMP/OPENBLAS/MKL各1线程，taskset -c 0。运行：

```bash
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
 tests/test_dh116_no_load.py tests/test_dh116_usd_contract.py \
 tests/test_dh116_linear_coupling.py tests/test_dh116_table_core.py \
 tests/test_dh116_panda_mount.py tests/test_dh116_contact_frames.py -q
```

## 实际USD只读检查

检查主仓库assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd，metersPerUnit=1。左右base_link及finger14/23/33/43/53_link均存在刚体。网格为instance proxy，普通PrimRange不会列出，必须使用Usd.TraverseInstanceProxies()。

每个上述刚体下找到两份显示/碰撞分支网格，不能只按mesh名称选取。拇指左1874/右1870面，食/中/无名3974面、小指3660面；左右掌体分别164732/159546面。Mesh自身无CollisionAPI不等同父级没有碰撞，未据此更改碰撞配置。

## 结论及下一步

纯计算基础DONE；D1实际十个指腹点/掌心帧与SHA标定、D2实际USD/PhysX差分仍OPEN。下一步读取实例代理的完整prim路径和网格到刚体变换，区别visual/collision分支，选面并输出标记证据。之后再做重力保持、桌面wire、真实闭指/夹持/抬升，未启动物理步或训练，未修改旧O6入口。
