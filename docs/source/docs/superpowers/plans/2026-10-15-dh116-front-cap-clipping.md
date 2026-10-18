# DH116 Front Cap Triangle Clipping Plan

> **For agentic workers:** REQUIRED SUB-SKILL: executing-plans. Inline Execution，现有隔离工作区，不派代理。

**Goal:** 修正粗三角面跨越原12mm指腹区域导致区域内部分也被当作非前区的过度保守误分类，保留原1mm/12mm/60°边界。

**Architecture:** 接触/完整表面查询网格不变。只对非前区保护用的remainder网格做源三角面几何分割：用严格内接于12mm球的凸多面体裁剪各个法向合格源三角面，保留全部外侧碎片；只有与参考点连通的内侧碎片可分类为前区。原body不整体排除。

**Tech Stack:** NumPy、SciPy ConvexHull、Open3D已有查询、pytest。

## Global Constraints

- 半径12mm、法向dot≥.5、局部实测patch6mm/30°、非前区1mm和穿透1mm不变。
- 不改源USD、碰撞PLY、力积分、wire半径/刚度/摩擦、关节或夹力限制。
- 裁剪多面体顶点必须位于球内；截断只减少合格前区，不允许扩到球外。分区面积守恒，源面ID可追溯。
- 不根据异常帧把整个三角面标成前区。未连接参考点的候选片仍留在remainder。
- CPU core0；不提交/推送。当前无运行中的Isaac试验；旧结果不能重新标PASS。

## Task 1: Geometric partition core

Create `Go2Pvcnn/go2_pvcnn/control/dh116/front_partition.py`, tests `Go2Pvcnn/tests/test_dh116_front_partition.py`.

Interface `partition_front_cap(vertices,faces,point,normal) -> dict`，返回`front_triangles`/`remainder_triangles`形状N×3×3及每片`front_source_ids`/`remainder_source_ids`，并记策略/数值收敛参数。源面法向不重估为其它方向。

- [ ] RED：单个长平面三角片穿过球中心，旧整面判法不会给前区；要求球内近处属于front、球外仍在remainder、片面积之和等于原面积；球外/法向背面/不连接组件不豁免；坏输入拒绝。
- [ ] GREEN：构造三级icosphere单位方向，全部顶点归一到.012m，ConvexHull得内接凸多面体；逐源三角片Sutherland–Hodgman半空间裁剪，外片累计，内片为候选；沿实际原边被多面体截出的非零线段建立邻接，从参考点所在源面BFS，只保留连通候选。多边形扇形三角化保留源ID。
- [ ] 核对实际左食指face168：内部近点距参考约4.2mm/27.18°，必须正确分区；远端32mm仍非前区。验证所有front顶点≤12mm及源法向dot≥.5，面积守恒和完整remainder。

## Task 2: Guard-only integration

Modify `Go2Pvcnn/go2_pvcnn/control/dh116/pad_region_guard.py` and relevant tests.

- [ ] RED：完整实际凸包，原记录切触/失败位姿可用新分区算出几何真实余量；所有remainder部分仍必须拒绝过近wire，不允许整面排除。
- [ ] GREEN：full原网格/signed-distance/6mm小patch完全不变，只将other建为remainder片。记录`front_cap_method`、源face IDs及分区SHA。原`classify_front_cap`留作旧保守分类/API，不冒充新裁剪。
- [ ] 运行全部DH116/moving-contact回归，重算新原位/独立压入余量/旧失败现场。仅几何通过才允许独立新编号GPU有载试验；仍需连续接近/1s保持/30mm抬升实际证据。

## Root cause evidence

`/tmp/dh116_limiting_pad_face.py`、`/tmp/dh116-limiting-pad-face.log`读取pilot6异常现场。实际left_index源face168顶点距参考4.057/4.205/32.386mm，面法向夹角27.18°。旧candidate要求整面三顶点≤12mm，导致靠近参考的前表面也被当作非前区。所有96左候选独立压入.2mm都在该外区保护失败，与该三角化问题一致；并不证明真实物理碰撞。
