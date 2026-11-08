# Cooked readback04 只读资格审查

本审查只读取2026-11-07已保存的8份原生 getter JSON、冻结捕获源码和安装接口声明；没有启动Kit/Isaac、调用cooking、修改碰撞几何或扩大数值容差。移动腕恢复仍未取得资格。

## 数据层次及重新计算

原始目录：`/mnt/mydisk/xk/dh116/runs/tripod_recovery_20261107T0815Z/cooked_readback_04`。派生分析保存在本阶段独立根目录：

- `cooked_readback04_topology_reanalysis01.json`，SHA256 `56643841cb68f578d35ec30807b3b5d70641f08f90866b3e8a7839dcd58b74e5`；输入8个文件逐一保存SHA。
- `cooked_readback04_frame_reanalysis01.json`，记录直接getter JSON的payload校验与三角面约束的仿射诊断。

272/272返回count=1。全部索引槽有唯一覆盖、每条有向边有唯一反边、Euler为2、绕向和体积正常，证明这些保存的索引表是闭合可定向拓扑；不能把原`closed_convex=false`解释成272个网格都开口。272/272均有非严格共面或非严格包络的quad/ngon，最大点面残差0.2189295mm；三角面均在原数值容差内。全部返回顶点均是各自离线凸包的极点，但离线凸包不是原生碰撞形状身份证明。

## 排除本项目捕获层漏变换

冻结collector SHA256 `ddf703d192958b2a1a88634c8a8433f46d43e43c15575cd02cee0a5803a75219`。原生probe直接把真正`PhysXCooking`接口交给collector；getter返回后先原样验证，再进行仅JSON编码类型转换的`_plain(data)`。mesh-to-body/owner/world变换只用于验证成功后另存的world顶点，04的272项均失败，因此根本没有执行该派生分支。JSON payload SHA272项全部匹配。未保存C++二进制buffer，不能声称可重新核验原二进制字节。

安装版 `_physx.pyi` 2005–2024行定义plane为n·x+d=0、num_vertices为每面数量、index_base为索引起点，官方随包demo采用相同索引切片。其SHA分别为 `b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9` 和 `63fb4fc050c5e1797cda08ba71a442b3ad94b9636d7d8a63106eaf1b3c45944a`。

进一步用每个hull全部三角面建立vertex到plane的共同12参数仿射拟合：272项均满秩12，A距单位阵最大2.265e-7，平移最大1.688nm；原三角入射残差最大12.26nm，而ngon残差拟合后仍0.2189303mm。结合冻结源码，未发现本项目对顶点单独scale或未同步plane逆转置的证据。

## 官方cooking允许近似合面；本批是否合法尚未获证

[PhysX 5.5.1对应106.5系列的固定源码](https://github.com/NVIDIA-Omniverse/PhysX/blob/32ae713f6cb35f1df2544ffa499fc98a39639a62/physx/source/geomutils/src/cooking/GuCookingQuickHullConvexHullLib.cpp#L1042)具有相近法向面的后合并阶段。其`canMergeFaces`重算法向/中心，并按`mPlaneTolerance`检查顶点距新面距离；判据不是float32机器精度下的严格共面。该源代码与本机106.5.7扩展属于相关发布系列，未证明本机二进制来自该确切commit，也未读取本次实际全局cooking参数。

[PxCookingParams官方说明](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxCookingParams.html)说明planeTolerance随hull尺寸增加，默认0.0007，参与QuickHull构建及合面；它不是接触判据或用户可随意扩大的几何资格容差。实测0.219mm与这种近似合面机制相容，所以不能仅据原严格共面检查失败认定坏cooking。实际参数/缓存版本/原生shape一致性未知，仍不能反过来宣称本批合法。

[固定版本ConvexHullBuilder检查](https://github.com/NVIDIA-Omniverse/PhysX/blob/32ae713f6cb35f1df2544ffa499fc98a39639a62/physx/source/geomutils/src/cooking/GuCookingConvexHullBuilder.cpp#L282)对属于该面的顶点跳过逐点面测试，并不采用本项目的精确incidence检查。现有诊断宜区分`closed_topology`、`plane_consistency`和`stored_vertex_envelope`；此建议没有被用于放宽当前qualification。

## 保守包络的有限用途

可离线构造包含返回顶点的凸包，并扩展原plane半空间以同时包含其原半空间集合和所有返回顶点。若归一法向为n，原约束为n·x≤−d，可令h=max(−d,max_v n·v)，采用n·x≤h。这在数学上包含保存的两种表示；必须另行验证有界性与所声明的frame/scale，不能把它当作PhysX重烹饪或等价面。

[固定版本GJK支持函数](https://github.com/NVIDIA-Omniverse/PhysX/blob/32ae713f6cb35f1df2544ffa499fc98a39639a62/physx/source/geomutils/src/gjk/GuVecConvexHull.h#L425)使用变换后的极点支持；这支持把顶点包络作为一种待绑定的保守诊断假设，但不证明当前getter数据就是live mesh。未知live映射下它只能新增拒绝/不确定，不能因包络无碰撞就准入接近动作、降低原碰撞保护或宣称接触资格。

## live attachment的真实资格路径与剩余阻断

[PxRigidActor::getShapes](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxRigidActor.html)是返回实际附着碰撞shape的公开C++读取接口；[PxShape](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxShape.html)提供geometry、local pose及相关flags。完整证明需要在安全的同一fetchResults后时钟读取实际actor/shape、PxConvexMeshGeometry的mesh/scale、局部位姿和simulation/query flags，再把实际mesh数组绑定到保存表示；当前Python getter不提供这组指针/mesh绑定。

[Omni官方scene query说明](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physx/docs/api/python.html)和已装stub支持box/sphere overlap及raycast，结果带collision和rigid_body路径，可在未来唯一原生实例的既有场景中做不写场景的反证检查。必须避免`overlap_mesh/shape`和mesh sweep作为只读资格捷径：官方接口注明输入mesh可能首次cooking凸近似。有限射线/球探测能发现不一致，不能证明全部支持方向等价；还要核query shape与simulation shape是否同一。

安装Python property query仅暴露路径、local pose、AABB和volume，没有完整convex mesh/scale/shape指针。公开C++shape读取需要与当前引擎合法scene/actor映射桥接，当前代码和已保存raw均未提供这个证据链。本次不实现新桥接、不启动native，也不绕过它。实际cooking设置、live mesh绑定/scale及source原几何证书仍是移动腕逐指恢复的明确阻断。
