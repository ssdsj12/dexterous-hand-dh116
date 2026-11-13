# Pinch v2：实际 link pose 下的单手初始几何与磁盘重算

基线 `e523957dca2ee826b18648055d777e11249447f1`。仅新增 `tabletop_pinch_initial_geometry_v2.py`、`test_pinch_initial_geometry_v2.py` 和本报告；未修改旧配置、physics、recipe、registry、README 或原生运行器。**24 新项 + 65 相关项 = 89 passed in 126.26s**。本工作没有运行 Isaac、DeformX、GPU、物理 step 或训练更新。

## 调用接口

`capture_pinch_initial_geometry(stage, hand, geometry, *, cooking_interface, read_clock, table_center_m, table_size_m, rod_nodes_w, rod_radius_m, minimum_table_gap_m, minimum_rod_gap_m)`

`geometry` 使用现有 `scripts.audit_dh116_pinch_initial_pose.load_geometry(project, side=...)` 的完整结果。函数重新通过同一 source/F4 lock loader 验证输入，核对 17 源 envelope、5 锁定 hull、geometry SHA 和 mesh-to-body offset。`read_clock` 返回调用方从真实 SDK 读取的 `sdk_step/sdk_time_s`；本模块前后核对完全相同，并核对 480Hz 时间关系，不伪造 solver clock。

读取并拥有 `hand.root_physx_view.get_link_transforms()` 的 XYZ/XYZW 副本、body names、actual link paths 和 articulation path。以实际 body pose 映射几何，不使用 USD world pose 或 FK 代替运行时姿态。逐一检查 17 source body、源/live mesh hash、collision owner/mesh 路径和 mesh-to-body offset，并要求 actual stage 中启用的手 collider 集合正好是这 17 个。无 stage authoring、prim cloning 或 stepping。

桌面 center/size 和初始 rod nodes/radius 是调用方显式输入；本模块没有取得 native table shape 或 solver nodes。返回明确写为 `caller_supplied_initial_geometry_not_solver_readback`。调用方须将这些参数与自己的真实场景构造、运行身份及 recipe 来源绑定。此函数可以在 rod 构造前运行。

## 三组证据及门槛

| 输出 | 内容 | 用途 |
| --- | --- | --- |
| `source_colliders`（17） | 全部原始 collider 的 convex envelope、实际 body pose/path/index、mesh offset、world vertices、table SAT 和连续 rod capsule 间隙 | 初始 source 几何门 |
| `locked_pads`（5） | 锁定 F4 cooked hull，使用同一实际 body pose；保留 capture SHA 和原 mesh/body coincident-frame 校验 | 独立 locked-pad 几何门 |
| `stored_cooking`（17） | 注入的已有 cooking interface 的纯 getter 结果、原始 hull、拓扑校验、query paths/counts、payload SHA 和 metadata；缺失/空结果/异常均保留原因 | 补充存储表示证据，**不作为 live PxShape attachment 或接触证明** |

已有 `read_native_pad_hulls` 同时触发 `request_convex_collision_representation`，因此本包不调用它。这里只读 `get_nb_convex_mesh_data/get_convex_mesh_data`：先查询实际 CollisionAPI owner，只有不可用/空结果才查询实际 mesh。原 `convexHull` 必须返回一个有效 closed hull；不 recook、不 force-load、不以 source path 替代 actual path。

已有 getter 返回的坐标仍标作 `mesh_local_hypothesis_not_live_shape_attestation`，其派生 world vertices/间隙仅属于该假设，不进入初始几何门。`stored_cooking_complete` 只表示 17 条 getter/拓扑记录完整，不能表示 17 个 live actor shape 已证实。

`initial_geometry_passed` 要求 source17 **和** locked5 都满足显式传入的 table/rod gap 门槛。桌面检查复用有限 AABB/convex SAT，rod 检查复用连续 segment-to-convex solid 距离，不只采样 rod nodes。距离求解不收敛即失败，不填零或改用粗采样。这个结果只覆盖离散初始姿态，不证明后续 closure/approach、材料锚、法向、动态接触或抓取。

以下字段恒为 false：`native_shape_attachment_proven/contact_free_proven/native_qualification/teacher_admission`。没有把几何无穿透或零净 wrench 称作完整接触已证实为空；真实 quiet/contact report 门由运行器另行负责。

## 磁盘验证与释放边界

`validate_pinch_initial_geometry(record, geometry, *, table_center_m, table_size_m, rod_nodes_w, rod_radius_m, minimum_table_gap_m, minimum_rod_gap_m)` 接收从磁盘重新读取的 JSON，返回 `{valid, initial_geometry_passed, reasons, native_qualification:false}`。

验证器重新加载源/锁定几何，并从 record 保存的 actual link poses/names/paths 重算全部 source17 和 locked5 的矩阵、world vertices、间隙及 pass/reason。它核对 caller layout/threshold、source SHA、SDK clock，以及本模块和三个复用几何模块的代码 SHA；不能只篡改 producer 的通过字段。stored cooking 的原始 payload SHA、拓扑、查询路径、选择顺序和假设下的派生几何也被复核；NA 保持 NA。

磁盘重算仅证明已保存证据内部一致，不能独立证明这些 pose 真由 SDK 产生。原生 post-exit 审计仍须将该文件与同一进程、源码、实际 ledger 时钟和外层退出回执绑定。

capture 的异常 wrapper 先复制 traceback 文本、清理已结束失败 frame，再在 finally 清空 stage/hand/geometry/cooking/read_clock 引用；完全离开 except 后只抛纯 `RuntimeError from None`。返回值为 owned JSON 值。测试保留失败异常对象时，stage、hand、view、cooking interface 和 capturing clock callback 的弱引用仍全部释放。这里不关闭或释放调用方仍持有的 SDK owner。

## CPU 结果与局限

24 新测试包含全 17 实际名字/路径重排、5 锁定 hull 独立门、proximal table overlap、连续 rod edge 穿越、纯 getter 的四种 NA、源/mesh offset/额外 collider 拒绝、时钟变化、旋转和显式阈值消费、异常弱引用释放，以及磁盘重算中的 pose/path/source/vertices/gate/threshold/stored-payload 篡改拒绝。测试使用真实原始 USD/F4 文件和明确的 SDK/cooking doubles；用于验证两个几何门独立的异常 cooked hull 是标注的 CPU counterfactual，不是新 F4 证据。

修正候选的 CPU FK surrogate 也经过 capture→JSON→独立 validator。它使用 `flexible_pinch_pose_candidate_v2.json` 的原 q/base pose；FK pose 仅作 CPU 测试输入，**不是本次实际 SDK 测量**：

| 几何 | 最小桌面间隙 m | 最小杆间隙 m |
| --- | ---: | ---: |
| source17 | 0.0020008968102463998 | 0.001204678163213205 |
| locked5 | 0.0029878023113341645 | 0.0013156433836962137 |

最终命令：

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q --capture=tee-sys tests/test_pinch_initial_geometry_v2.py tests/test_tripod_cooked_geometry.py tests/test_dh116_pinch_initial_table_clearance.py tests/test_dh116_cooked_convex_hulls.py tests/test_dh116_cooked_hull_device_parity.py tests/test_dh116_usd_surface.py`

**89 passed，0 failed**。额外 import probe 确认 `ropesim/warp/isaaclab.sim/omni.kit.app/omni.physx` 均未加载；本包没有创建 app 或 solver。

证据目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-initial-geometry-v2-cpu-1udbdwd3`。

| 文件 | SHA256 |
| --- | --- |
| `final-cpu.log` | `7465eb17dc4a1807e79083baeeaf421baf9e49861efdb2489a949013757d3769` |
| `final-cpu-manifest.json` | `d9a34bab238862709af95a0d450b57d71e5214c56c2e0583967743eea62a244a` |
| `import-boundary.log` | `b2391aca4a4af382a24eebca1f363cb6ddc11da9fb2c3423fb0ceb320bc241c3` |
| `tabletop_pinch_initial_geometry_v2.py` | `3c272069c43eaed71f11c17d73af28f810f39762cea792771a83647c04924bd9` |
| `test_pinch_initial_geometry_v2.py` | `d0a4341190379e9a2b2c7af06c901fd715c707b2d5dbde4301acbc83e93ec5f6` |

没有新 recipe 或原生许可；运行器、独立 post-exit 调度资格链和新的冻结审批由主集成工作包负责。
