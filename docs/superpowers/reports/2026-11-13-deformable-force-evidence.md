# DH116 柔性杆力反馈证据合同

锁定 DeformX `96b8514c…` 的 `RopeBatch.contact_forces()` 读回每个杆节点上所有表面的合力，单位 N。`mesh_wrenches()` 返回杆作用在各个已绑定 mesh 上的合力与绕 mesh 原点的力矩，单位 N 和 N·m。两者均在世界坐标系下，按最近一次 `step_free` 平均。两组 host copy 不含逐接触点、节点与 mesh 的对应关系、法向或法向／切向分量。

`audit_solver_force_row()` 核对 env/episode/reset/rod 身份、SDK 物理时间、rod tick、节点阵列和独立 mesh slot。合力闭合只能核对整步总量。零净力矩无法证明无桌面接触或外部承托，非零节点力也无法指认 partner。R0 接触与滑移、R2 和真实硬件反馈均需各自的独立来源。

重读已保存 Pinch 第 3 步原生账本，节点活跃集合为空，合力残差 0 N，接触对和桌面接触为未知。账本 SHA256 `0c8062301e82b3127b0c49aadc5fdfd0096310f27d00a1f9c9a2f552216a20d1`。复核 JSON SHA256 `7c2678a8ba60c70a6f2e29d3fffd3b71a38c872e918627502f2a2a12a5473423`。本工作包物理步与优化更新均为 0。

配置仍保留 Pinch、Tripod、Power 三个独立 checkpoint namespace。`native_ready=false`，分别依赖同杆双手接触、真实 Reset/371D 与 R0。Tripod/Power 尚无各自已验证的 cooked-hull 起始姿态。Stereo 同钟与 Teacher 身份回归通过，未进行新采集。

CPU 回归 121 passed。测试先因缺少审计模块失败，再通过，包括拒绝时间错位、重复 mesh slot、错配的 active-node ID、非有限数与伪造 contact pairs。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_deformable_force_evidence.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_task_budget.py tests/test_dh116_deformable_privileged_teacher.py tests/test_dh116_deformable_privileged_ppo.py tests/test_dh116_stereo_runtime_rod_identity.py tests/test_dh116_stereo_runtime_teacher_identity.py
~~~

在新原生配方启用完整接触对合同前，需要同钟记录逐对力、作用点、partner、法向和作用侧。PhysX 的手—桌接触对与 DeformX 的杆—网格接触对需要分别取得来源，聚合数据不填补其中任一侧。
