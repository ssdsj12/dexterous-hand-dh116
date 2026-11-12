# DH116 长时续任务第二批：实际进展、失败证据与恢复点

本批从 Ubuntu integration、Windows 发布缓存和 GitHub 一致的 `b2de6ec4a0e1879b55052b84e0a2a412d7a0cf7c` 继续。全文读取 `DH116_Long_Running_Continuation_Task.md`，SHA256 `e4f93658ea0c4c90324e7cde7e54a0dc104565f1bf9cbf4082ef9405c7f16802`。旧 P0/F0–F4 通过证据未重做；本报告不代表整个工程或 Teacher 完成。

## 环境、发布与保护

- 开发/运行：Ubuntu `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/integration`，目标分支 `codex/dh116-stereo-privileged-dagger`。
- 发布：Windows bare relay 导入完全相同 Git commit/tree，校验文件清单后正常 push、fetch、核对远端 HEAD；未从 Ubuntu push。
- 原 `/mnt/mydisk/xk/dh116` 保持其 `11-3` / `c849cfeb7aae277d8ed56c50f2dce8fb41e48e40` 及已有未提交修改，未覆盖。对齐的是独立 integration 工作树，不宣称原项目根仓库已切换。
- `11-10` 保持 `6cee7418a5e49f56b7f0090fc5eaf20b143c0b1b`。PR #2 保持 open Draft，未 merge。
- 528 个保护文件及两份历史 Checkpoint SHA 全部吻合；未上传原始数据、模型、CUDA .so 或大型日志。数据和运行证据留在私有 Ubuntu evidence 目录。
- 最近资源读回 GPU3 RTX4090：7MiB used / 24075MiB free，0%利用率；本批没有新 Isaac 进程。

## 本批独立工作包

| 发布提交 | 实际改动 | 本包 CPU 回归 | 原生/训练边界 |
|---|---|---:|---|
| `dc7faa01b0b932030f375e3720fb38ef04c5c480` | C5 telemetry holder 所有权单变量探针、GDB finalizer/fatal 栈与动态库映射 | 44 | 初始版本未执行，随后被修正版取代 |
| `e0262caff8baf58b7a23f955c000a0850d4be5aa` | 旧 WireSeg 诊断导入、可继续保存 model/Adam/RNG/sampler、官方 depth21 入库 | 62 | 导入与 CPU 测试不等于新训练 |
| `4db819e96fb37d69c7bbe44a67eb67a5d457f03e` | C5 worker 子进程实际重入专属入口，修复父进程设置未传播的问题 | 45 | 原生预算尚待该新指纹批准 |
| `4d9807e5fe7bb5c93515fe773ca02ade7378286a` | 冻结模型 4 train+4 heldout 深度误差只读归因 | 67 | 32条双模型/双视图记录，外层7.31s、exit0，无优化 |
| `e6caa7280e6b4ded01d33122d62108e6acfb26ae` | 自由杆左手 Pinch 实际入口、5个指腹SDF、反作用力、专属worker及退出后独立账本审计 | 156 | 未运行原生，不授予抓取/371D/R0资格 |
| `a04fdae8415c477f287c46d861a62a655634b959` | 固定16训练样本/完整128验证样本误差域复核、可恢复缓存及完整实现身份绑定 | 77 | 288视图，复用16；新增272推理，25.92s、exit0，无训练 |

独立包计数存在重叠，不相加作为总测试数量。最终26个测试文件按隔离进程运行，共458 passed、0 failed、0 skipped。

## P0–P6 当前状态

| 工作包 | 代码 | CPU | 原生 | 本批训练 | 当前状态 / 下一动作 |
|---|---|---|---|---:|---|
| P0 左右拇指预算复验 | 保留已通过版本 | 回归通过 | 沿用左240+右240，各自然exit0 | 0 | native_passed，仅零重力双端固定单指接触标定；不重复执行 |
| P1 单指/单手/双手同杆 | 新自由杆单手候选入口已实现 | 156通过，整批覆盖 | 新候选未执行 | 0 | cpu_verified；等待240步单次左手候选对应授权；右手/双手后继仍待证据 |
| P2 WireSeg | 导入/恢复/损失域审计/缓存身份修复 | 77通过，整批覆盖 | 官方单图数据，不是原生Stereo | 0 | 深度仍不合格；先解决训练有效性，不盲目追加更新 |
| P3 Stereo 相机 | C5最小所有权探针与真实worker修复 | 45通过，整批覆盖 | 既有C3自然0；C4 SIG11/外层255；C5未执行 | 0 | 等待新指纹批准；不阻塞无相机物理 |
| P4 Reset/R0 | 复用独立柔性任务/预算/Reset/R0合同 | 回归通过 | 完整真实371D、材料锚滑移、同杆双手闭环未齐 | 0 | 不能把CPU字段或聚合wrench当成完整原生资格 |
| P5 Pinch Teacher | 独立Actor/Critic/Adam/结构化特权分支准备保留 | 回归通过 | 未开始柔性Teacher原生更新 | 0 | 等Pinch接触+Reset+状态动作+R0；不依赖WireSeg完成 |
| P5 Tripod Teacher | 独立模型/配置准备保留 | 回归通过 | 未开始 | 0 | 该模式接触资格待完成；不与Pinch混记成功 |
| P5 Power Teacher | 独立模型/配置准备保留 | 回归通过 | 未开始 | 0 | 该模式接触资格待完成 |
| P6 Student | 保留既有BC/DAgger/PPO与部署门控 | 合同回归通过 | 无新独立验收 | 0 | 合格Teacher与同步双目缺失，正式训练blocked |
| GitHub发布 | 每包独立commit，Windows relay | 树/文件清单核对 | PR2 Draft | — | Release Manifest记录已核对提交；本报告随后同流程发布 |

## 真实结果与限制

### 自由杆入口

新入口使用真实 RopeBatch、自由端、-9.81m/s²重力、有限桌面与五个真实link/pad SDF，记录实际节点、q/qd、目标、wrench与SDK/rod时钟，反作用力转换保留作用点。所有启动步包含在240预算内。独立进程退出后重读JSONL/退出码/自然关闭/节点逆质量/各pad身份才能写合格回执；通用调度入口不能绕过。

这是单手接触候选，不是自由双手抓取资格：候选材料ρ=0.065kg/m、EA=800N、EI=0.018N·m²、半径5mm来自原P0系数，并非实测材料。固定手基座、逐材料锚滑移和逐接触法向尚不满足正式抓取条件。原环境、控制限制、关节保护未改变；R2/Force Proxy保持关闭。

### WireSeg

官方完整blob增至9/148；新增depth21与既有RGB精确配对360样本，718个instance，中心192裁剪有监督294/无监督66。Windows下载/Ubuntu再次SHA核验后独立入库，未覆盖旧manifest。官方单图depth不作为Stereo disparity。

原train manifest含128行，本次训练合同固定前16；validation完整128。两个视图合计288记录，16条从先前审计精确复用，272条新增CPU推理，权重保持不变。训练中心有效深度仅4.309%，全部在线缆ROI；持出中心100%有标签，其中96.17%为背景。

持出中心线缆ROI MAE=0.141680m，训练常数基线=0.087620m；背景MAE=0.138472m，常数=0.078238m。全部128个持出中心样本均差于常数。域差不能单独解释线缆ROI也差的结果；相关性不是唯一因果证据。完整视场缩小到192后的预测范围压缩，不外推成所有center视图均压缩。

pooled P95不能从缓存单图P95重建，保持null并保留每图P95。MAE/RMSE由计数和误差矩精确汇总；没有平均百分位伪造群体P95。下一步先明确线缆/背景监督域及优化有效性，不能把验证集移到训练集或靠更换分母掩盖误差。

## 失败与修复

1. 旧C4仍是渲染路径app.close之后 `_extensions+0x1e445` 的SIG11，外层255；与C0–C2原AppWindow栈不混为同一根因。C5清理telemetry holder只是待验证假设，无新原生结论。
2. C5父进程局部替换未传入worker：在执行前发现，通过真实子进程CPU回归修复；原未执行代码指纹废弃。
3. Pinch通用worker可能绕过独立后验审计：增加wrapper来源、候选SHA、输出绑定；CLI在Isaac导入之前拒绝错误入口。
4. Pinch首次集成裸Python未加载pxr，收集exit2；使用现有scripts/python.sh，无安装或升级。新增registry依赖scope类型错误被发布校验阻止，已修为现有scope/native对象。失败事务保留。
5. 首次整批单进程457 passed/1 failed：USD材料测试在收集时导入pxr，触发纯CPU合同无模拟器依赖断言。将三份纯合同测试与其余USD/运行时测试分进程，所有458项与原断言保留，未跳过、未改受保护测试。
6. WireSeg缓存漏绑实际depth helper：审查中复现后修复，helper/聚合函数/脚本SHA全部绑定。原run01不改写，独立事后核验证明现有数据与实现相符，不能冒充运行时原本记录过helper字段。
7. Ubuntu官方HF下载遭TLS hostname mismatch；失败日志保留，改由Windows正常验证证书下载、Ubuntu二次LFS SHA入库，未禁用TLS验证。

## 新批预算与授权

已批准的P0修正版480步和C3/C4两格均已消费，不重用。当前没有新批准的原生预算余额；下列仅是已冻结并向用户提出的请求，不能自动执行：

| 候选 | GPU/环境/更新 | 最大SDK物理步 | 最大墙钟/尝试 |
|---|---|---:|---|
| C5 telemetry所有权 | GPU3 / 0物理环境 / 0更新 | 0（最多1次显式App update） | 180s / 1次，不采集、不重试 |
| 左手自由杆Pinch接触 | GPU3 / 1环境 / 0更新 | 240=2初始化+238测量 | 180s / 1次，失败保留账本和栈 |

两个任务只共享单实例资源锁，没有互相的资格依赖；批准后按锁串行。某项失败只阻断自身后继。同版本不得重跑或强制退出制造成功。Teacher smoke、完整episode、评估和训练各另建预算；5s Hold至少2400个480Hz物理间隔，不能以本接触候选代替。

## 恢复与实际运行

恢复依据：`docs/contracts/dh116_continuation_batch2_recovery.json`、工作包Registry、Release Manifest、私有native receipts。现有完成回执按SHA复用，不重新执行。新的exact-native许可未到时，调度器没有native_ready项；P5/P6未越过物理/视觉资格门槛。

本批新原生执行0，SDK物理步0，rod步0，PPO更新0，真实模型优化更新0，完整episode0；因此本批原生成功率不适用。CPU模型/恢复测试不计正式训练，CPU夹具不计原生验证。

实际冻结模型审计命令见对应两个WireSeg报告；下面列出最终两进程回归的完整命令及证据。

### pure-contracts

Result: 80 passed / 0 failed / 0 skipped; outer exit 0.
Log SHA256: ce6ce80ed9a692aecfd09f781f85620f46ff28bf1b74be43caf52674ff6bd590

The existing scripts/python.sh launcher supplied USD/Python paths to each separate pytest subprocess:

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_stereo_action_contract_audit.py tests/test_stereo_privileged_state_contract_audit.py tests/test_privileged_teacher_reward_audit.py
~~~

### usd-runtime-and-models

Result: 378 passed / 0 failed / 0 skipped; outer exit 0.
Log SHA256: 3eef83314cd3b5357bb51f0ef3851579d6445e56bd9ca7669b5f082de121b2aa

The existing scripts/python.sh launcher supplied USD/Python paths to each separate pytest subprocess:

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_work_packages.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_task_budget.py tests/test_dh116_deformx_bridge_lifecycle.py tests/test_lift_hold_window.py tests/test_dh116_stereo_runtime_rod_identity.py tests/test_dh116_wireseg_checkpoint_resume.py tests/test_dh116_wireseg_overfit_diagnostic.py tests/test_dh116_wireseg_pretraining.py tests/test_dh116_wireseg_blob_relay.py tests/test_dh116_simulation_app_lifecycle.py tests/test_dh116_render_capture_boundary.py tests/test_dh116_deformable_privileged_teacher.py tests/test_dh116_privileged_teacher.py tests/test_dh116_deformable_privileged_ppo.py tests/test_dh116_camera_c5_ownership.py tests/test_dh116_wireseg_legacy_import.py tests/test_dh116_wireseg_depth_error_audit.py tests/test_dh116_wireseg_loss_domains.py tests/test_dh116_pinch_scheduler.py tests/test_deformable_grasp_runtime.py tests/test_dh116_contact_material.py
~~~

Aggregate elapsed: 47.17s. Evidence SHA256: de998f2e45c27d8225343d57dfb8866fb357d4a0a7e8e9ef31daa0553178f55b
Red aggregate log SHA256: 725dbb836670137e73e29f0acefcb7259ab03309179c44e4aa60cb7536225a81

## 冻结身份、数据与模型

~~~json
{
  "version": 1,
  "base_head": "b2de6ec4a0e1879b55052b84e0a2a412d7a0cf7c",
  "tested_head": "1988eeb76349c0b68d90acfb1df2178c0f1ce248",
  "specification_sha256": "e4f93658ea0c4c90324e7cde7e54a0dc104565f1bf9cbf4082ef9405c7f16802",
  "branch": "codex/dh116-stereo-privileged-dagger",
  "publication_route": "ubuntu_commit_windows_relay_only",
  "qualification": "batch_checkpoint_only_not_training_qualification",
  "cpu_evidence_sha256": "de998f2e45c27d8225343d57dfb8866fb357d4a0a7e8e9ef31daa0553178f55b",
  "actual": {
    "native_runs": 0,
    "physics_steps": 0,
    "rod_steps": 0,
    "ppo_updates": 0,
    "training_optimizer_updates": 0,
    "complete_episodes": 0,
    "native_success_rate": null
  },
  "requested_native_batches": {
    "flex.f6-camera-c5": {
      "command_spec_sha256": "b2111b376320ba4dfb4babd302034e77a0de12678f080d5386903ff335855322",
      "code_config_sha256": "c9a7e7ec6effcb72a59b97b22138dd568a31579e057d18a8258be2451f4f42aa",
      "budget": {
        "environments": 1,
        "gpu_index": 3,
        "min_disk_free_bytes": 1073741824,
        "min_gpu_free_mib": 4096,
        "physics_steps": 0,
        "timeout_seconds": 180,
        "updates": 0
      },
      "approved": false,
      "execution_count_this_batch": 0,
      "entrypoint": "scripts/run_dh116_camera_c5_work_packages.py"
    },
    "flex.f4-free-pinch-left-candidate-native": {
      "command_spec_sha256": "49084203742e4ff3487eeb1305fde2a342204edc06ecd34d7b4157bec38685d4",
      "code_config_sha256": "d0ec5e2f6146c82e637331acd8dac173fd46cc80425e769578d36e7ce98273de",
      "budget": {
        "environments": 1,
        "gpu_index": 3,
        "min_disk_free_bytes": 1073741824,
        "min_gpu_free_mib": 4096,
        "physics_steps": 240,
        "timeout_seconds": 180,
        "updates": 0
      },
      "approved": false,
      "execution_count_this_batch": 0,
      "entrypoint": "scripts/run_dh116_pinch_work_packages.py"
    }
  },
  "completed_native_receipts_reuse": {
    "flex.f4-left-contact-native": {
      "result_sha256": "460ce27e089ae3178b912dd22e72286122c4eee4d11769b154ef27051d0dd6ca",
      "outer_exit_code": 0,
      "native_qualification": true,
      "physics_budget": {
        "initialization_physics_steps": 2,
        "warmup_physics_steps": 0,
        "measurement_physics_steps": 238,
        "other_physics_advancing_steps": 0,
        "total_physics_steps": 240
      },
      "qualification_scope": "dh116_left_thumb_deformx_contact_lifecycle"
    },
    "flex.f4-right-contact-native": {
      "result_sha256": "8007694e7b4424de4743f6ccd6660c85479b3540506647db725a34fa246a179f",
      "outer_exit_code": 0,
      "native_qualification": true,
      "physics_budget": {
        "initialization_physics_steps": 2,
        "warmup_physics_steps": 0,
        "measurement_physics_steps": 238,
        "other_physics_advancing_steps": 0,
        "total_physics_steps": 240
      },
      "qualification_scope": "dh116_right_thumb_deformx_contact_lifecycle"
    },
    "flex.f6-camera-c3": {
      "result_sha256": "c0c48203d13e7de5416fca42d9a8d40096c588d8ca9eef5aa858f733c4a24f42",
      "outer_exit_code": 0,
      "native_qualification": true,
      "physics_budget": null,
      "qualification_scope": "camera_c3_isolated_lifecycle"
    },
    "flex.f6-camera-c4": {
      "result_sha256": "7ef892c4dc65ea85168e167b023ec5d07c496f7aed12f99911d45238de5c05d6",
      "outer_exit_code": 255,
      "native_qualification": false,
      "physics_budget": null,
      "qualification_scope": "camera_c4_isolated_lifecycle"
    }
  },
  "protected_file_count": 528,
  "protected_checkpoints": {
    "/tmp/dh116-wp3-teacher128-v02/teacher_diagnostic/teacher_update_000000.pt": "ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511",
    "/tmp/dh116-wp3-teacher128-v02/teacher_diagnostic/teacher_update_000001.pt": "c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999"
  },
  "data": {
    "wireseg_complete_blobs": 9,
    "catalog_total_blobs": 148,
    "depth21_lfs_sha256": "eeddfc47beff6abd0f92f7e3bb9da64fe0d9c734dfa9b4e3d39cc760ebcf7c08",
    "subset_manifest_sha256": "84ce17fc28c99b46762d414003fe8b7b57a5351760c84d8526344ce24fbf60f8",
    "full_audit_result_sha256": "c639628246161ba4cbd2ae48b69a083cdb1bfd6a3f9bee8f7eea01c35791cf48",
    "cache_identity_proof_sha256": "2583d25ab88677d270af129f39c36a56d5d8aa224712728bb5d377eea1504ded"
  },
  "preserved_model": {
    "original_seg_sha256": "77459b47f26ccfdeea024f298fe59bedbb9c119242cddda305aeea0b71cf5f39",
    "original_depth_sha256": "56826811b64075446486fc0309a60a616dabcafc738d40700be04ec39ac21adb",
    "imported_depth_sha256": "1d22ce17a31177af013c0b945a042de9cc04792549a99bcbe12c477711fd5a30"
  },
  "next": [
    "C5 or free-Pinch exact approval then single-instance bounded execution, independently validate and publish",
    "WireSeg wire/background objective and optimization-effective diagnostic before any new training budget",
    "P4 true full-state/reset/R0/material-anchor bindings after physical evidence",
    "Independent Pinch/Tripod/Power Teacher batches only after own gates",
    "Student only after qualified teacher and synchronized stereo"
  ],
  "r2_enabled": false,
  "formal_teacher_started": false,
  "formal_student_started": false,
  "original_project_preserved": true
}
~~~

Private recovery: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/continuation-round2-recovery-mbwl9qrq
Recovery SHA256: cef8f7d2fb906c2938196a0e24f4c8663fddde6b0c16647fa18aeac47efee2b5
Queue SHA256: 854262f1ed45041b2e93b4f98c15c2ae8c7ab5362492ddd7c93078579b40197e

整批状态：工作包已测试并保存恢复点；工程未完成。后续批准不沿用旧指纹，不自动扩大任何原生或训练预算。
