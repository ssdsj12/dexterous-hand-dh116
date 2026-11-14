# eb31c79 批次执行、预算修复与恢复点

审查基线 `eb31c793d60cb50eb8d6dd2e06bae2f15c5bd8d5`。开发和所有 Python/原生运行均在 Ubuntu 独立 worktree；同一 commit 经 Windows 导入后推送。最终代码验证点 `d266d021f7e75265e100352f6ab906314fe6fc94`，后续本报告及发布 manifest 只增加证据元数据。

## 实际完成与失败边界

| 工作包 | 实际结果 | 资格边界 |
|---|---|---|
| 左/右独立配方与回执 | 分别核验原始 USD、SDF、link、canonical index 1/7；右侧用本次左真实回执及 SHA 自动串行启动 | 未复制左侧身份；恢复没有重复启动已完成尝试 |
| 左拇指 v2 | 240 测量行，68 接触行，峰值 0.047305321 N，GDB 与进程自然退出 0 | 零重力、杆两端固定，只有接触/生命周期技术证据 |
| 右拇指 | 240 测量行，68 接触行，峰值 0.043695419 N，GDB 与进程自然退出 0 | 同上；不是双手同时作用同一杆 |
| 本批预算审计 | SDK reset 每侧还推进 2 个实际 PhysX 步，实际总额 **484**，超批准 480 共 **4** | **本批预算不合规，完整资格已撤回；原始数据及回执保留** |
| 预算修复 | 新配方每侧 2 setup+238 measured=240 total；独立校验实际 SDK 计数；清理前冻结物理并检查末时钟 | 仅 CPU 通过；新指纹没有本轮原生授权，未重跑 |
| 相机 C0/C1/C2 | 三次都 `app.close()` 返回后 SIGSEGV；当前首次栈均在 AppWindow 同一偏移 `0x15e55`，保留动态库映射 | 全部原生退出失败；C3/单相机/RGB/任务双目没有新增资格 |
| WireSeg fixed16 | 分割和 depth-only 各 128 个有界诊断更新；独立 held-out family、全场与 ROI 指标、梯度和失败记录 | 深度不及常数基线，不能解释为感知或泛化通过 |
| 数据续传 | 第三个 RGB Parquet 完整 LFS SHA 核验；6/148 blobs，新增 800 条仅属 test family，无精确配对深度 | 不把 test 作为 train、不伪造配对，不等待全量才允许 verified subset |
| Teacher/Student | 本批 0 PPO 更新、0 BC/DAgger 训练 | 单手/双手自由杆接触、Reset、状态动作、R0 尚未全通过，R2 仍关闭 |

左 v2 物理账本与此前左 v1 的 240 行字节完全相同；本次改变了退出结果而未改变这段物理记录。之后发现的初始化预算缺口单独记录，没有改写这一比较。相机 C0 SDK 默认 experience 和 C1/C2 stage 单变量对照均直接采集本次栈，没有只引用旧 F6 结论。

## 代码与证据

- 调度回执校验、依赖实际路径/SHA、同次 GDB：`scripts/run_dh116_work_packages.py`。
- 左右动作身份、2+238 总计数、收尾冻结：`scripts/verify_dh116_deformx_thumb_contact.py`、`tabletop_deformable_rod_contact.py`、两份 thumb lifecycle recipe。
- 相机当前对照：`scripts/probe_dh116_simulation_app_lifecycle.py` 与 C0–C3 独立配置；本批仅执行 C0–C2。
- 固定16诊断：`scripts/diagnose_dh116_wireseg_overfit.py`，复用原 subset/resume/模型模块。
- Teacher/评估预算 v2：`tabletop_deformable_task_budget.py`、`configs/dh116_training/deformable_task_budget_examples_v2.json`。v1 保留 phase-only 读取；任何版本 setup 未知都不能通过冻结评估预算 scope。
- 未改原 Teacher/Student/PPO/MPC 控制、原始 USD/材料/关节保护、371D/3844D/24D 合同及 30 mm/5 s/0.5 mm 标准。

对应详细报告：

1. [左右原生账本与484步偏差](2026-11-12-deformx-thumb-v2-paired-native.md)
2. [总物理步与关闭保护修复](2026-11-12-deformx-thumb-total-physics-budget.md)
3. [当前C0–C2栈与模块映射](2026-11-12-camera-c0-c2-current-native.md)
4. [固定16张学习与持出误差](2026-11-12-wireseg-fixed16-overfit-diagnostic.md)
5. [第六个完整数据文件与测试家族隔离](2026-11-12-wireseg-sixth-blob-verified.md)
6. [Teacher/完整回合预算v2](2026-11-12-deformable-task-budget-v2.md)

## 最终集成验证

Ubuntu `d266d021f7e75265e100352f6ab906314fe6fc94` 执行下列 CPU 回归，**127 passed，0 failed，0 skipped，20.99 s**。这些 CPU 测试不计原生验证次数。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
 tests/test_dh116_work_packages.py \
 tests/test_dh116_deformx_thumb_contact.py \
 tests/test_dh116_deformx_bridge_lifecycle.py \
 tests/test_dh116_simulation_app_lifecycle.py \
 tests/test_dh116_render_capture_boundary.py \
 tests/test_dh116_deformable_task_budget.py \
 tests/test_dh116_wireseg_overfit_diagnostic.py \
 tests/test_dh116_wireseg_blob_relay.py \
 tests/test_dh116_wireseg_pretraining.py
```

完整 CPU 日志留在私有证据目录，SHA256 `424fb037ce1320fa1d919caa0e25118a1665244eabc4aa41de8077ea2aaeaf98`。528 个受保护文件与 2 个历史 Checkpoint SHA 已重新核对。原生后未遗留本批 Isaac/GDB/调度运行进程，GPU3 回到 7 MiB、0% 利用率。没有通过 SIGTERM/SIGKILL 或 `os._exit` 伪造退出。

## 恢复与后续门槛

私有恢复点：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/eb31c79-budget-ended-z_4fzexl/recovery.json`，SHA256 `7b0d7b0daeb407e684116be3d295be7d1e78153b50dd6434c38c75f280bc0f0e`。它明确 `EXHAUSTED`，不是新批准文件。

当前旧批次审批在修复后的源码上得到 `native_ready=[]`、无待发表原生结果；左侧等待新审批，右侧等待新版左侧回执。CPU 自动队列也没有未执行的就绪项目。所有历史回执保持原始字节；registry 和 readiness 中左右的预算合规资格均为 false。

新配方供下一次独立批准审查，本轮没有执行：

| 项目 | 左 | 右 |
|---|---|---|
| command spec SHA256 | `196eee54825ef8b9a2f4f3878dd34cd978b0f22bdf78f975367995e5c583f1cb` | `03137ad9ebbd0cefc729b18e363944b0f180ecb7a7c04971c758e769e1aaacdb` |
| code/config SHA256 | `8cc880d43d5b750bf226e2b14f231d98d2456b11c9b6ad1dece9e5d8f1dc7137` | `3983b99e2a4efb6e26756bb85e5168d8a77730cbe99a7a8f20d305a4b952889b` |
| 环境 / GPU / 更新 | 1 / 3 / 0 | 1 / 3 / 0 |
| 控制与物理账本 | 直接标定循环，无PPO rollout；2初始化+238测量，240总物理步 | 同左，且必须使用新左真实回执 |
| 最大墙钟 | 180 s | 180 s |

单手多指、双手同一自由杆、真实重力和无额外支撑审计仍是独立任务，不能继承本次固定端零g标定配置。Teacher 示例 2 env×32 control×8 的 512 行和冻结评估示例的 2880 行仅计测量阶段，setup/总额未知时保留 null；5秒Hold本身每环境至少2400步，60步接近示例的可达性尚未验证。Pinch/Tripod/Power 保留独立后继，MPC BC只是有合格同步数据时的初始化选项，WireSeg和相机失败不会成为无相机Teacher的永久前置。

后续批准后先验证总步数修正版，再推进单手/双手真实自由杆及F7物理状态合同。相机C3与对象/回调所有权排查独立推进；未通过自身同步与自然退出前不能采集正式任务视觉。未自动追加预算或启动长训练。
