# DH116 柔性抓取长时续任务批次结项

续任务基线 `2ef7f90ed9af9039dd32fb197063040ae11996f3`，开发分支 `codex/dh116-stereo-privileged-dagger`。结项证据起始核查树为 `f4fe60f7b2af66e561b3d3718fffdef4a62e0afe`。本轮最终集成与发布核对见下方。

## 工作包结果

| 工作包 | 代码 | CPU | 原生 | 训练 | 当前状态 | 下一动作 |
| --- | --- | --- | --- | --- | --- | --- |
| P0 左右拇指预算复验 | 四阶段物理预算、独立逐行接触验证、右侧绑定新版左回执 | 预算与调度回归 78 项通过 | 左、右各一次合规单拇指标定，外层均自然退出 0 | 0 PPO 更新 | `native_passed`，仅限单拇指标定 | 保留四阶段账本和左右回执，向独立 P1 自由杆任务提供后端与指腹来源证据 |
| P1 单指、单手与双手同杆 | `flexible_dual_hand_grasp_v1.json` 隔离自由杆任务合同，三种指腹组独立配置 | 任务与 R0 集成回归 172 项通过 | 本批只有 P0 两侧单拇指原生记录，尚无自由杆、单手多指或双手同杆结果 | 0 | `blocked`，重力、材料、桌面接触、实际 Reset 和逐指 partner 仍待原生资格 | 先核对真实材料、杆端部与桌面耦合，再准备独立单手 Pinch 接触配方 |
| P2 WireSeg | 严格 v2 断点、游标与旧断点只读审计，新增一组已验 RGB/depth 分片 | 30 项通过，合成 8 次连续和 3＋5 次恢复状态一致 | 无 Isaac 或双目数据采集 | 0 次新真实更新 | 数据入库及严格续训软件 `cpu_verified`，旧第 128 次断点的严格续训 `blocked` | 另行设计保留模型、AdamW 和 Torch RNG 的显式旧断点迁移，或取得完整 v2 源。分别冻结诊断 cohort |
| P3 Stereo 相机 | C3/C4 控制、AppWindow 读回、故障栈与映射审计 | 相机生命周期与采集合同回归 30 项通过 | C3 一次自然退出 0。C4 一次退出失败，inferior SIGSEGV 11 | 0 | C3 `native_passed` 仅限无相机生命周期。任务双目采集 `blocked` | 静态审查 rendering experience 的扩展依赖与对象所有权，再准备有新指纹的单变量对照 |
| P4 Reset/R0 | 源身份逐步合同、回合支撑锁存、真实 5 秒 Hold、固定材料与切向 Slip 参考 | 任务/R0 172、支撑锁存 36、Hold/Slip 51 项集成回归通过（集合重叠） | 未运行真实柔性 Reset、R0 或完整成功判据 | 0 | `flex.f7-physics-state` 为 `blocked` | 绑定真实 rod/手状态、逐指接触与支撑源，验证 Reset 清理和完整物理时钟 |
| P5 Pinch Teacher | 独立 371D common＋材料节点分支、24D Tanh Gaussian，结构化 PPO 和新断点入口 | 模型合同153项，结构化 PPO 含断点完整性修复170项通过；最终合并回归见末尾。 | 未做真实 Pinch Teacher update 或完整回合 | 0 真实更新 | `flex.f8-pinch` 为 `blocked` | Pinch 两侧各两指原生对向接触、Reset/R0 后冻结独立更新配方 |
| P5 Tripod Teacher | 同一实现，独立 Tripod 配置及 Checkpoint 命名空间 | 同一共享回归，结果不作三倍累加 | 未做真实 Tripod Teacher update 或完整回合 | 0 真实更新 | `flex.f8-tripod` 为 `blocked` | 先验证左右各三指可达及材料站位，再按 Tripod 证据绑定 R0 与冻结预算 |
| P5 Power Teacher | 同一实现，独立 Power 配置及 Checkpoint 命名空间 | 同一共享回归，结果不作三倍累加 | 未做真实 Power Teacher update 或完整回合 | 0 真实更新 | `flex.f8-power` 为 `blocked` | 先验证左右各五指安全接触来源，再按 Power 证据绑定 R0 与冻结预算 |
| P6 Student | 保留 Student 执行、无特权输入与独立部署门控，本批无新训练路由 | 既有部署隔离 47 项、CPU 准入 22 项历史通过，本轮未计作新独立测试 | 无同步任务双目证据或 Actor-only 原生评估 | 0 BC、DAgger 或 PPO | `flex.f9-three-students` 为 `blocked` | 仅在合格 Teacher 和同一 pre-action 时钟的双目 RGB/深度/Native ID/物理状态齐备后申请正式训练 |
| GitHub 发布 | Ubuntu 工作树开发，Windows 导入同一 Git commit 后发布和回读 | 见本报告末尾的合并回归 | 按独立回执记录，不作为发布推断 | 0 | 各独立发布由远端清单核对，结项合并回归见末尾 | PR #2 保持 Draft，按现有 Windows 中继核对后续发布 |

CPU 报告覆盖的测试有重叠，表中各包通过数不相加。源提交与命令见各独立报告。P0 预算修复 `0b053b2`、左侧回执 `fa5bbbd`、右侧回执 `22210b4`。P1/P4 的柔性任务、支撑锁存、Hold/Slip 修复分别为 `094baf9`、`fc7430e`、`dd32f1b`。P2 严格续训为 `075db84`。P3 的 C3、C4 回执和栈审计分别为 `27b881e`、`d7f6bb6`、`e1cc651`。P5 独立模型合同为 `dea157f`。P5 结构化实现只通过合成 CPU 更新，使用原 Tanh、GAE、KL、Adam 事务与回滚路径。旧 `tabletop_privileged_ppo.py` SHA-256 仍为 `f0a96f4f8c9548b5153d004013bef9d77b97250aa0bc43272626cffb15416854`，其 Checkpoint 路径保留。Pinch、Tripod 和 Power 分别采用 `deformable_teacher_pinch_v1`、`deformable_teacher_tripod_v1` 和 `deformable_teacher_power_v1`，目前均无真实训练产出的新 Checkpoint。

## 本批实际原生执行

| 执行 | 实际 SDK 步 | 实测 JSONL 行 | 杆子步 | 外层 / GDB inferior | 墙钟 | 判定 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 左拇指 `attempt-_7cva3ff` | 240 | 238 | 2856 | 0 / 0，自然退出 | 32.027 s | 左单拇指预算及接触通过 |
| 右拇指 `attempt-nwy4b_wj` | 240 | 238 | 2856 | 0 / 0，自然退出 | 33.071 s | 右单拇指预算及接触通过，绑定真实左回执 |
| C3 `attempt-b5nskwks` | 0 | 0 | 0 | 0 / 0，自然退出 | 26.843 s | 无相机 lifecycle 通过 |
| C4 `attempt-iqt7vw5q` | 0 | 0 | 0 | 255 / SIGSEGV 11 | 54.807 s | rendering lifecycle 失败 |

4 次实际原生调用，共 **480 SDK 物理步、476 条实测记录和 5,712 个杆子步**。P0 每侧 2 初始化、0 预热、238 测量、0 其他推进，SDK 测量索引为 3 到 240。两侧分别记录 66 个接触步，峰值力为左 0.047305321 N、右 0.043695419 N，记录的最大作用力残差均为 0 N。场景为零重力双端固定的单拇指标定。历史共 484 步的批次仍标记预算不合规，原账本保持可读。

右侧实际启动和完成回执均绑定左侧 SHA-256 `460ce27e089ae3178b912dd22e72286122c4eee4d11769b154ef27051d0dd6ca`，右侧回执 SHA-256 为 `8007694e7b4424de4743f6ccd6660c85479b3540506647db725a34fa246a179f`。两侧均通过实际外层退出、锁、GDB、冻结后的 SDK 时钟、rod、接触及关闭产物检查。C3、C4 均仅调用一次显式 App update，未记录同步任务画面。

C3 保留 GDB 正常退出日志与 `before_app_close.json` 中 104 条过滤库路径，未发生崩溃，因此没有崩溃栈或信号时完整映射。C4 关闭前实测默认 AppWindow 不存在，AppWindow 扩展及 viewport 仍在。首故障为 `_extensions+0x1e445`，后续为 `_carb+0x4213d`、`_carb+0x425de` 和 `Py_FinalizeEx`。完整动态库和信号时映射来自同次 GDB，独立 live proc maps 的模块区间也匹配。C1 历史首帧为 `_appwindow+0x15e55`，无法据此将 C4 定为同一具体回调。C4 原生退出失败只阻塞任务相机路径。

**真实 PPO 更新 0，真实训练 0，完整抓取评估回合 0，抓取成功率未产生。** 本批 3 个原生资格结果和 1 个原生退出失败都属于诊断范围。两批批准额度已用完，同指纹无重复尝试。GPU 3 在收尾读回约 7 MiB、0% 利用率。

## 冻结命令与证据位置

调度器实际使用 `bash scripts/python.sh`，解释器为 `/mnt/mydisk/xk/envs/dh116/bin/python`。调度器在物理 GPU 3 上给子进程设置可见设备，探针内使用 `cuda:0`。按私有 `launch_argv.json` 复核了下列命令入口。

| 执行 | 实际入口与预算 | 私有记录位置 |
| --- | --- | --- |
| P0 左、右 | `scripts/verify_dh116_deformx_thumb_contact.py --headless --device cuda:0 --side left\|right --steps 240 ... --execute-native`。右实际额外传入左 `--left-scheduler-receipt` 和 `--left-receipt-sha256`。原 solver、SDF、capture 和独立输出参数见实际 argv | `continuation-p0-approved-93c17beb/thumb/flex.f4-{left,right}-contact-native/attempt-*/native` |
| C3、C4 | `scripts/probe_dh116_simulation_app_lifecycle.py --case C3\|C4 --output <独立目录> --execute-native`。1 次 update，0 物理步 | `continuation-camera-approved-3vbcw13a/camera/flex.f6-camera-c{3,4}/attempt-*/native` |

| 执行 | command spec SHA-256 | code/config SHA-256 |
| --- | --- | --- |
| 左拇指 | `1beb31b45c63aa1fb60eeb2a63652ea9009088d40d2c6ece7389d72207013d73` | `0918aca3f52b88c3fe24fb6d3d457eed81d9f8cb5f272125f9e453debc7b69fb` |
| 右拇指 | `2fcbf8b7d42711f42db4a7f2f84237d4084b85db77ef558070782c51f02da6cc` | `51144337a568425e2ac4c7d4c1c90421f156e21bf28907d23c2b3ae222703889` |
| C3 | `47fd6d1b18c2f2cb40ed9a6082d564f5e4cd0982d9d8113398fc0588586ac4e3` | `3218f2fa287f9b65d01cddc79b8e6ceeda2140c18aa7af7f3d0491c69d8ab5c2` |
| C4 | `ef59621be11834327c73be02da5391b9e117c46a8851f863cab1b0af5ac55d49` | `a5d3eccdd4d0da813208b9ddafd7462ff11fe642e63a0c2ac7540d60009fcf0e` |

私有证据根为 Ubuntu 上的 `$HOME/.local/share/dh116-autonomous-3c80166/evidence`。每次实际 `launch_argv.json`、`outer_exit.json`、`native.log` 与 `receipt.json` 位于 `native-attempts/<工作包>/<code_config_sha256>/<attempt>/`。原始日志、训练模型和数据留在此路径。

- P0 批次批准 SHA-256 `dcd89c5b010d4731460f01817212f11883f6a0d0e8c20352df88eb900d93187e`。独立账本审计 `continuation-p0-approved-93c17beb/independent_batch_audit.json` SHA-256 `410cfc4fd5a61351b49ea6209a14c1d2d5b2260c80c02888dae2ad7fda369956`。原配方与源码身份见左右运行回执。
- C3/C4 批次批准 SHA-256 `f232863a283acb3bf4a5f4d1e7333b4b1079a83e53814687637c4a23216f79a7`。C3 回执 SHA-256 `c0c48203d13e7de5416fca42d9a8d40096c588d8ca9eef5aa858f733c4a24f42`。C4 回执 SHA-256 `7ef892c4dc65ea85168e167b023ec5d07c496f7aed12f99911d45238de5c05d6`。
- 相机精简栈与映射审计 `continuation-camera-c3-c4-stack-audit-VI8fRW/audit.json` SHA-256 `f953921955af541d3849fc0bfd863f0123f4bfdb093d0f3538e823dad8053576`。同进程 C4 live proc maps SHA-256 `4a1639447a7a3459b310a48bc1ca84a5c839f09c851bc5ba3c6110328bd0b0cd`。

## 真实数据与断点

WireSeg 已从 6 增至 **8/148** 个完整官方 blob。新 `data/train-00009-of-00045.parquet` 和 `depth/depth-00020-of-00100.tar.gz` 分别核验 LFS SHA-256 `b51ef63355390dc333d4139b307bcebd1ecbb40db015046a68921657c1e12307` 和 `612322216330fedb4d969c7e3b5ee31829571df1c8a045dca06d52c5d128d514`。新 RGB 分片 800 个训练样本，新深度 360/360 精确配对、713 个实例掩码。192 中心裁剪有 292 个有效深度/ROI 样本、68 个无对应裁剪标签。它们是单目 metric depth，双目视差监督仍待本任务同步采集。

私有 `continuation-paired-depth20-train09-rkBj9Q` 下的入库回执、训练子集和逐样本账本 SHA-256 分别为 `eb32001bb97293b72e5339f9dbbf11e2c98641ef42578d066c514c4d9c155995`、`dafdda5a62a35b401872dd11ead4219888c0928cc7032fcb0bdcc87bb2eab524`、`ea07e6b56a4a348b51f086efe19851d341e272df3cb4eddbff076846da32ad31`。原诊断训练子集 `38159e2f94a7528f983a1b9dca3a74f9f8bb91e5576046d31c9413b386ae905d` 与验证子集 `5d5cc1331115737bfd039fa8223bd0f23cd67ea385595ffc0f03e71e6115f78f` 仍为原字节，新配对未混入。

两份历史第 128 次更新断点分别为 segmentation SHA-256 `77459b47f26ccfdeea024f298fe59bedbb9c119242cddda305aeea0b71cf5f39` 和 depth SHA-256 `56826811b64075446486fc0309a60a616dabcafc738d40700be04ec39ac21adb`。只读审计确认 Python/NumPy RNG、显式采样状态和完整 v2 数据及配置合同缺失，严格恢复拒绝并保留历史原件。训练 JSONL 可读回 128 次 segmentation 访问、0 跳过，以及 158 次 depth 访问、30 跳过，并与旧权重中的更新记录逐项对应。后续若准备显式迁移，可保留 model/AdamW/Torch RNG 并由账本恢复游标，须明确记录无法证明 Python/NumPy 状态等价。本批没有迁移、训练或自动回退到从头初始化。

## 后续资格和预算

1. P1 保持独立 `deformable_rod` 配置，先补自由杆重力、原始材料、端部边界、杆桌双向接触和实际 Reset。单手 Pinch 必须在同一根真实杆上同时证实两指及接触身份，然后才能验证双手同杆、不同固定材料位置和真实无外部支撑。Tripod 和 Power 各自建立可达性与安全证据。没有新原生配方获得授权。
2. P4 实物记录需在 480 Hz 物理和 60 Hz 控制时钟上提供同一 episode/reset/rod 身份。外力、附着和桌面外支撑的已确认违规锁存至下次有效 Reset。Hold 同时要求至少 2,400 个连续完整物理间隔与实测跨度至少 5 秒，且 Lift、接触与 Slip 每一步均满足有效性。材料和逐手切向参考 UID 在本回合冻结，正常弯曲与沿材料或指腹切向漂移分别计算。保留 30 mm Lift、5 s Hold、0.5 mm Slip 标准。配置中的 R0 正式训练系数和 `training_reward` 仍未获资格，R2 关闭。
3. P5 对每个抓法先验证真实 371D/24D、结构化节点、GAE 终止、原生 R0 和 Reset，再冻结无相机短 rollout、独立单次更新和 Checkpoint 恢复的确切预算。32 个 60 Hz 控制步对应每环境 256 个 480 Hz 物理间隔，若 2 环境则为 512 条环境级测量行，启动和 Reset 额外物理步另计。这是预算换算，未授权执行。完整 Hold 预算还须覆盖至少 2,400 个合格间隔以及前序接近、抓取、抬升。三种 Teacher 不依赖相机或 WireSeg 完成，合格 MPC 专家可选用于初始化。原生完整成功 Teacher 轨迹方可进入 Student 专家池。
4. P2 可继续有界下载和 SHA 复核。新诊断可按独立 8–16 个配对样本、单一 head、最多 16 个新更新和 120 秒制定预算，输出全图与 wire ROI 误差。真实训练仍待可验证源断点或明确迁移决定及对应运行范围。
5. P3 按已保存栈先审查 rendering experience 的 Extension/Carb 所有权。后续只在提出具体新单因素配置并获批准后运行一格，不沿用已消耗的 C4。P6 的 8 初态同步采集与每种抓法至少 100 独立初态 Actor-only 验收均另有证据门槛与预算。

旧配方的批准与回执在现有恢复逻辑下已消耗，本批剩余可执行的原生次数、物理步和 PPO 更新均为 **0**。后续新指纹、样本规模或训练时长须在实际执行前独立冻结。

### Registry 说明修正

| 工作包 | 结项状态说明 |
| --- | --- |
| `flex.f4-dh116-same-rod` | 修正版左右单拇指标定各 240 SDK 步并自然退出 0，预算合规。历史 484 步证据仍标记超额。自由杆真实重力、单手多指、双手在同一杆不同材料位置及外部支撑审计未取得原生资格。 |
| `flex.f4-single-hand-contact-native` | 两侧单拇指标定的预算与生命周期已通过，场景为固定端零重力。独立自由杆任务的单手 Pinch 两指对向接触仍需真实原生证据与新预算。 |
| `flex.f5-wireseg-pretraining` | 已核验 8/148 个官方 blob。新配对 360/360，中心 192 裁剪 292 有效、68 无标签。两份第 128 次更新的历史断点缺严格恢复字段，新 v2 软件回归通过，原诊断清单保留。本批无真实训练。 |

## 完整性与发布回读

528 个保护文件和两个历史 Teacher Checkpoint 已通过 SHA-256 核验。历史 `teacher_update_000000.pt` 为 `ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511`，`teacher_update_000001.pt` 为 `c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999`。保护清单摘要为 `fb6b58cfd2fe86408eb498a4390ecda35df2221d7198ca3a01ff59f86dea05ec`。现有刚性环境、原控制合同、旧 Checkpoint 和 11-10 基线不在改动范围。

合并 CPU 回归结果与最终测试树由结项发布回读填写。

最终集成 CPU 回归：357 passed / 0 failed / 0 skipped，exit=0。测试树 HEAD 为 `28b6984b75444d9de4282d9e38c8df02d44f52a1`，运行日志 SHA-256 `bc5594410e2ab3d588284d8845f4a371bc21f7d2b0a27cfe5557bffb78e8bcdc`。

命令：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_work_packages.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_task_budget.py tests/test_dh116_deformx_bridge_lifecycle.py tests/test_lift_hold_window.py tests/test_dh116_stereo_runtime_rod_identity.py tests/test_stereo_action_contract_audit.py tests/test_stereo_privileged_state_contract_audit.py tests/test_privileged_teacher_reward_audit.py tests/test_dh116_wireseg_checkpoint_resume.py tests/test_dh116_wireseg_overfit_diagnostic.py tests/test_dh116_wireseg_pretraining.py tests/test_dh116_wireseg_blob_relay.py tests/test_dh116_simulation_app_lifecycle.py tests/test_dh116_render_capture_boundary.py tests/test_dh116_deformable_privileged_teacher.py tests/test_dh116_privileged_teacher.py tests/test_dh116_deformable_privileged_ppo.py
```

首次合并回归保留 356 passed/1 failed，日志 SHA-256 290d7b78aaf8610c212d6fe43c09d982037e33970503477be9ac5f839985ef14。失败为新测试对 float32 复算使用了额外1e-6断言；原冻结策略校验一直为atol/rtol=2e-5。对应修复和最终结果分别保留，未重跑原生。结构化 PPO 已集成为 `5f5398cb2083c60315f7e9cc8bee0b33cf115c52`。严格 Adam 设置与完整 rollout 计数在修改 live 状态之前校验，9项专项独立复核通过；CPU 结果不构成原生训练许可。

最终 GitHub SHA 由本报告之后的 Windows 发布回执与 release manifest 记录，避免报告自引用提交 SHA。PR #2 保持 Draft。下列为此代码冻结点之前本轮的实际提交：

```text
28b6984 Attest verified local GitHub releases
1c5c05e Test frozen policy against its preserved float32 audit contract
d7fdeca Attest verified local GitHub releases
5f5398c Add isolated structured rod PPO runtime and exclusive checkpoint recovery
f4fe60f Attest verified local GitHub releases
e1cc651 Record C3 natural exit and C4 extension-wrapper shutdown crash
f5863fc Attest verified local GitHub releases
dd32f1b Require five measured hold seconds and immutable material slip references
13486e9 Attest verified local GitHub releases
d7f6bb6 Camera C4 isolated lifecycle [flex.f6-camera-c4]
27b881e Camera C3 isolated lifecycle [flex.f6-camera-c3]
dea157f Add isolated grasp-specific privileged rod Teacher model contracts
c52b9dc Attest verified local GitHub releases
fc7430e Reject lift qualification after transient external assistance
075db84 Preserve complete WireSeg diagnostic state and audit legacy continuation gaps
9869c17 Audit AppWindow teardown and independently verify synchronous depth archives
094baf9 Add isolated flexible grasp reset, material slip and R0 evidence contract
53bf6ac Attest verified local GitHub releases
22210b4 右拇指与柔性杆原生接触 [flex.f4-right-contact-native]
fa5bbbd 左拇指与柔性杆原生接触 [flex.f4-left-contact-native]
3f065c7 Attest verified local GitHub releases
0b053b2 Verify corrected thumb phase budget, rod contact identity and natural shutdown

```
