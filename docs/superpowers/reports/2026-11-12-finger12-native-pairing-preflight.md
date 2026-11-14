# 右手 finger12 原生配对预检查

检查 Ubuntu 集成分支 `codex/dh116-stereo-privileged-dagger` 的 `1e2a45ac6a8f4966b55919f0f4b9be288cbf9f2d` 与保存的三组原生数据。当前结论限于合法目标下实测位置越过保护线。三路动作的物理因果对照仍待执行。

## 归档实测对齐

| 证据 | 动作与初态 | 480 Hz 关节证据 |
|---|---|---|
| Teacher128 v02 | 原 `teacher_update_000000.pt` 的采样动作真正执行。7 个 episode 对应 2 个不同的实测初态摘要。CPU 在同一 371D 状态上复算 mean，原生轨迹只执行 sample。 | 2,030 个样本含 12D 主动 q/qd。五个右 `finger12` 终止位于物理 tick 250、301、540、784、835。归档缺完整被动关节与腕部 q/qd。 |
| Cold8 v02 | `teacher_update_000001.pt` 的确定性 mean 真正执行。32 个策略状态来自 8 个不同初态，每个初态 4 步。与 Teacher128 的初态摘要交集为 0。 | 256 个连续子步含主动 q/qd。对 8 个 reset 文件重新计算完整 articulation 初态 SHA，全部唯一且匹配保存值。后续子步缺被动 q/qd，0 个 terminal。 |
| MPC sync32 v02 | `pressure_geometry_constrained` 原路径执行。64 个 371D/24D 行来自两个原始 MPC 初态，seed 42。以原生固定单目 RGB-D 捕获。 | `native_physics_substeps.pt` 确实保存 256 scene ticks × 2 env × 2 hand × 17 SDK joint 的 q/qd，以及每手 5 个 mimic 残差。同步状态文件的 `robot.left/right_q/qd` 各只有 6 个主动关节。真实逐接触冲量与硬件力矩资格仍为空。 |

按实际 joint names 重排左右各 6 个主动关节，使用 SDK 索引 `[6, 11, 7, 8, 9, 10]`，从保存的初态直接计算最大差值：

| 抓法 | MPC 对 Teacher128，初态 max Δq / max Δqd | MPC 对同抓法 4 个 Cold8 初态，max Δq 范围 |
|---|---|---|
| 二指 | 0.200000003 rad / 0 rad/s | 0.199508727–0.199906752 rad |
| 三指 | 0 rad / 0 rad/s | 0.000387549–0.000492334 rad |

MPC 与 Teacher128 两组的环境局部物体位置及四元数也完全相等。三指这些量相同，但 Teacher128 缺完整初态被动与腕部 q/qd、双手根状态及同一测量时钟绑定。Teacher128 seed 为 116，MPC 为 42。Teacher128 物理合同摘要 `2bd771fb22f951dd0bdd8f52a0af55c01d24382ad5efe09ab19e22603a936566` 与 MPC 的 `7128e10e341316dd76132eec5ee1d5af6f769691eecbe281d99a26b974c00612` 不同，至少包含 grasp initialization 源和传感器批次配置差异。没有可证明的完整同初态三路样本。

| 只读证据 | SHA256 |
|---|---|
| Teacher128 `native_physics.jsonl` | `470e8c5d0f24b2cf97c1664ea26dbd566aa58969677d0b3f64660ff22e2d6ffc` |
| Teacher128 `teacher_update_000000.pt` | `ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511` |
| Cold8 `native_physics.jsonl` | `b4d34bacf2dea7175d8339432de38e96c26d47da5105f353449102c25593a64b` |
| Cold8 `teacher_update_000001.pt` | `c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999` |
| MPC sync32 `initial_native_state.pt` | `f5385dbee211b4409493ca47bb1a49964e74d950ca6b1b542e9ce06f31f3826d` |
| MPC sync32 `native_physics_substeps.pt` | `84f8c69558a68301cb80273ceca735ba453067fbb723c06d40a9431613a90646` |
| MPC sync32 `mpc_privileged_pre_action_unqualified.pt` | `5f9a419049121a61caa451a03878c6d5be673f94b2fa98b28bf6bca56d0fdfed` |

## 可复用入口与缺口

- `scripts/evaluate_dh116_privileged_teacher.py` 固定 deterministic 和至少 100 初态/抓法的正式验证。`scripts/collect_dh116_stereo_teacher.py` 使用明确预算的 Cold8 manifest，执行 mean，现有配置绑定 2 env、每初态 4 步。候选采集路径不会触发完整 `TeacherJointActionAudit`。
- `scripts/train_dh116_privileged_teacher.py --record-diagnostic-rollout` 提供现有全 articulation 探针，入口绑定 2 env、32 或 128 步、1 次 update。`NativeTensorBridge.set_teacher_action_diagnostics` 当前依赖有效随机 latent 样本来启用此探针。此训练入口缺单次无优化的 cold triplet。
- `scripts/collect_dh116_mpc_privileged.py` 保留原 MPC reset 与控制，入口只有 baseline plan 和输出等参数，缺同一 cold manifest 安装。复用 `_teacher_optimizer` 的原 `pressure_geometry_constrained` 实现和原控制步调用，先验证相同输入上的 24D 输出与旧调用一致。通过旁路补齐统一的关节记录。现有执行命令无法单独满足三路配对。

## 有界原生配对计划

用现有 `prepare_cold_trial_batches`、`_install_reset_trials`、`_readback_reset_trials`、`_restore_reset_buffers` 与原单实例锁构建独立的诊断入口。Teacher 两路使用同一 update-0 原始 Checkpoint、原分布与 source 检查。MPC 使用同一场景、同一动作合同和 SHA 绑定的原 expert source。用共同基线重建一对原生完整初态，二指、三指各一个。若根据现有 Cold8 选择近邻初态，使用 `derive_policy_trial_manifest` 显式重绑 update-0，保持试验物理状态，同时将结果标记为新初态上的配对。原 Teacher128 的终止状态保留为历史证据。

三路分别为 A：Teacher sampled，B：同一 Teacher mean，C：原 MPC。每路只运行一个 2-env batch。先运行每路 4 步的独立 pilot 验证账本，再在核验通过后从相同初态重新运行每路最多 128 个 60 Hz 控制步，0 PPO 更新。pilot 上限 24 env×policy 样本和 192 env×480 Hz 样本，正式诊断上限 768 与 6,144。全程顺序单实例，独立输出，实测资源合格后才启动。预算覆盖原失败时间范围，达到任一真实 terminal 就停止该回合的有效采样。保存对应末次原生物理样本及缺失后续步。

- 三路使用一致的 seed 116、相同 env slot、固定同一 calibration SHA、完整物理合同和 reset 缓冲。首先逐值比较本地手根姿态/速度、每手 17 个 q/qd、线缆位姿/速度、控制器和 MPC 初始历史，核对一致的完整 reset SHA 及 371D pre-action 状态/validity。按现有 2e-5 写入/读回容差核对预设，在策略执行前检查每一路实测初态相互匹配。旧运行的 seed 一致性仅作记录。
- 每个策略步保存相同 actor 下的 mean、std、A 路 latent 与实际 sample、实际 issued 24D 和 C 路独立 reference。每个 480 Hz 子步用 `TeacherJointActionAudit.capture` 既有关节读回逻辑，补齐相同的 env/trial/episode/seed/frame/tick 身份、前后投影、控制器和 SDK 目标、全关节限位与原始保护余量。同步记录 mimic 残差、掌位、载荷、contact valid、终止与力学来源。对 deterministic/MPC 的无采样字段保留 NA；根据来源单独标注仿真估算力矩或独立 PhysX 冲量，资格缺失时保留 NA。
- 第一组 pilot 如出现初态、合同、时钟、动作 owner、SDK 映射或字段完整性差异，保留失败证据并停止后续执行。出现原生关节 terminal 时正常保存失败轨迹。三路各自报告首次跨线、位置速度与目标、接触与 passive 残差时序。若只能复现跟踪越界，接触和被动耦合机制保持待验证。只有在同初态配对重复后才考虑局部单因素干预，现阶段沿用原关节保护、物理和 30 mm / 5 s / 0.5 mm 成功标准。

## CPU 验证

Ubuntu 只读比较原始 JSONL、`initial_native_state.pt`、`native_physics_substeps.pt` 和 Cold8 reset readbacks。集成 `1e2a45a` 的对应 CPU 回归为 40 项通过，0 失败，3.62 秒。本报告记录归档和执行准备，本次原生配对次数为 0。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= DH116_NUM_THREADS=2 \
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_privileged_action_audit.py \
  tests/test_mpc_privileged_capture.py \
  tests/test_dh116_stereo_candidate_cold_trials.py
```
