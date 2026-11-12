# Cold8 双目同步原生采集

采集源码基线 `b2f5d84ff5e3f9d87d5cd24ba55a3f4edb6f8ca6`。沿用现有 `solid_cylinder`，任务身份为 `rigid_proxy`。先确认 PhysX v3 的 `initial` 与 `restart` 各完成 240 个 480 Hz 步并自然退出，再在 GPU 3 的 RTX 4090 上顺序运行采集。v3 汇总 `shutdown_retest_passed.json` SHA256 为 `a33ec3726252a3ac08607fc4b0825e02291679a5f7dbf7b6032472b8476c46c3`。

## 两次执行

| 尝试 | UTC | 外层退出码 | 结果 |
|---|---|---:|---|
| v01 | 04:30:43–04:31:42 | 2 | 初始化读取资产来源时触发 `FileNotFoundError`。受版本控制的清单指向旧机器绝对路径，缺失文件为 `d134680ab619_DH116_left.usd`。只生成场景 monitor 和失败 summary，采集 0 条。 |
| v02 | 04:32:47–04:33:47 | 0 | 使用新的冻结目录，逐文件核对清单内 4 个 USD 的 SHA256，只重定位快照清单的 `sources[].path`。四个批次各完成 4 个控制步，原有 reset buffers 已恢复。 |

v01 的日志及输出原地保留。v02 左右手 USD SHA256 分别为 `7c6b319074281775c45bc83ac8f434b9a4b7e7e7ba18815e345577d068eb9e6c` 和 `37ff2a3a724778696f96732f713c0b55c01cc96de0f1c35d4c7e8a7fd7532f71`，均与原始清单一致。v02 重定位清单的 SHA256 为 `3b9c9335747a02b1a793b74fce1003b68962e80b4816190c901f102755b79353`，逐文件重定位记录 SHA256 为 `67aad6a58b0a4be20aa8e82d81dc8a9471da35e4678fd5cf78aa549848fcc000`。原 USD、物理、控制器、371D 和 24D 合同沿用基线。

## 原生数据与只读核验

| 抓法 | 实测独立初态 | 同步策略行 | 640×480 RGB 对 | 双侧 Native ID | 双侧 Optical-Z | 480 Hz 样本 | Center XYZ / 线速度 XYZ 可用行 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 二指 `pinch_thumb_index` | 4 | 16 | 16 | 16 | 16 | 128 | 12 / 12 |
| 三指 `tripod_thumb_index_middle` | 4 | 16 | 16 | 16 | 16 | 128 | 12 / 12 |

- 8 个实测初态 SHA、episode UID 和清单 trial ID 各自唯一。逐批核验了实际 reset readback 与其 SHA，检查了 RGB、双侧 Native ID、双侧 Optical-Z、371D 状态及逐维 `bool` validity 的 frame、reset、physics tick、时间戳和标定 SHA。
- 32 条记录的 24D issued action 与 Teacher deterministic mean 逐元素一致，每个控制步绑定 8 条连续、有效的原生物理样本。另核对了 12D 实际 SDK 关节目标、控制器读回及 2×7 实测掌位。原始账本含 8 条 `episode_start`、256 条物理样本，0 条 terminal，0 条标记为 `native_success` 的样本。0 个完整回合达到正式评估所需覆盖。
- 每条的两侧 ID 张量均可读，并解析出非空的原生线缆 mask。16 条二指策略行的左右 mask 像素累计 32,399 / 32,461，16 条三指为 33,067 / 33,261。16 个共享批次帧进入候选，另有 4 个动作后参考帧归档。Reference 深度保留为标签及核验源，部署输入使用 RGB 计算深度。
- `read_candidate` 通过原生候选、480 Hz 原始账本、标定与 reset 证据的绑定校验。每种抓法的逐字段有效位累计为 456。中心坐标和线速度的资格分别计算。刚体旋转对称性及独立验证的数据限制仍适用。

## 证据索引

原始数据与日志仅保存在 Ubuntu。v01 结果目录 `/tmp/dh116-stereo-cold8-native-20261112-v01`，日志 SHA256 `dc4c4a7f2e4eaec4e8a0530752b82abc956cdec2fe2822f6687ba738b93c78fb`。v02 结果目录 `/tmp/dh116-stereo-cold8-native-20261112-v02`，日志 SHA256 `618a0f7bb903d51b65d95dc770034b8566e60ed5d7fc767aa542ea9666a9a53a`。v02 冻结快照、执行命令和重定位记录位于 `/tmp/dh116-cold8-execution-20261112-v02-8UbaB8`。

| 项目 | SHA256 |
|---|---|
| Teacher checkpoint `teacher_update_000001.pt` | `c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999` |
| Cold8 清单原始文件 | `e2e1280e02cfe193678b2474a179e4e7207c4239d75641b7e2efc2ad5c969e83` |
| Cold8 清单语义身份 | `afbb1984ab20c296921e251a78846eb9cab665582b35a6328eec73eeea5fed78` |
| 双目相机标定 | `02fabb87c548cd982728dac5d6357ca08327258514269b163e880427b2556700` |
| Ubuntu 相机配置文件 | `6b4a3d341a76c75e11163d16b79c70661b051e87ba66658f962037c6864e16e8` |
| `stereo_native_task_v1.json` | `2bd172450eb6b900cc14f22b204762f2cdd4aa3d129b7b78c5573c482fd3093b` |
| `completion.json` | `b40120d414a268dd23c407e205271a3f5a5cb19eb31f76b4537265e8228c4682` |
| `cold_initials/cohort_report.json` | `737e694665a93e696e5907df5a462ecdc7466b5766f4c4ddde1bea116653684a` |
| `stereo_teacher_candidates.pt` | `1366bf862643c0c7ecd629e785da27cdb4c5e01087dc23b04b74e8cf937678c8` |
| `native/native_physics.jsonl` | `b4d34bacf2dea7175d8339432de38e96c26d47da5105f353449102c25593a64b` |

## 命令与 CPU 回归

实际通过冻结目录内的 `scripts/python.sh` 执行，`DH116_NUM_THREADS=4`，`CUDA_VISIBLE_DEVICES=3`，指定现有 Ubuntu 解释器 `/mnt/mydisk/xk/envs/dh116/bin/python`。`--check-config` 预检查通过。采集命令如下，`--output` 需要新目录。

```bash
bash scripts/python.sh scripts/collect_dh116_stereo_teacher.py \
  --config configs/dh116_training/stereo_candidate_cold8_v1.json \
  --teacher-config configs/dh116_training/privileged_teacher_ppo_guarded_v2.json \
  --teacher-checkpoint /tmp/dh116-wp3-teacher128-v02/teacher_diagnostic/teacher_update_000001.pt \
  --camera-config configs/dh116_training/stereo_camera_roll20_surface96_diagnostic_v1.json \
  --native-task-config configs/dh116_training/stereo_native_task_v1.json \
  --candidate-only --state-reference \
  --cold-trial-manifest /tmp/dh116-wp2-cold8-plan-20261111-v01/candidate_cold8.json \
  --cold-minimum-per-style 4 --num-envs 2 --steps 4 --seed 116 \
  --output "$new_output" --enable-experiment --execute-native
```

Ubuntu 的 Autowp2 独立 CPU 回归 40 项通过，0 失败，用时 9.05 秒。以下测试覆盖候选冷重置、准入和同钟状态监督。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= DH116_NUM_THREADS=2 \
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_stereo_candidate_cold_trials.py \
  tests/test_dh116_stereo_cold_candidate_admission.py \
  tests/test_dh116_stereo_state_supervision.py
```

## 下一步及准入状态

采集资格为 `diagnostic_only_cold_cohort`。保留独立初态分组，可用于 Center XYZ 和刚体线速度的小规模离线诊断及持出评估。正式监督资格、双目精度资格和原生抓取成功需要各自的验证。后续 MPC 同步专家轨迹须记录实际 371D pre-action 状态及 issued 24D，只有达到 30 mm 净抬升、5 s 保持、0.5 mm 最大滑移及其余原生条件的完整回合才能进入成功 BC 池。正式 Teacher、BC、DAgger 和 Student PPO 在本采集范围之外。
