# T5 原 MPC 完整首回合原生结果

冻结源码 `9a74290b5acc4660d103f6028b360b1046228739`，沿用原 `solid_cylinder`、`pressure_geometry_constrained`、seed 42 和原 371D／24D 合同。GPU 3 单 Isaac 实例运行，8 环境、480 Hz 物理、60 Hz 策略。进程 05:01:51 UTC 启动，06:21:21 UTC 自然退出，外层退出码 0，原采集计时 4673.243 s。完成 1800 个策略步、1800 个 RGB-D 帧、0 PPO 更新。结果路径 `/mnt/mydisk/xk/dh116/runs/wp3_mpc_t5_first_episodes_v01`，约 5.7 GiB，仅在 Ubuntu 保留。

## 原生首回合

完整 371D／24D 同钟归档 10,559 行，匹配首回合的有效 480 Hz 环境级物理记录 84,461 条。原生终止、动作 owner、帧、episode、SDK／World 时钟逐项核验。二指成功四回合 4,698 行，三指失败四回合 5,861 行，未完成零回合。原生成功、失败和未完成分开保存。成功使用原任务 Lift ≥30 mm、Hold ≥5 s、Slip ≤0.5 mm 和接触样式判据。

| 抓法／环境 | 首终止策略步 | 原生结果 | 首回合策略步观测峰值／终止净 Lift（mm） | 终止 Hold（s） | 终止 Drift（mm） |
|---|---:|---|---:|---:|---:|
| 二指 0 | 1174 | 成功 | 31.590953／31.589504 | 5.0 | 0.00235407 |
| 二指 1 | 1175 | 成功 | 31.537025／31.535705 | 5.0 | 0.00220549 |
| 二指 2 | 1175 | 成功 | 31.531059／31.530101 | 5.0 | 0.00649427 |
| 二指 3 | 1174 | 成功 | 31.627685／31.627685 | 5.0 | 0.07963194 |
| 三指 4 | 1800 | timeout | 0.002182／−0.118671 | 0.0 | 0.0 |
| 三指 5 | 1800 | timeout | 0.012875／−0.137716 | 0.0 | 0.0 |
| 三指 6 | 1092 | object_dropped | 0.004342／−10.677688 | 0.0 | 0.0 |
| 三指 7 | 1169 | object_dropped | 0.008547／−10.567251 | 0.0 | 0.0 |

八个首回合均无恢复。原历史 T5 的 `initial_native_state.pt` 与本次 SHA 相同。独立 `cmp` 确认两次全部 1800 行 `teacher_native_physics.jsonl` 逐字节相同，涵盖时钟、净 Lift、Hold、Drift、物理状态、终止、失败及成功。两次 `all_executed_supervision.pt` 的 10,559×24 实际 issued action 和 reference action 也逐元素相同。此次增加旁路归档，物理与控制行为按该次配对证据复现。历史二指 4/4 与三指 0/4 的结果分别复现。

## 状态证据和训练准入

原始 371D 归档独立核验 24 个主动关节位置与速度字段。额外只读的原生 sequence context 数值审计核验全部 10,559 行中 204 个字段，四个腕部区间字段 367–370 在 10,551 个非 reset 行逐值匹配。该审计退出码 0。它证明模拟器观测的数值和源一致性，保留原采集的逐字段有效性，reset sentinel 与完整物理区间仍分开。

正式 BC 输入资格仍关闭。四个二指原生成功首回合在成功物理归档内，独立可组装正式专家回合数为 0。独立准入预检记录三个实际缺口：成功行缺完整 371D 有效性、缺统一的 480 Hz episode 与力学支撑资格、缺独立原生回合资格。此次未产出正式专家池、Actor-only BC 权重或 R0 PPO 更新。三指需要独立原生成功证据。

## 证据 SHA256

| 文件 | SHA256 |
|---|---|
| 冻结源码逐文件清单 | `2380e42a349b5564bfe5ee09cbe12ae1667fd704cd7d8a882118c513e8e0df4e` |
| T5 实际 preflight | `15434c7b275cb8d36db10c242ea4ea92407cce82ec3ce2addcd7db1d115e847d` |
| Ubuntu 资产路径核验记录 | `0a74803042240b86eb826d786718e3d52af751648ffa461c29e3022776542cc5` |
| 原生启动器日志 | `0b0ce8bf82747da2bb012494877dcd548472cf91951f9ccf5e4aab61a245fc81` |
| 新旧相同的实际 reset 文件 | `7a8d7987b7c0f72d441244f746ad8dbf82f826f6e2e56066fe9523f9352ca67e` |
| 新旧相同的策略步物理 JSONL | `6c8c88c7cf3338662e72267cea0c089f10060dda7355de272f9744ac8918a5a3` |
| 新原生采集报告 | `4f1404abf537835e639db1588a9f13b3cf640e1034dbff8f37f5e2bc85d96c36` |
| 同钟 371D／24D 完整归档 | `bc3043c14ace766980123b08daa5b2dc180d7baa756a58ed88aef33bfe4981fe` |
| 原生成功四回合归档 | `f78cc413df95f04f937d29ded05d7316b3cef3da94e596614f68a97d11969b44` |
| 原生失败四回合归档 | `17851584b7faa5b6c938ae1cca817f3500387186163f07e3e6242777dd847c0a` |
| 原生未完成零回合归档 | `627460476d22a6485154920b7a37513327050f2b4680646ee4cdd852112a63e3` |
| 480 Hz 物理状态归档 | `ff3fc0dfbe55420a4b2a9cb7ea6fcc77e9bdc6d9f29ab57bab6279bdc12259a2` |
| 480 Hz 原始传感器归档 | `bf2fc1243f7793bef28a00c95a72658238c18b4de6e1942bab2631793d30ce55` |
| 实际终止快照 | `f95f97ca49a7bf06cc66ea16a25c108aefdb2c344e2349e5fa0a403ef355104f` |
| 原生 sequence context | `28a6f3ac189e9dc869ccb8a2617d9b2e0e9d3a6cc38864591f12e3a81940eb45` |
| 完整采集独立核验 | `75cef015456afb07d12cb5252adb287908fb4b2d44910872ddefbefce6096803` |
| 独立 BC 准入预检 | `53fed7957d65c81281df6f1e4ca085b39b53e20efce4a1ba8bd34c15ae4a39ce` |
| 204 字段与腕部区间的只读证明 | `1d0743b48a60e643ba2516558b5eb2f7644c4362e6603a51ef8df258be8f379f` |

## 运行与 CPU 核验

原生执行使用 [冻结方案](2026-11-12-mpc-t5-first-episode-capture-preflight.md) 中的命令追加 `--execute-native`，确认旧的实际输入 SHA 与独立输出，记录目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-t5-mpc-native-20261112-SgPe6Q`。原始输出和物理资产未改动。当前报告工作树复核 528 个保护文件和 2 个历史 Checkpoint。

```bash
export PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
python=/mnt/mydisk/xk/envs/dh116/bin/python
"$python" -m pytest -q -p no:cacheprovider tests/test_mpc_privileged_capture.py \
  tests/test_dh116_privileged_action_audit.py
cmp "$new_run/teacher_native_physics.jsonl" "$original_t5/teacher_native_physics.jsonl"
```

此报告工作树 Ubuntu CPU 回归 37 项通过，0 失败。集成 CPU 回归记录 265 项通过，日志 SHA256 `c9279bbc749797b648ac03f1213fb91b244e11c073a2dbc6bf9d889a4c678321`。正式 Teacher 的下一门槛仍是完整 371D 证据和独立物理力学资格，之后才能运行 Actor-only BC、R0 和冻结 Teacher 验收。
