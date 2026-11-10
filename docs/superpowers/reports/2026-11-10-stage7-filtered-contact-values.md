# Stage 7：首回合退役记录与 filtered-contact 数值观测

本单元保留 M0、原 Teacher/MPC、3844D Actor、371D Critic、24D Tanh 动作及原物理、材料、执行器和 30mm / 5s / 0.5mm 标准。新增观测均默认关闭，只供仿真 Teacher 诊断，不进入 Student。当前对象仍为 rigid Cylinder proxy，不能称为柔性线缆动力学验证。

## 实现与验证范围

- `--vision-mpc-filtered-contact-epoch-capture` 在独立 v2 ledger 中永久退役每个环境的首回合。保留首 terminal；后续 reset generation 只能通过显式 before/after 收据归档，不恢复首回合资格。v1 原严格拒绝行为保留。
- `--vision-mpc-filtered-contact-values-capture` 依赖原 state、aux、F0 及 epoch 开关，复用原 34 个 sensor view。按实际 sensor row、filter column、env 和 SampleClock 绑定 normal/friction 原始数组、owned interval 和 sensor-side world 向量。两通道独立 valid/NA，不把缺失补为零。
- 保存完整 source/stage/view/getter 时钟与原 issued24。读取、原 scene 或归档失败时保留 partial 和首 terminal；只有完整行成功写入后才推进 epoch ledger。未增加 step、refresh、外力、附着或控制目标。

CPU 覆盖混合 live/terminal、退役后的 no-op/reset、首 terminal 保护、source/view/clock 漂移、owned 数组与 dtype/count/capacity、无效通道、写入失败及原默认路径。所有 fixture 仅计 CPU 测试，不计原生回合。

## 实际冻结与 CPU 命令

独立目录：`/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_contact_values11`，下文记为 `$B`。源码 manifest SHA-256：

`83a98ce023e22895539af69413266e8c48b1d71a7a604bde65cc3515f281c8ec`

独立 `/tmp/dh116_stage7_mpc_contact_values11_cpu` 副本执行原 24-file 命令，只删除未发布的 `tests/test_bimanual_mpc_objective.py`，成为 23-file 公开范围。实际 639 passed、1 CUDA-unavailable skip、238 subtests passed，21.95s，exit0。运行前后完整源码 SHA 一致；descriptor、mounted FK、objective 均不在本次公开验证范围。

```sh
/mnt/mydisk/xk/envs/dh116/bin/python -B "$B/run_stage7_contact_values_public_cpu11.py"
```

公开 CPU validation：`88371d94dd72c489591ed3ba7c7464f4ff19c9aa3c99b8161a0fcfa84fe9ea71`。含独立 objective 的较大 CPU 组合另为 687 passed / 1 skip / 238 subtests，不能用它代替本次明确范围。

## 实际原生短测

每个实验独立目录、显式 prepare/execute、非阻塞全局单 Isaac 锁，无队列或自动重试。原 Teacher02 模板 SHA 为 `1a5faead16451a7dd0b523cdc6a1e236a6882c80c26a2a2bd295d2a2dfe2e2d9`；仅缩短记录预算，原控制器、输入与资产前后 SHA 不变。

| 实验 | UUID | 控制步 / 物理帧 / 环境状态 | collection wall time | terminal / 截断 |
|---|---|---:|---:|---:|
| epoch1_v01 | d11a1490-8e84-480e-9d1b-8da819e6b184 | 1 / 8 / 64 | 22.522068s | 0 / 8 |
| values1_v01 | f4e5a52e-fbc5-4dc6-8ee7-5c38dd611b6b | 1 / 8 / 64 | 36.861371s | 0 / 8 |
| values16_v01 | cec76db0-7ca2-42ff-9605-30902cfb61a7 | 16 / 128 / 1024 | 567.229840s | 0 / 8 |

三次均自然 process/launcher exit0。短测全部截断，没有新增 Teacher、MPC 或 Student 成功；采集耗时不证明 60Hz 实时控制能力。

| 数值通道 | values1 有效 / NA | values16 有效 / NA |
|---|---:|---:|
| normal sensor-side world | 2176 / 0 | 34816 / 0 |
| friction sensor-side world | 2176 / 0 | 34816 / 0 |

valid 仅表示保存数据可按既有 SDK 数组合同解析，包含零向量。values1 数值清点的两通道非零观测均为0，各448个 owned slots；values16两通道非零观测也均为0，各7168个 owned slots。这些 slots 也可能是零力或释放样本，不能等同接触承载。清点在通过完整后审后进行，proof/原始输入/工具前后 SHA 一致；values1/16清点 SHA 分别为 `958d18a6a353b55f19dcd6cdc6c312e24c6d7b69f6cbd8da1d289790c077a6e6` / `3789716f0aa54c42943ad3afc9d1176f2241db808b180013db1400daa795f5e4`。

## 已执行命令与封存后审

原生有限 wrapper `stage7_contact_values_native.py` SHA：`bb640f0b0d019cb957e43d815030643c17d15cbf392da7e5ca2feefcb6e4cbc3`。严格保留 F0/layout/base 的固定源码身份、原 Teacher02 输入、资产、CPU、独占输出与单实例保护。实际执行各自既有 command.json：

```sh
/mnt/mydisk/xk/envs/dh116/bin/python -B "$B/stage7_contact_values_native.py" --execute --plan "$B/epoch1_v01/command.json"
/mnt/mydisk/xk/envs/dh116/bin/python -B "$B/stage7_contact_values_native.py" --execute --plan "$B/values1_v01/command.json"
/mnt/mydisk/xk/envs/dh116/bin/python -B "$B/stage7_contact_values_native.py" --execute --plan "$B/values16_v01/command.json"
```

对应 command SHA：epoch1 `dc71b0b30c7247ff7ac8a947aa7cb5782249c0e097bd6d7ddd2c3cd0a0dc5be2`；values1 `bfc6773cc946d42069d7d2fd65efa58dd7c980980c5b7bf3d8b01014b70dc0ee`；values16 `406bb8abbb0780db2633583232584deab809452edf1e0890709725e30adb9fd4`。

独立封存分析器 SHA `9d8bb589cf0a3d2592c708047a25f7e1b0c8fb29d538115ab756c4a8535c0e2d`，及其18项 synthetic 测试经过独立 CPU 审查。实际只在自然退出后读取 gzip；先核 source/输入/退出，再查 EOF/CRC、Core/Aux/issued24/clock/UID、完整 F0 raw、v2 receipt/history SHA 及 F1 数组重算。分析器与全部8个运行依赖由独立13-file capsule绑定，manifest `7e716c8c6af8373b6cbe027599a080f02fd9af3b17e4121ecbbb65d8c29ab5a8`。

```sh
/mnt/mydisk/xk/envs/dh116/bin/python -B "$B/analysis_capsule01/artifacts/analyze_stage7_contact_values_capture.py" --run "$B/values16_v01" --output "$B/values16_analysis_v01"
```

epoch1 后审 integrity/epoch 通过、failure0，proof SHA `6f75be4b772ffafa23ec5e3779f42c866ef5e71bbd64567555749d66f992a7b7`；values1 integrity/epoch/F1 replay 通过、failure0，proof SHA `8e6ddef2f22ec2aa9fe3e208b57f79950055f9efa6e7859ae5f9d76a814e57d8`。values16 实际后审也完成，integrity/epoch/F1 replay 通过、failure0，proof SHA `985a532db6389bd0cd2b05b5638b1f462e84bf4df052112d9cb59f9945ca5cc1`，outcome SHA `3e7da6a56c24b6cb684805a66a77760628a5e6835e16a59732e90da2c02221b5`。

## 限制与下一步

上述结果只验证归档数据、时钟、来源与 owned 数值重算，未核定力的作用侧/符号、每个 contact 的 numeric partner ID、上游 overflow absence、accepted pad load、桌/支架反力、合格 support 或 airborne。特别是零力或空报告不能证明无接触、无承托。各力/动力学/planning/native success/new expert 资格仍 false。

实际短测0 terminal，因而异步首回合退役目前只有 CPU 证据，完整原生 terminal/reset 仍待后续。独立180控制步的接触建立观测窗口已在全部16步后审通过后显式启动，使用相同冻结源码、原Teacher02与限制，仅改变记录预算/输出；command SHA `62a2c92d02753ee2fd1ef1e525789e7bdd5ced66b2c2f3af4ad3c022fc4c60f4`。本报告成稿时该窗口仍运行，不计结果或新成功；不是新H8 MPC执行。

后续还需作用侧/法向与接触对象核验，以及完整 mounted tree 的独立只读 FK/J 校准。H8 动力学/求解器、逐指恢复、LOAD_TRANSFER、共同暂停完整回合、每抓法100独立初态配对及新专家/Student仍未完成。原 Teacher 保留默认和回退；不启动 B3/PPO、不更换默认，PR继续 Draft。
