# finger12 冻结 Teacher 配对诊断入口

复用现有 `solid_cylinder` 的 TensorWorld、TensorTask 和 TensorEnv。新增独立的 `diagnose_dh116_teacher_pair.py`，在两个独立启动中分别发出同一冻结 Teacher 的 sampled 动作和 deterministic mean 动作。每路固定二指、三指各一个原生槽位，运行 4 或 128 个 60 Hz 控制步，到第一个真实 terminal 即停止本批，保留相应的终止前 480 Hz 样本及未终止槽位。原控制、物理和限位照常运行，PPO 更新数为 0。

实际开始后，每路由现有 reset kernel 恢复同一 authored manifest，并回读原生完整初态。比较双方每手 17 个具名关节的 q/qd、双手根位姿/速度、物体位姿/速度。mean 路要求已执行 sampled 路的 `start_binding.json`，实际回读 SHA、物理/控制器摘要、相机标定、seed、pre-action 371D 摘要和控制器重置上下文均一致才允许第一个动作。每个控制步保留 371D 原始状态与实际 24D issued action，480 Hz 旁路使用已有 joint audit 捕获主动、被动、腕部关节、SDK 目标与真实 terminal。mean 路不额外抽样，sampled latent 和 action 保留空值。原 Teacher checkpoint loader 继续用于实际启动。CPU preflight 单独检查 checkpoint 的物理身份与有限张量，检查历史 CUDA RNG 不需要初始化 CUDA。

## 实际 CPU 预检

以集成基线 `9c7058fd4874c5b34098f32713e4afa3a5f76df3`，在原有 Ubuntu 环境检查真实 update-0 checkpoint `ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511` 和 Cold8 保存的真实 inherited contract。复用 `derive_policy_trial_manifest` 显式重绑定 update-0，8 个原始物理初态完整保留。派生 manifest SHA `9d5b33e6e332f080cd3b2e1f768f50d170c8655ae262a0be75a14199cd255b30`，两路选择 `test:pinch_thumb_index:0000` 和 `test:tripod_thumb_index_middle:0000`。sampled 入口的 4 步预检退出码 0，结果 SHA `addf674a99a87d2fd8c0bf3c44f1a7ee0aa8479b3d1367a41eb6623b0838c1b9`。

历史 MPC 语料 `f4fa715c58ce0d767588a6f2bf5aed248d62d22833d4794a97343baf31b8aaf9` 对真实 Teacher inherited contract 的预检被原 loader 拒绝，错误为 `saved and current contracts must both be declared reset family members`。现有 Teacher 的实际抓取初态 SHA 不属于历史 MPC 的两个 reset family 成员，移除可识别的两个 reset 摘要后，`native_sensor_readback` 仍不一致。阻断日志 SHA `b7a3412a2abeb2611696236f31334e690d016347f6be646fd519ad91aa6c9b73`。两路 Teacher 启动与后续三路物理因果判定保持独立。三路工作仍需受支持的共同场景和真实 MPC 匹配证据。

Ubuntu 相关回归 53 项通过，0 失败。预检和测试均不启动 Isaac，原生配对次数仍为 0。本工作包不授予成功专家池、BC 或 finger12 根因资格。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= DH116_NUM_THREADS=2 \
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_teacher_pair_diagnostic.py \
  tests/test_dh116_stereo_runtime_teacher_identity.py \
  tests/test_dh116_privileged_action_audit.py \
  tests/test_dh116_stereo_candidate_cold_trials.py
```

## Sampled 回合完成证据与三路接入条件

mean 启动前读取 sampled 的 `completion.json`，同时核对首态绑定、原生 480 Hz 账本、策略输入产物的文件 SHA256。账本必须存在两个真实首回合的身份和实际 Teacher sampled 动作 owner。若 sampled 在写入 `start_binding.json` 后失败，mean 在启动前拒绝。CPU 回归增加到 54 项通过，日志 SHA `9be56d303ebc1556d9ba78012e8f25e609c8354858f5fc7f4405df25b23a51d4`。

原 loader 可在历史 T5 的实际 `tensor_contract.json` 下加载同一 MPC 语料，二指、三指两个参考均存在。T5 合同中的 `native_sensor_readback.batched=true` 与当前 Teacher 的 `false` 不同。两个原 Teacher checkpoint 内的 `native_task_contract` 均为 `5bd521d7a2922ea52b1925b26a0546544b7abc399268af19a589507c100c8786`，实际 T5 对应 `cc7d9f525f6ebeb04ff97a2ad0ba9afac02767f897ff3f14558c33c1ae56a0d6`。两条可独立验证的接入路线如下。

1. 保留现有 Teacher checkpoint 的路线需要在当前 exact task contract 下采集新原生成功的同期动作轨迹。用现有 `tabletop_surface_demo_library.merge_datasets` 依据真实来源建立 reset family，并要求 `load_teacher_reference_dataset` 对实际三路继承合同都接受。以同一冻结初态和真实执行旁路验证原 `_teacher_optimizer` 的 reference、alive、tick 和 `observe_result` 边界。
2. 保留历史 MPC 的路线可基于历史 T5 的合格物理参数、传感器读回方式及复用原 MPC，使用独立的新 Teacher checkpoint。该新 checkpoint 从训练、保存到评估都绑定实际 legacy TensorEnv 合同。先用真实 2 环境回读核对 reset family、控制器和物理摘要，再进行 4 步三路 pilot，最后运行 128 步因果诊断。该路线的结论只适用于新 Teacher；现有 update-0/1 和历史越限归档保留为独立证据。

## v02 原生启动失败及修复

实际 sampled 4 步 pilot 在执行策略动作前退出码 2。实测继承合同中唯一差异是观测归档的 `rollout_steps=32` 与启动器的 `rollout_steps=4`。原始日志 SHA `f730e2a92efa658700ab72018a0a4a52c28cdf91d5c8d6655d38e77d12dc761f`。原生冻结模型身份检查继续使用完整未改动合同。

启动器现在沿用实测合同的 32 步配置，诊断回调仍独立限制 4 或 128 个控制步。原生回调返回前 flush/fsync 真实 480 Hz 账本、首态绑定、输入记录和 `completion.json`，外层自然退出证据继续由 supervisor 单独记录。CPU 回归增加到 55 项通过，日志 SHA `f3305967ace4567b1b556f81116ba0d4a02f7ccff39873c0d0149f71ab934c2c`。修复后的实际预检确认 32/4 两个预算，待独立原生复验。

## v03 原生合同编码核验

v03 sampled 启动在首个动作前退出码 2，日志 SHA `4e80bf0816633b441db118afb206ee1714c3689ed75a2051569d94aa17964ba8`。冻结目录中的 6 个输入 SHA 全部匹配。v03 的实际归档继承合同和任务身份与 Cold8 归档的 canonical JSON 逐字一致。先前使用 Python 原始容器相等性检查，运行时 tuple 会与 JSON 读回的 list 不等，即使持久化语义相同。

回调现逐字段比较 canonical JSON，不移除任何字段。真实限位数值变化、任务身份变化和输入文件改写分别拒绝。CPU 回归 56 项通过，日志 SHA `3018a198298de5b4e842f35a0df965513d1f67aabddb66e4a03fa824531b1d88`。v02、v03 结果目录和退出证据均保留，修复仍待独立原生复验。

## 各槽位首回合独立终止

v04 的二指首先触发位置保护，三指只保留 272/280 个物理子步前缀。新增显式参数 `--episode-stop per_slot_first_terminal`。二指或三指首回合终止后，原槽位发出 `diagnostic_zero` 的实际全零 24D 命令。新 reset episode 的原生样本、初态身份和 371D 状态留在首回合诊断之外。另一个槽位继续执行原冻结 Teacher 动作，直到自身首回合终止或达到既定 4/128 步上限。首终止发生在最后一个槽位时停止整批。原默认行为 `first_batch_terminal` 继续可用。

配对两路的终止策略需要一致。CPU 用双槽位桥接测试模拟真实首终止的账本边界，覆盖零命令约束、退休槽位随后再次终止、原终止样本保留、活跃槽位继续执行及身份隔离。真实归档上的 CPU 预检确认 opt-in 128 步、原生启动配置 32 步，预检 SHA `04f7a3e894fb3ac970a6face02091244b6b5716854ec889621faf08e105418c5`。128 步只覆盖约 2.13 秒策略时长，原生成功和完整三指回合仍需要实测。

Ubuntu CPU 回归 57 项通过，日志 SHA `e3de17d512969a563ca4e39a37eaedc0d352e3475f65ddfd88c842eaafa5be0c`。本工作包原生运行次数为 0。
