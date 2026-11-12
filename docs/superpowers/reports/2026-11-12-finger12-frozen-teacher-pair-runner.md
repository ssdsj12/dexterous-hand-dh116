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
