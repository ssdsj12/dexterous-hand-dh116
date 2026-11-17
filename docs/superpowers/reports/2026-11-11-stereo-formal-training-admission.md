# 正式 BC、DAgger 与 Student PPO 准入复核

状态为 `blocked`。Teacher128 是 2 环境、128 步、1 次更新的原生诊断，完整成功回合数为 0。该 Checkpoint 和失败回合不进入成功专家池。已保存 Roll20 三指正确三维点完整率约 4.33%，感知资格仍未通过。8 初态同步采集继续等待单独批准。

沿用已有 BC 成功池、DAgger 准入、Student 自有 PPO rollout 和冷重置候选隔离。CPU 执行了 `python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_bc_mode.py tests/test_dh116_stereo_dagger.py tests/test_dh116_stereo_student_ppo.py tests/test_dh116_stereo_cold_candidate_admission.py`，42 项通过，0 失败，24.24 秒。包含未资格 Teacher 拒绝、失败状态不升级、reset 边界、Actor 排除真值和候选准入。未启动正式训练。

下一步先验证经批准的同步候选采集和原生状态身份，再为独立视觉监督及完整 Teacher 成功轨迹建立资格。继续使用现有 30 mm Lift、5 s Hold 和 0.5 mm Slip 标准。
