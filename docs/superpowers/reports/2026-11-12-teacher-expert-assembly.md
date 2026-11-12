# Teacher BC 原生专家数据装配

`scripts/build_dh116_privileged_teacher_expert.py` 使用有 SHA256 绑定的目标 Teacher identity、输入清单、原生资格证据和原始同步账本。沿用完整首回合原生证明验证器与 30 mm／5 s／0.5 mm 门槛。输入清单 `dh116-privileged-expert-assembly-v1` 的每个 episode 包含 `episode_uid`、`source_path`、`source_sha256`、`qualification_path`、`qualification_sha256`。源账本使用当前 `observations`、`privileged_valid`、`actions`、`sample_metadata` 格式。装配重新核对外层退出证据的输出清单，逐控制步比较物理执行账本的 run、episode、env、策略步、原始 480 Hz 步号、终止标记和实际下发的 24D 动作。整个候选轨迹的 371 维均需有效。成功输出中的原生证明与源账本仍在实际 Actor-only BC 阶段重新验证。

```bash
bash scripts/python.sh scripts/build_dh116_privileged_teacher_expert.py \
  --manifest "$reviewed_inputs" --manifest-sha256 "$input_sha" \
  --model-identity "$target_identity" --model-identity-sha256 "$identity_sha" \
  --output "$new_assembly_run"
```

Ubuntu CPU fixture 测试验证有效性缺失、逐物理步动作偏差、原生证明拒绝和 SHA 错误都会拒绝候选；通过序列化测试的 mock 仅用于 CPU。正式专家资格仍依赖真实逐步物理证明、真实完整成功和实际 371D 来源。外层未记录产出的旧数据、短诊断、失败和不完整回合没有资格。本包未启动 Isaac，成功专家池没有新增实际样本。
