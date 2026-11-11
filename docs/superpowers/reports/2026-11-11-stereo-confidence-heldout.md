# WP1 已归档候选的分组置信度审计

基线 `5a7eeb41c3d2f4efed98541ad76c375f9a1ee1d2`。审计只读取已有 Front 与 Roll20 CPU 复放的候选池，每份摘要及其八份帧档案通过 SHA 校验。通过已绑定的同像素三维 ≤5 mm 标签比较分数。标签只在审计中读取，无有效参考的点保持 NA。现有采样器、默认匹配器、相机合同和模型权重保持原样。表面点只作为实测表面样本。

物理初态 SHA 与来源族用现有 `split_estimator_rows` 的连通分组，重复复放同一初态仍在同一分区。分数按相机标定、分割 Checkpoint、抓法和深度来源分别审计。原始匹配分数有界，但不具备概率语义。Brier 和 ECE 仅用于诊断将其视为概率会产生的偏差。每组等权，组内重复帧不增加组权重。

当有足量独立组和实际双侧证据时，固定分箱只用训练分区估计各箱正确率。验证和测试分区只评分，未见分箱返回 NA 并单报覆盖率。分割结果不能替代右侧原生 ID 或参考可见性。同源稠密深度与 RGB 软补测不能混池拟合。本工具不发布阈值、模型或晋级记录。

## Ubuntu CPU 实际结果

8 帧，共 1,124 个候选记录。Front 和 Roll20 虽有不同相机标定和 episode 族，但复用了相同两个物理初态。故实际连通组仅 2，无法构造独立的训练、验证和测试分区。8 个点缺左参考，全部 6 个分层的有效右参考计数为 0，无拟合、无 held-out 误差、无部署资格，晋级 0。Roll20 原色正确点 91/89 全部保留，Front 6/0 全部保留。本次未修复上一轮记录的 48 条固定尾部。

| 相机 / 抓法 | 来源 | 候选 | 左参考 NA | 左参考 ≤5 mm | 原始分数诊断 ECE |
|---|---|---:|---:|---:|---:|
| Roll20 二指 | 已有稠密深度 | 216 | 0 | 98.15% | 0.019 |
| Roll20 二指 | RGB 软补测 | 132 | 0 | 33.33% | 0.481 |
| Roll20 三指 | 已有稠密深度 | 200 | 0 | 96.00% | 0.040 |
| Roll20 三指 | RGB 软补测 | 136 | 0 | 11.76% | 0.593 |
| Front 二指 | RGB 软补测 | 268 | 8 | 12.31% | 0.625 |
| Front 三指 | RGB 软补测 | 172 | 0 | 11.63% | 0.671 |

相同原始分数不代表相同正确率。下一步需单独批准的同步原生来源提供左右实例 ID、参考深度和独立物理初态，再做分组 held-out 校准。CPU fixture 可检查训练分箱与评估标签隔离，不授予原生资格。

```sh
export PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
p=/tmp/dh116-wire-retention-lf-g4HBlj
/mnt/mydisk/xk/envs/dh116/bin/python scripts/audit_dh116_stereo_confidence.py \
  --replay "$p/front/summary.json" 3918698a06146363427ef1f070b9befb73e812a33799c61d0801fc08d165b413 \
  --replay "$p/roll20/summary.json" c389bed11e2ac652f58533abe9da7cbe0d00bb12e519a17ad06dc291717655cb \
  --output <new-private-output>
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_stereo_confidence_audit.py tests/test_dh116_stereo_wire_sample_retention.py \
  tests/test_dh116_stereo_wire_correspondence.py tests/test_dh116_stereo_wire_depth.py \
  tests/test_dh116_stereo_wire_geometry.py tests/test_dh116_stereo_wire_tail_audit.py
```

Ubuntu Python 3.10.12 上 48 项 CPU 测试通过，0 失败，6.04 秒。输出 SHA256 `9f3682f19b33940b7adff9b76176d14a5da92c864302f1ae7165384695945b52`，审计和先失败后的回归日志保存在服务器私有目录 `/tmp/dh116-wp1-confidence-VZeuCk`。本工作包未启动 Isaac、采集或训练。
