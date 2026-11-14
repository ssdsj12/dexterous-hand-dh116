# Student 独立评估的动作与任务身份

基线为 `5a7eeb41c3d2f4efed98541ad76c375f9a1ee1d2`。Ubuntu CPU 复现了三个独立冷评估缺口：接受未记录发出动作的 tuple，回调就地改写动作时比较的是同一个被改写的张量，以及观测回调在策略输出前推进物理时钟。

现有冷评估在 Student 观测与推理前后比较物理 tick、仿真时间和 policy tick。执行回调收到独立动作副本，必须返回带实际发出动作的 `StepResult`，评估器逐元素核对原始 Actor 24D 输出与实际记录。现有 Teacher 评估仍可读取原生 tuple。原生冷评估报告包含实际任务身份。

最终 Student 准入现在要求声明的任务身份 SHA 与每条原生 480 Hz 实况样本一致，并拒绝跨抓法或回合复用实际物理初态。现有 TensorTask 证据生产者只支持 `rigid_proxy`；Cosserat 声称须等待独立的柔性生产者和实测证据。保持原有 100 初态/抓法、30 mm/5 s/0.5 mm 标准及 Teacher、MPC、Critic 隔离门槛。

在 Ubuntu 现有 Python 3.10.12 中执行：

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_trials.py tests/test_dh116_stereo_evaluation.py tests/test_dh116_stereo_student_cli.py tests/test_dh116_stereo_inference.py tests/test_dh116_stereo_cold_candidate_admission.py
```

92 项 CPU 通过，0 失败，63.81 秒。含 tuple 证据缺失、就地替换、时钟推进、逐样本任务不一致、柔性错标和重复初态测试。私有红/绿日志分别为 `/tmp/dh116-student-proof-X5Y2og/red.log` 和 `green.log`，绿日志 SHA256 为 `5494a9ffb461965b1206565d36faab3d003650806a7c2be860a57b58a6921420`。这些验证没有启动 Isaac、采集或训练。独立 Lift/Hold 原生验收仍待合格模型、接触来源和明确审批。
