# 子步时间来源显式区分（2026-11-14）

后继于 `68323f1754041538e52f0813db950b88c3008af3` 的 CPU 子步姿态审计。此前字段 `substep_time_s` 是 SDK 步号和 480Hz 公式推算，名字未明确区分测量与推算；现在改为 v2 `clock_provenance`，不会把插值时刻说成实测 timestamp。

实际修改：
- `tabletop_deformx_substep_pose.nominal_substep_clock` 输出 `nominal_substep_time_s`、原样保留的 `endpoint_timestamp_s` / `endpoint_solver_time_s`，以及显式 `measured_substep_time_s=None`。
- `audit_dh116_substep_contact_frames.run` 按每条真实 pair 保存上述来源标记；旧 v1 结果文件只读保留。
- `test_substep_clock_provenance.py` 覆盖不同时间原点、计数类型、非法时间、输入不被改写。

端点 timestamp 的来源是原账本 identity；不重新解释它是实际 SDK callback 或 wall-clock 时间。solver_time 是原账本保存的 solver 时间，未投影到 SDK n/480。该 run 在两步 SDK 初始化后才从 solver 零点开始，两种时间不能直接互换。

真实历史整份离线重放再次核对 **232 行/2,958 pair**。逐条 identity、所有旧区域计算与计数均完全不变（888步末候选、872子步候选，16分类变化）。最后一条记录：
- nominal 子步时间及原 endpoint timestamp：0.4875 s；
- 原 endpoint solver_time：0.48333296179771423 s；
- 实测子步 timestamp：NA。

CPU：**29 passed（7新+22既有），0 failed/0 skipped**。无新原生、无 solver 调用、无 PPO。时间命名纠正不授予 Reset、正/负有效接触、深度或 Teacher 资格。

命令：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q \
 tests/test_substep_clock_provenance.py tests/test_deformx_substep_pose.py tests/test_substep_contact_audit.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh \
 scripts/audit_dh116_substep_contact_frames.py --output <new-independent-v2.json>
```

私有 evidence 根目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence`。
- 新结果 `projected-pinch-substep-clock-v2-20261114.json` SHA `5884ae4246e822c9138b905562838ded350fa3adc7205c61a71e4c70323c87fd`。
- 保留旧结果 SHA `4955ce70bc491fb0e9908dcdecae936a5a7196525c1b76ca537aec59a0732597`。
- 首次失败 log SHA `3ebb1ac89742091f65b47ae21c62a1a5ebb90ca9a470de5fa291e47be180879c`。
- CPU log SHA `f09b9bfe87d88a1f62a3a7a55e5c1ddd33d8c03913f5842fb781d9a1855c765c`。
- 全重放 log SHA `20d3bf70eb07b546906350562a777093eb91a0f1e99c5f0cf7e0fc1ce83d6a22`。

下一项仍是新后端直接姿态/接触包和真实 Reset 原生验证；不得以这份离线数据代替。
