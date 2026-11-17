# Pinch 等字节账本复验配方

数值 Reset 配方完成 234 个 SDK 步，实际 Reset 两份数值快照的 23 项完全一致。完整数据包含 232 个 v2 packet 和 116,676 个 contact pairs。外层及 inferior 均自然退出 0，203.793 秒超过 180 秒，旧回执保持失败。

后继配方仅在该 worker 的物理账本写入中使用已验证的等字节编码。原始 621 MB 账本全量 CPU 对照确认编码结果逐行字节一致。33.8845 秒降至 6.7839 秒是 CPU 编码时间，原生总耗时由本配方独立测量。

使用 GPU 3、单实例、1 环境、234 个物理步、180 秒及 0 次 PPO 更新。动作、时钟、Reset、保护判据、数据源、接触与独立退出审计按原合同执行。保留已消耗 v2 指纹和私有回执。新调度与物理入口分别绑定真实输入。

| 指纹 | SHA256 |
| --- | --- |
| 配方 | `341a3ea1fb256474e1a1d1a65a2ad70238bc0e9da82f011d04285dbbb18ecf64` |
| 调度代码和配置 | `f1702ac5d5945edc580c11e25d58a8ce11ab16cc62076063715c3012f4170de0` |
| 物理入口代码和配置 | `e859f079f1e6eec1803af2da3dfd9490ee12091552335cce1599eb3523c2f024` |

Ubuntu CPU 回归 34 项通过，日志 SHA256 `09d15915d87d64a9509e27e2268865e14a2f78a598898022bfa7aef21101bd23`。

```bash
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_pinch_fast_evidence_worker.py tests/test_physical_evidence_encoding.py \
  tests/test_mesh_reset_numeric_worker.py tests/test_dh116_dual_hand_evidence_journal.py
bash scripts/python.sh scripts/run_dh116_pinch_fast_evidence_work_packages.py \
  --run --scope all --limit 1 --jobs 1 \
  --evidence-dir "$DH116_PRIVATE_EVIDENCE" \
  --native-approval "$DH116_FROZEN_BATCH_APPROVAL"
```

原生结果和资格单独发布。
