# Pinch 物理账本的等字节编码

Pinch 数值 Reset 原生复验完成 234 步、自然退出 0，耗时 203.793369 秒，超过 180 秒。原始物理账本 621,194,597 字节，232 个物理行，独立审计重新验证了所有 232 个 v2 packet 和 116,676 个原始 contact pairs。实验状态保持失败，修复不能追溯修改该回执。

`tabletop_physical_evidence_encoding.encoding_scope()` 复用原账本和 JSON serializer。只加速已持有的有限数值 NumPy 数组和标量，继续保留 dtype、非有限值标记、带符号零、字典字段顺序以及原 JSON 输出字节。Scope 退出恢复旧 binding，原 journal `_Sequence`、terminal flush/fsync 和 guard 全部继续执行。优化仅为后继独立配方提供，不改变旧 worker 或配方指纹。

CPU 原账本完整对照以 SHA256 `6bab9bfbc12a61164b56f800daf7159f6f53572556b866a4152f1562cbc0c364` 的 295 个 payload 行为输入。逐行分别用原编码和新编码生成 JSON，并要求字节完全相同。保留的数值证据如下。

| 项目 | 实测 |
| --- | --- |
| 物理行 | 232 |
| 原编码累计时间 | 33.8845 s |
| 新编码累计时间 | 6.7839 s |
| 本次 CPU 对照时间差 | 27.1006 s |
| 对照产物 SHA256 | `b3f7fbf548ec1c6d66eb1545b2fdce9a2e230a21575cd9fea70c3d6b3e9762af` |

独立 Ubuntu CPU 回归 19 项通过。

```bash
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_physical_evidence_encoding.py tests/test_dh116_dual_hand_evidence_journal.py
```

全量对照是 CPU 编码计时，不能保证原生启动、控制和 teardown 的总耗时。新原生配方仍需满足相同的 180 秒及原先的全部物理和来源检查。该包未执行新物理或 PPO。
