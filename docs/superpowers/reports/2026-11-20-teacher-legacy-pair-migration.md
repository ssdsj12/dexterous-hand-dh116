# 历史配对证据迁移修复

数据盘迁移后，配对 Reset 的历史路径仍能定位到文件，但经过符号链接，阻断了左侧 Pinch 候选重建和旧 Tripod 八项交换回归。核对 canonical 路径与原文件 SHA `996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37` 后，将实际读取绑定到数据盘绝对路径。历史配置的来源键保留原字符串，冻结配置逐字段一致。通用 canonical 路径检查和哈希检查照常执行。

CPU 定向结果 49 通过、1 项单独回归，日志 SHA `7d0c5532ed1da891f2b06c18b8ebd6c44c3e4eab6ef16cb03a41408068ee9d4b`。命令 `python -m pytest -q -p no:cacheprovider -k 'not test_independent_entry_has_exact_parent_and_closed_budget' tests/test_dh116_pinch_margin_contact.py tests/test_tripod_exchange_admission.py tests/test_tripod_quiet_gate.py tests/test_dh116_tripod_right_tracking_native.py`。新增回归修复前失败，日志 SHA `a07fa56a235f78197f4405be691afb27793d9050189869b05c02dd8782cdc145`。包含该项的扩展回归 49 通过、1 失败，日志 SHA `5cde87ebb7e0e8d293d69e5868489b25d0436718a274a6f9c1f8baca7d009dc7`。失败来自更早的 parameter-task 下游原生父回执，读取路径同样经过符号链接。该回执和它绑定的工件需要独立迁移核验，旧工作包的准入暂沿用失败状态。

本工作包无 Isaac/原生/PPO 运行，未修改证据文件、策略、姿态、资产或成功门槛。
