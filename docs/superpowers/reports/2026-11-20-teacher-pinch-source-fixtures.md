# Pinch 来源审查与 CPU 夹具修复

本轮扩大回归覆盖 1219 例。初次运行 1203 通过、16 失败，日志 SHA `5cfb13664ea44904bf68b5386465aef94bcb73d5dfad45cad2cf5d931b7fde1e`。以下复跑在稳定源码上完成。

- 12 个 Pinch 真实适配器 CPU 用例未传入新增的观察器计时回调。修复测试调用以显式关闭计时，复跑同文件 16 项全部通过。原生控制器未改动。
- 旧 R08 Reset 正例错误地期望可变工作树仍与封存来源一致。测试现在从 Git `54bae79365886d37e68a8d1eaf3ddeeea8eeb387` 提取文件到临时隔离目录，逐字节核对原 SHA，然后复读实际原始 Reset 账本。畸变测试仍覆盖 seed、杆状态和回执哈希。该目录只用于 CPU 测试，未经改变的现行运行时继续严格拒绝不匹配的来源。复跑 63 项通过。
- 初次长回归期间修改了 Pinch 测试文件，导致 3 个 checkpoint 审计用例的源码快照过期。固定测试文件后，本次复跑这三项及物理账本对照共 4 项通过。没有跳过、降低或重写封存哈希。

复跑命令为 `python -m pytest -q tests/test_native_teacher_policy_pinch_distance.py`，`python -m pytest -q tests/test_native_teacher_preflight.py`，以及 `python -m pytest -q tests/test_native_teacher_worker_checkpoint.py -k 'reference_is_unchanged or exact_integer_checks_preserve or fresh_physical_audit'`。测试 SHA 分别为 `7165b8de1dd263613c9495d4596bdff069fe8739da52401f0644008bd863db2c`、`8e29befca7e12bea6fe1076468b6f0f391d15290eca2b3103c0ba0f8339688ad` 和 `4b064a4991368bf0350e4a17ac8c2f14635a5d17d6e4374d5596efaf44541b0a`。本工作包未新启动 Isaac。
