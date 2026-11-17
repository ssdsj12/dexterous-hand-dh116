# 原生 Reset/R0 账本编码优化（CPU 对照）

基线 `f8e890705ad90631b4869d8d68ab4060681e0de8`。只优化诊断数据编码，不改变物理、动作、奖励、字段或文件持久化。原递归编码实现保留，新增独立 opt-in scope，退出和异常恢复原绑定。

## 实际改动

- `tabletop_fast_owned_codec.encode_owned/codec_scope`：精确 builtin 类型与有限 ndarray 走等价快路径；非有限值、扩展类型及非法对象继续使用原处理。保留 NaN/Inf 标签、dtype、负零、数值精度、validity、所有行。
- `profile_dh116_live_r0_observer.run`：完整编码流 SHA。
- `compare_dh116_owned_codec.compare_results/main`：同一不可变原生账本两个 CPU 路径，比较全部输出流、计数、阻断项、来源身份。剖析不是新原生采集。
- 测试覆盖标量/数组/torch/stride/endian/未知值/所有权/作用域恢复、原 R0 CPU owner 的回归和无原生晋级。

## 证据与失败保留

来源原生 receipt SHA：`e92282a2a01596b551342ac56cd046f4f5a0c289eda51cf76578b8712e4d89d1`。
此源运行自然 exit=0，但 187.245 秒超过 180 秒，仍为失败，不晋级。

原始证据目录（Ubuntu，不上传原始数据）：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/fast-owned-codec-20261114`。

1. 初版仅跳过数组标量递归：compare01 为 46.4268 → 46.5925 秒，没有改善。失败结果保留。
2. 标量调用回归首次失败：4099 次旧实现调用，预期最多 1 次。消除二层分派后通过。
3. compare02：相同 1 Reset、29 begin、232 physical、29 end；59,948,724 编码字节完全相同。
4. 编码流 SHA：`9fbbe47b94212874613d520b470a5c527403242c248f4117614f783bf1643f37`。
5. CPU 剖析耗时 46.199173 → 27.658837 秒，比率 1.6703。含 cProfile 开销，不代表原生墙钟；没有测量磁盘 I/O。
6. R0 阻断项逐项相同，Teacher admission=false，training reward 未启用。

## 测试与命令

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_owned_codec_comparison.py tests/test_live_r0_profile_scope.py tests/test_fast_owned_codec.py tests/test_deformable_r0_replay.py -q`

**126 passed**，原始日志 `regression02.log`；此前红测试保留。

对照：`bash scripts/python.sh scripts/compare_dh116_owned_codec.py --receipt <原始 receipt.json> --receipt-sha256 e92282a2a01596b551342ac56cd046f4f5a0c289eda51cf76578b8712e4d89d1 --output <独立 compare02>`。600 秒 CPU 上限，0 新物理步、0 PPO。

## 资格与下一步

本包只通过 CPU 字节等价与同账本性能检查。原生是否能在 180 秒内完成尚未验证。下一包将独立冻结优化后入口和外层审计，在站立授权范围内一次执行 234 步 / 1 环境 / 0 PPO / 180 秒，保留原失败入口。有效承托、真实稳定性、approach、slip 和正式 R0 仍未全部合格，不能开始长 PPO。
