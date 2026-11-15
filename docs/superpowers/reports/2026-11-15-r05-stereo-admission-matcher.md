# R05：Student 与 collection 共用 matcher 准入比较

日期：2026-11-15。实现基线：`1e956caf1816232841a3efba9f76c61c76859e12`；隔离分支：`codex/repair-r05-admission-20261115`。

## 问题与改动

原 collection 已比较双目标定及 matcher，Student 的 `_formal_admission` 仅比较 calibration SHA。同一内参/标定下更换 matcher 或省略 matcher 元数据时，Student 可以越过该配置准入检查。

新增 `tabletop_stereo_admission.require_qualified_stereo_processing(verified_stereo, camera_config)`，两个正式入口在既有 Teacher 和 Stereo proof verifier 成功后调用。共享函数从实际相机配置构造 `processing_contract`，比较 calibration SHA，并对测量 proof 的 `depth_estimator_config` 深拷贝，仅删除 `image_processing_batch_size` 后，与实际 `depth_estimator` 作完整字典比较。不存在第二个忽略字段列表，不删除置信度、视差、代价、预处理或 segmentation 参数。缺失、非字典、字段/值变化及额外字段均按 matcher 差异拒绝；资源批量 1/99 可通过，输入 proof 不被修改。

保留 Student 显式 Teacher proof、实际 Teacher checkpoint、Stereo proof、选定 calibration，以及 actor-package warm start 或 Student resume 的要求。初始化准入只确认显式提供来源，实际包文件和合同仍由后续加载、`_warmstart_processing` / `require_raw_processing` 校验。既有处理合同、在线 DAgger 时序、R02 loss-v2 和 R04 sealed source 逻辑未改动。

`tabletop_stereo_preprocessing.py`、原配置、371D/24D、PPO/Tanh、历史 checkpoint 和成功标准不在修改范围内。共享 helper 是验证后的配置比较，不创建原生 proof，不授予原生资格。公共 README、CHANGELOG、Registry 和 readiness 由主代理另行处理，本提交仅五个文件。

## TDD 与 CPU 验证

所有 pytest 均在 Ubuntu 上通过既有解释器 wrapper 运行；没有修改环境或安装依赖。命令前缀：

```bash
env DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= bash scripts/python.sh -B -m pytest -q -p no:cacheprovider
```

- RED（修改生产代码前）：上述命令加 `tests/test_dh116_stereo_admission_matcher.py`；exit **1**，**17 failed, 38 passed in 2.23s**。全部失败均来自 Student 的 `DID NOT RAISE`，包括换 subpixel、换 method、缺 matcher；collection 对应对照通过。这些是预期的缺陷复现，原始失败日志保留。
- GREEN：同一命令与同一新增测试文件；exit **0**，**55 passed in 2.43s**。
- 指定回归：上述前缀加以下五个文件；exit **0**，**108 passed in 39.40s**（新增 55 项、既有 53 项），无失败或警告。

```text
tests/test_dh116_stereo_admission_matcher.py
tests/test_dh116_stereo_training_cli.py
tests/test_dh116_stereo_student_cli.py
tests/test_dh116_stereo_preprocessing.py
tests/test_dh116_stereo_student_ppo.py
```

新增测试还覆盖实际相机配置在相同 calibration SHA 下变更 subpixel、默认 matcher 的全部 11 个非资源字段、非法字符串视差、额外字段、资源批量及 proof 不变性、校准差异、proof 拒绝传播顺序，以及 Student 的显式 proof 与初始化要求。配置隔离用例只 mock 下层 Teacher/Stereo qualifier，真实入口、真实 processing contract 和新 helper 均未 mock。另有两个入口分别恢复真实 Teacher 或真实 Stereo verifier 的负向用例，确认 summary-only 输入仍被拒绝；这些负向 CPU 用例不构成原生资格证据。

实现前、指定回归后分别执行以下保护校验，均 exit **0**，结果为 **528 protected sources + 2 historical checkpoints**：

```python
from pathlib import Path
from scripts.run_dh116_work_packages import Repository
repository = Repository(Path.cwd(), evidence_dir=Path('/home/xk/.local/share/dh116-autonomous-3c80166/evidence'))
repository.verify_protection()
```

使用 `verify_protection()`，没有在隔离分支调用要求发布分支的 `verify_local_target()`。

## 私有证据索引

证据目录（不进 Git）：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r05-admission-20261115T141500`。同名 `.json` 保存实际命令 argv、cwd、exit code 和日志 SHA256；原始日志保留在该目录。

| 日志 | exit | SHA256 |
| --- | --- | --- |
| `red.log` | 1 | `a6045af36cfec5fd71abca04feae6de26860a8ba60b1e32c856a9a0c487b3a55` |
| `green.log` | 0 | `69c3db8a5aa5956704d8b6f02d781a42f8ab7bda5a15f6cb8f52c7b4d0c74112` |
| `regression.log` | 0 | `57e713e7a2f5047c57b73b0f689b6206985d886236690d349c3d3b5d3803a800` |
| `protection-before.log` | 0 | `43a75e539c20bb1be4f60f6cf0906eb1e7759e8c5ad1217fcada19312c950d9e` |
| `protection-after.log` | 0 | `43a75e539c20bb1be4f60f6cf0906eb1e7759e8c5ad1217fcada19312c950d9e` |

## 验证边界与交付

CPU 配置准入修复和指定回归通过；当前失败项为零，RED 的 17 项历史失败仅证明缺陷复现。未启动 Isaac、native、GPU、训练或实际抓取，没有消耗原生重试预算。没有修改历史数据、模型、运行环境或共享设备状态，没有推送。本地提交交主代理进行独立 SPEC/QUALITY 复核与 Windows 发布核验。
