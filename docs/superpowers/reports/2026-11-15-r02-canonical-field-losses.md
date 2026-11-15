# R02 canonical field losses 修复报告

日期：2026-11-15。状态：Ubuntu 本地实现与 CPU 验证完成，待主代理执行独立 SPEC/QUALITY 审查、集成和 Windows 发布；本工作分支未推送。

工作树：`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r02-loss-20261115`。
分支：`codex/repair-r02-loss-20261115`。基线：`7a648ce035599f3244370f25f15a4d2a6fa29403`（含已发布 R01）。

## 根因与结果

`StereoEstimationLoss.forward` 原先将 canonical 371D 中的分类字段也送入 Smooth L1，而部署 adapter 使用 sigmoid 解码。例如字段 149、label=0、logit=0 时，旧损失和梯度均为 0，但解码概率为 0.5。正确 BCEWithLogits 损失为 `log(2)=0.693147...`，该坐标梯度应为 `0.5-label`。独立 state supervision 先前已使用 BCE，两条训练路径存在数学分叉。

新增 `StereoEstimationLoss.canonical_state_loss(prediction, target, mask)`，统一非分类标量 Smooth L1、分类 BCEWithLogits 与完整四元数组 sign-invariant geodesic 损失。分类坐标为 149–152、255–258、303–304、309–318，共 20 维；从已有 canonical 合同生成分类分区。helper 在切片前验证 [B,371]、target shape/device/floating dtype 和 boolean mask，并复用现有 `_safe_pair` 与 masked 数学函数。

`forward` 仍先执行 `state_allowed`、逐来源 fail-closed mask、未知/NA mask 和 `rotation_symmetry_valid` 处理。`_prediction_loss` 复用共享 helper 一次，移除了重复训练项；分类 sigmoid residual、四元数弧度指标保留。force 资格验证、额外 logits 与回归 heads 的计算保留。

损失版本为 `dh116-stereo-canonical-field-loss-v2`：暴露 `STATE_LOSS_VERSION` 与 `criterion.loss_version`，并写入 BC step 报告、BC 合同 config、state supervision 模式报告以及新保存的 `state_estimator.pt`。BC 中间 Checkpoint 内的合同同时携带版本。新版本改变新训练的合同身份；原始 371D 接口、24D action/Tanh、模型参数布局及旧模型预测不改。

## TDD 与回归证据

新增测试 63 项，包括 40 个分类坐标×label0/1 组合的 BCE 数值、`0.5-label` 梯度、非选中坐标零梯度与 standalone 一致性；未知 NaN 的零损失/有限零梯度；regression116 的 Smooth L1 尺度；四元数符号不变性与不完整组排除；匹配 shape/dtype/mask 验证；未资格化旋转、NA、force gate；额外 head 数学；报告、合同和新 Checkpoint 版本。

RED 在任何生产代码改动前运行，真实得到 57 failed / 6 passed；其中 40 个分类用例均在 BCE 数值断言失败，非仅缺失接口。首个失败为字段149、label0：得到0，预期0.693147。第一次 GREEN 新测试为63 passed。

完整回归首次得到167 passed / 1 failed，失败位于 `test_privileged_quaternion_groups_never_receive_raw_component_huber`。该旧测试对所有371维设 valid，因而同时引入20个正确 BCE 项，原来的总损失0断言不再成立。经主代理明确扩大必要文件范围，仅将该 quaternion 专用测试的 mask 收紧为所测64、156、163起始的三组完整四元数，保留原有符号、零角度和有限梯度断言；生产数学没有为兼容旧断言而退回。原始失败日志保留。最终指定回归为168 passed / 0 failed。

证据目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r02-canonical-field-losses-20261115`。每个日志对应同名 JSON，记录实际完整命令、环境、基线、耗时、退出码及日志 SHA256；日志和模型 fixture 产物不纳入 Git。

| 证据 | exit | 实际结果 | 日志 SHA256 |
|---|---:|---|---|
| protection-before.log | 1 | 首次保护命令转义失败，Python SyntaxError，未执行保护校验 | `586df38225604abfb1eb9a56174039f90d6633e64302bee7c541bf0e71938141` |
| protection-before-valid.log | 0 | 528 个受保护源码/脚本/配置及 2 个历史 Checkpoint 校验通过 | `ed46bf8180a04dee9cb9dca2afc1d7699400b7036183add2d17ae306208ac5f7` |
| red.log | 1 | 57 failed, 6 passed；40 个分类数值断言全部真实失败 | `b4825fccbff24bcaefdcf628b6073bf4051dcbc42bb3c72615caa33f3262850a` |
| green-new.log | 0 | 63 passed | `3fc52457fc23a7ab3e5e86fe5ce6183d1c4d72980cdfc87bef704a7d3f8b24e4` |
| green-regression.log | 1 | 1 failed, 167 passed；旧 quaternion 测试混入全分类字段 | `a893b9d0b21cf6801abd6ce9ed08ff6073b2abbfbff69bf97c239de967b66244` |
| green-regression-final.log | 0 | 168 passed | `d6f8371ec4a5a4fc83917ad511ed86011089e40b5e51821d5aa2fcabb8679bc9` |
| protection-after.log | 0 | 528 个受保护源码/脚本/配置及 2 个历史 Checkpoint 校验通过 | `ed46bf8180a04dee9cb9dca2afc1d7699400b7036183add2d17ae306208ac5f7` |

最终回归命令（Ubuntu、现有环境、CPU）：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_canonical_field_losses.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_runtime_task_state.py tests/test_dh116_stereo_bc_actor_inputs.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_training_cli.py
```

RED 和单文件 GREEN 使用同一环境、命令前缀，只运行新增测试文件。保护检查通过 `scripts.run_dh116_work_packages.Repository(Path.cwd(), evidence_dir=Path('/home/xk/.local/share/dh116-autonomous-3c80166/evidence')).verify_protection()` 执行；未在工作分支运行 `verify_local_target`。首次保护 invocation 的 shell 转义错误另存失败日志，修正为私有脚本后前后两次保护检查均 exit0。

## 范围与限制

提交范围为4个生产模块、新增canonical测试、必要的旧quaternion测试窄修，以及本报告，共7个文件。README、CHANGELOG、registry、readiness、manifest由主代理统一处理，本分支没有修改。

仅执行 CPU 单元与集成 fixture，包括一次受控 BC update 和 diagnostic tensor fit 以验证实际保存路径；这不构成原生训练、原生相机/物理资格、泛化、Actor 部署或抓取成功证据。未启动 Isaac/native/GPU/正式训练，未修改现有 Python 环境、历史模型、Checkpoint、原始数据、成功判据或 R01 Actor features helper。最终 `git diff --check` 通过；528+2 保护校验通过。发布结果须以主代理后续 Windows 核验为准。
