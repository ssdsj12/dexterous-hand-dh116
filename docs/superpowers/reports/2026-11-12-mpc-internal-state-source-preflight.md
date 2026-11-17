# MPC 371D 内部状态的同钟读回

已有 v02 原生旁路能验证外部运动学、已提交命令和上一完整腕部区间。原始 sequence context 未记录下面的内部任务源。本包增加可选 `--capture-internal-source`，用于后续独立运行。

旁路在原 `TensorEnv.reset()` 返回及原 `TensorEnv.step()` 返回后直接读取 `TensorTask` 的命名张量及 `SurfaceFeedback` 的原始累加器。读取前后 World／SDK 时钟必须一致，原 371D 由原采集器保留。本包的旁路不调用 `observe()`、不取 `SurfaceFeedback.features()` 和 `_features`，不改变任务状态、物理或 issued 24D。读取的原始源按原执行账本的环境、episode、相机帧、动作和时钟加入输出。每个成功、失败、未完成归档只保留对应行。

| 索引 | 同钟命名来源与处理 |
|---|---|
| 36–47、52–54、56–59 | Reset 参考、原任务计时器、闭合进度、descent stop，按原归一化逐值比较 |
| 74–77、88–99 | 原抓取参照及其 flag，原控制器 residual。抓取参照仅在原 `grip_valid=true` 时记为可证 |
| 121–130 | 原力控制命令和历史量，保存命名原始值，当前有效性保持 false |
| 131、144–146、185–190 | 旧 slip 历史、接触时钟、ready/scoring flag、原请求腕旋转，按原单位复算 |

`internal_state_value_valid` 最多覆盖 45 个模拟器内部值。该 mask 与原 `privileged_valid` 分开，模拟器命令、时钟和内部值采用模拟器语义。

`reset_feedback_initialization_value_valid` 只核对原 367–370 的 reset 初始化值。要求原全环境 reset 刚返回，并从实际 `_valid`、`_open`、`_ok`、`_window_good`、`_elapsed`、`_sum`、起末腕位和其他累加器验证已清零，且原记录对应输出为零。普通 step 即使观察值为零也保持 false。这个证明不代表 reset 存在上一物理区间。351–366 仍需完整源资格。

原接触和力学 qualification、BC qualification 及原 371D／24D 合同均沿用现有状态。该选项尚未进行新 Isaac 原生验证。可选的私有 transition journal 记录的原 371D／24D 与当前字段 mask 不受本次扩展影响，晚期异常时当前版本的内部源快照可能未落盘。

## 验证

Ubuntu CPU 回归 56 项通过，0 失败，包含原生产 `TensorTask.observe` 的 371D CPU fixture、原 `SurfaceFeedback.reset`、同钟原生归档 join 逻辑，以及错源、错误 reset 事件、残留累加器和字段数值不匹配用例。Fixture 不提供原生物理证据。新模式在原 T5 启动文件和 Stage7 输入上 dry-run，实际核对 10 个 SHA，预检 JSON SHA256 为 `9b79c9b5043774e2f282ab475f84b5f8bec90a7d42df21d956ed21760e7a855c`，输出和 journal 文件均未创建。

```bash
export PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
python=/mnt/mydisk/xk/envs/dh116/bin/python
"$python" -m pytest -q -p no:cacheprovider tests/test_mpc_internal_source_capture.py \
  tests/test_mpc_capture_journal.py tests/test_mpc_privileged_capture.py \
  tests/test_dh116_privileged_action_audit.py tests/test_surface_feedback_wiring.py
```

下次独立原生实验可在已经固定预算的 T5 命令上附加 `--capture-internal-source`，先用现有 dry-run 核对输入。取得真实输出后检查两种 mask 的逐列有效样本，报告初态及当前接触条件下实际通过的数值。本包完成时，当前 T5 使用原冻结源码继续执行。
