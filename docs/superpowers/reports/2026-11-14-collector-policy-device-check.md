# Collector 权重摘要前的 CPU device 复查

日期：2026-11-14。父提交：33859bdcff815fcb42579550499db349bbd19e01。
分支：codex/teacher-collector-device-20261114；仍使用独立 teacher-collector-owners-20261114 工作树。

此前构造器只检查 parameters，_policy_check 在 _model_weights/_digest 内可能先调用 .cpu()。
本后继在摘要前检查 parameters 与 buffers 的 device.type；构造器也覆盖 buffers。
只读取 device 元数据，不分配/初始化 CUDA、不做 GPU→CPU 拷贝。
owner 协议仅同步 collector 源 SHA，没有扩展 owner graph 或原生入口。

新增两个参数化负例使用 CPU 上的 SimpleNamespace(device=torch.device("cuda:0"))，
替换 parameter/buffer iterator；把 _model_weights 设置为一旦调用立即失败。
先红 2 failed，修复后两项均证明摘要未调用、SDK/rod 未推进、PPO pending/rollout 未创建。

focused 回归：**46 passed in 32.36s**（新增 2 + 现有 collector/owner 44）。
本后继没有重跑所有旧物理/PPO测试；父提交完整回归为历史 137 passed。
原生、GPU allocation/copy、正式训练、synthetic optimizer 更新均为 0。

工作目录：
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-collector-owners-20261114

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q \
 tests/test_dh116_no_camera_policy_device.py \
 tests/test_dh116_no_camera_fixture_owners.py \
 tests/test_dh116_no_camera_collector.py
~~~

证据根：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collector-device-20261114

- red.log SHA256：b92a8687a784bceba93fc04bb116a2e86224e6dfb10ba4787b0b5d4f94f0f9f1
- regression.log SHA256：d2582b9e33d12c6e1c792cc6f95de2cbbe2cd8eca7723233370ec093cb62c0dc

真实 collector factory 的 mesh Reset / 正式 R0 缺源拒绝保持不变。没有 push。
