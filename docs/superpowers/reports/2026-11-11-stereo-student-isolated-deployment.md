# WP6 Student 双目 RGB 独立部署验证

基线为 `3c80166da75fb5496701c9f3ff673296c5db5615`。验证范围是 `scripts/infer_dh116_stereo_rgb.py` 的独立 CPU 推理。部署输入包含同步左右 RGB、测量本体状态和相机时钟。输出是 Student 的 canonical 24D 动作，脚本不向设备发命令。

## 读取与身份修复

原推理入口在校验包结构前使用通用反序列化，并多次打开 Actor 包和相机文件。对 Actor 与诊断分割权重使用 `weights_only=True` 的单次字节快照。Actor 文件可通过 `--actor-package-sha256` 绑定预期 SHA，分割继续绑定配置声明的 SHA。两者都先核对声明的文件身份，再从同一份字节构造模型。实际相机内外参、预处理合同、视差配置和报告 SHA 均来自同一份相机 JSON 字节。收到不一致的 SHA 会在加载前退出。

新路径复用旧 v1 的 Actor 状态形状、合同和张量 SHA 校验，拒绝混入 Critic 或其他额外顶层字段。输出新增 `action_owner=student` 和 `mpc_executed=false`。图像新鲜度上限要求有限正数，避免无效浮点值绕过旧有时钟比较。旧 Actor 格式、预处理身份绑定、Teacher、PPO、模型结构和原始权重文件均沿用现有定义。训练与原生运行继续走各自入口，本轮验证限定为独立 RGB 推理。

## Ubuntu 验证

Ubuntu 22.04.5，Python 3.10.12，PyTorch 2.7.0+cu128，使用 CPU。运行：

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_stereo_inference.py tests/test_dh116_stereo_runtime_actor_modes.py \
  tests/test_dh116_stereo_runtime_integration.py tests/test_dh116_stereo_preprocessing.py \
  tests/test_dh116_stereo_student_ppo.py tests/test_dh116_stereo_training_lineage.py
```

**68 项通过，0 失败，58.63 秒。** 在 Ubuntu 用基线文件重现 6 个预期失败，覆盖包读取模式、文件身份竞态、Actor SHA 绑定和无效新鲜度限制。初次联合测试有两项跨进程浮点差异，最大绝对差为 `1.863e-9`。最终同进程使用一致的无梯度模式验证逐位一致，跨进程采用 `atol=1e-7, rtol=0`，最终全部通过。

三种模式 `latent`、`estimated_task_state`、`hybrid` 均实际写出 Actor-only 测试包、在全新子进程中通过真实 RGB CLI 计算视差并推理，再与同一输入和权重的 Actor 直接推理比较。子进程主动禁止导入 Teacher、MPC、特权 PPO、Isaac 和原生桥接模块。另有回归覆盖已有 PPO Checkpoint 的 Actor-only 导出和重载。合成图像和初始化权重仅用于验证软件合同。

| 证据 | SHA256 |
|---|---|
| 最终 CPU 日志 | `06025d8b8a148ef118ed03e24790007b5430c94d705e4fad55310228e9d81dc9` |
| Ubuntu 基线失败日志 | `9cca2d3ad7ecd235a416986bda4149ee2a3a5fde46b9a1154bf6b85407a57ad7` |
| `tabletop_stereo_inference.py` | `57fe1f6a0af95561fa32088bf788039f2aa8ab3217f14d97738de5db9c5599a4` |
| `test_dh116_stereo_inference.py` | `559a173ad47c56c7412157ef427653063de9cfd6257be49cb97ef95a2c1667ea` |

完整日志保存在服务器独立证据目录。Isaac、实机、BC、DAgger、Student PPO 和正式独立 Lift/Hold 验收本轮均未运行。下一步在合格 Teacher 与双目数据具备后，用经验证的 Actor 包开展分别批准的原生 Student-only 评估，继续使用每种抓法 100 个独立初态和现有物理成功标准。
