# DH116 右拇指生命周期 v2 调度预检

基线为 `eb31c793d60cb50eb8d6dd2e06bae2f15c5bd8d5`。本包加入右拇指调度配置。左、右配置使用同一 GDB 观察方式和原 v2 物理及释放顺序。预检期间 Isaac 实例数和原生物理步均为 0。

| 锁定证据 | 右侧 |
| --- | --- |
| 原始 USD | `3046ad2ed855_DH116_right.usd`，SHA256 `37ff2a3a724778696f96732f713c0b55c01cc96de0f1c35d4c7e8a7fd7532f71` |
| 独立 GPU cooked readback | `right_finger14_link.json`，SHA256 `62615e9350789c8f4ccd319c2c78cbd584830a56e0a2b9ed39cc61d57b8c9c29` |
| 独立 SDF | `right.finger14_link.npz`，SHA256 `a001ba259df9b9884f065d3f631b816ab6be3e35475d002286513876762807c0` |
| 场景、SDK 和 solver | `/World/DeformXThumb/rightHand/finger14_link`、DeformX slot 0、单杆 `rod0` |
| 24D 动作 | 仅 canonical 索引 7，原始右 `finger12` 关节轴；命令窗口外 24D 均为零 |
| 预算 | 1 env，物理 GPU 3，240 行 480 Hz 物理账本，0 PPO updates |
| 调度前置 | `flex.f4-left-contact-native` 的合格调度回执及其 SHA，使用独立核对的自然外层 exit 0 |
| 结果验证 | 240 行、原始身体路径/右 SDF/右 hull/右 USD/slot/动作索引，前 48 步无接触、后续至少 3 步接触，力残差不超过 0.0003 N，关节和 SDK 目标检查无失败 |

使用实际右 USD 的只读 OpenUSD composition，得到 authored collision surface SHA256 `6e8d14f405a375eb229e0caa2a41bd3de5b1ac365ea18fea1bdf3a51674540d9`，与历史 GPU cooked 记录绑定的源 SHA 一致。原始网格只用于身份核对，solver 输入为锁定 SDF。检查 13 个外部输入、528 个受保护文件和 2 个历史 Checkpoint。用独立 CPU fixture 验证调度结果解析拒绝右记录中的左动作索引；用历史失败左账本的 240 行回归新身份检查，账本仍保留其原始退出失败结论。上述 fixture 未产生真实右侧接触资格。

运行命令以 Isaac 安装中的 OpenUSD 作为 CPU 库。OpenUSD 0.22.11 加载成功。

```bash
usd=$(find /mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache -maxdepth 1 -type d -name 'omni.usd.libs-*' | head -1)
PYTHONPATH="src:.:$usd" LD_LIBRARY_PATH="$usd/bin" /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_deformx_bridge_lifecycle.py tests/test_dh116_deformx_pad_sdf.py tests/test_dh116_cooked_convex_hulls.py tests/test_dh116_cooked_hull_device_parity.py
```

18 项通过。新增两项测试先分别因缺少右配置和账本验证入口失败，再经实现通过。右配置 SHA256 `086d7570166384ec9e42a5541f14b7fa87f1529ef96c1b0b5dffb61ba5ae3125`。外部预检 `flex-a-right-preflight-5GUUa0/preflight.json` SHA256 `22f45e5df98950ebe8c0882499c0fa8b2d2bd66841d1ac49c1981f36fbdf43ee`。

## 同一次运行的 C++ 崩溃证据

Ubuntu 的 core soft limit 为 0，hard limit 为 unlimited，`core_pattern=core`，`ptrace_scope=1`。GDB 12.1 可以作为目标进程父级追踪。CPU-only SIGSEGV fixture 验证同次捕获 C++ stack、共享库和进程映射，并继续传递原信号。GDB 本身状态为 255，inferior 明确由 SIGSEGV 终止，调度器须分别记录二者并拒绝资格。Fixture 日志 SHA256 `d46597b847c02fc1859815b3bf732dfd6b09f103921a8158d5524672f0e7b51c`。使用配套调度器的 `gdb_on_segv` 与动态左回执支持后再冻结指纹。只启动一次左，左资格完整通过后才启动一次右。

本试验是 21 节点、双端 pinned、零重力的单拇指接触标定。待独立任务配置冻结实际重力、自由端点、外部支撑和计量来源，再进入双手抓取及 Teacher。
