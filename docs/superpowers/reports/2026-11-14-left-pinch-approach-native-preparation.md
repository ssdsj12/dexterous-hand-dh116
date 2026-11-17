# 左 Pinch 高位 Approach 原生执行准备

## 已完成范围

在原右手tracking-aware路径基础上，仅接入已通过完整CPU路径审计的独立左手Reset/Approach。末端及路径证据SHA a5ca10a1f612a20e216b545ccbd71a7aeeceb56117ddddb1101f4e7dfa1b4d74。完整原配置、五指SDF、原几何/关节保护、真实自由杆/桌面、材料和右手动作不变。左手起点提高1.5mm，160个物理步下降至原候选末端；手指闭合增量不变。

新增plan将左手static_reset_only显式设为false，保存完整Reset seed SHA。29个60Hz动作覆盖232个480Hz测量子步，前48静置、160闭合、24短保持；加2个初始化步，共234SDK步。左/右主动关节各6D、掌位动作各3D、旋转各3D仍遵循原24D合同。

## 实际代码与保护

- prepare_dh116_left_pinch_approach.py：从独立SHA证明生成冻结配置。新左手Reset不覆盖右手/杆数据。
- tabletop_left_pinch_approach_geometry.py：替换旧“左offset必须0”的实验假设，逐边界明确核对左右request/projected/SDK root targets、全部24D动作、policy/substep身份和原安全投影来源。裁剪或异常原因均拒绝。复用所有原source/cooked、非目标接触、SDF和关节保护，没有删除碰撞检查。
- 独立prepare/verify/work-package入口复用已验证自然退出资源owner、同钟账本、完整post-exit审计、GDB栈/映射和单Isaac锁。作用域在正常/异常路径都恢复。
- 独立execution JSON冻结158项源码和原输入SHA；没有读取临时假回执作为依赖。

新入口preflight SHA 76846b82c0a36217e877acd4d8b1214c418aa5a2d7b4e672ff18b260cc29ece1。
配方SHA 7af604a3a6caa566c8736b566abe55ca94a8343022a7308ed1deff6d0d68af18。
调度器自身code/config指纹在发布后按相同源码和配方生成并绑定独立授权批次；不复用旧原生授权文件。

## CPU结果

5个新测试文件共43项通过（plan+geometry 17，native+worker+owner 26）；0失败/跳过。后一组142.80秒，包含真实CPU子进程重入和显式合成SDK buffer owner测试。**没有将合成buffer算作原生验证**。

覆盖全部232边界双手q/掌位目标重建；原控制器回放无projection clipping，左右q误差在1e-7rad内、root误差在1e-8m内。对左/右offset、root、rotation、safe-z、reason、24D动作和时钟损坏均拒绝。完整owner以及物理源故障/terminal路径保留最终样本，原关闭顺序正常调用，不用强制退出。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_left_pinch_approach_plan.py tests/test_left_pinch_approach_geometry.py tests/test_left_pinch_approach_native.py tests/test_left_pinch_approach_worker.py tests/test_left_pinch_approach_owner.py tests/test_tracking_pinch_plan.py
```

私有失败/通过日志保存在 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/left-pinch-tracking-20261114。先保存red再实现。发布集成阶段另运行上述新旧回归，保护528文件及两份历史Checkpoint。

## 冻结原生范围与未决问题

按照用户“批准，后续自动批准”的持续授权，发布并校验后可绑定本独立配方：GPU3、单Isaac、1环境、0PPO、最多一次234总SDK步、180秒。原接近动作/速度边界不变；新左手腕路径是显式独立配置。失败保留完整账本/GDB/外层退出证据，不自动重跑同版本，不扩大预算。

本准备包新增物理步0、PPO0、WireSeg训练0。下一步才是新原生执行。左手动态跟踪/接触、双侧有效承托、固定材料锚Slip、完整R0和Teacher资格仍未取得；短保持不能替代Lift30mm/Hold5s/Slip0.5mm。PR#2继续Draft。
