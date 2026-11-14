# A 左拇指同杆接触诊断及外层退出失败

按授权在 GPU3 运行一次原始左 DH116 articulation、一个核验后的 `finger14_link` SDF、同一根 21 节点柔性杆的无相机标定。外层使用现有 `native_instance_lock` 与独立退出码，早于新调度器集成。原手 USD、11 个原关节、5 个 mimic、固定根、原控制动作合同和原训练文件均未修改。原生 collision owner 的源几何 SHA 与活动 GPU cooked hull 一致，实际 thumb body 路径为 `/World/DeformXThumb/leftHand/finger14_link`。场景关闭自碰撞，其他 PhysX 碰撞体逐个登记，DeformX 只注册 slot 0。杆两端节点 0、20 是标定约束，这次不会形成任何抓取成功样本。

**左拇指原生接触样本通过，进程退出资格失败。** 记录了 240 行连续 480 Hz 物理数据，SDK step 3–242。首 48 行无接触，之后第 173–240 行连续 68 个交换步记录到材料节点 10 与左拇指接触。原生调用返回并写出 SDK manager view 释放及 App close 标记，外层实际退出 **139**。右拇指未启动。

| 量 | 实测 |
|---|---:|
| 初始最近线缆节点 SDF | 9.330 mm，半径 5 mm |
| 计划 `finger12` 目标 | 0.35 → 0.26 rad，160 步，−0.27 rad/s |
| SDK 实际 `finger12` | 0.350000 → 0.260332 rad |
| 实际 `finger12` 与全部关节绝对速度峰值 | 0.273341 rad/s |
| 24D 命令 | 仅左 `finger12` 索引 1 有 160 个非零值 |
| 最小记录节点到 SDF 表面距离 | 4.999761 mm |
| 杆对手与手对杆峰值力 | 0.047305321 N / 0.047305321 N |
| 逐交换步手／杆力和最大残差 | 0 N |
| CoM 反作用力矩峰值 | 0.000270324 N·m |
| 下一交换步实际输入峰值力 | 0.047305338 N |
| 反作用力与下一步 composer 实测分量最大差 | 1.50e-8 |
| 安全 guard 或 SDK 目标偏差 | 0 次 |

账本保留原 canonical 24D、实际 SDK joint position targets、前后全部 q/qd、SDK link/CoM 位姿与速度、实际杆节点、节点力、wrench 作用侧与 slot。后端手侧关于 link 原点的 wrench 换算到 CoM 后，仅在下一交换步由本 probe 独占 composer 发送一次。杆侧节点合力是本次实际独立读回，关于 link 原点的杆侧力矩使用同交换步末节点位置估算，不能当作求解器积分力矩。手指关节约束和根部反力缺少来源，标为 NA，不套用自由刚体冲量验收。

## 退出证据与后续隔离

`shutdown_complete.json` 表示 `app.close()` 已返回，`process_exit_verified=false`。其后的外层退出 139 表明失败发生在 close 返回后的阶段。未获得此次运行的 native backtrace，也没有对应 core 文件。历史 F3 的 GDB 结果不能替代本次的栈。

本次 `ownership.jsonl` 在 close 返回后仍观察到 `Articulation`、view、composer、`RopeBatch` 的 Python weakref 存活。已验收的 F3 刚体探针在 native loop 内显式删除桥接与杆引用，随后在 STOP 后、App close 前用 GC 和 checkpoint 确认 body、view、composer、杆均不再存活。此差异是接下来可测的生命周期假设，尚未证明是故障原因。接下来应单独改变显式拥有引用的释放顺序并提高退出观测，保留当前接触输入和证据。此次原配置不会自动重跑，右侧继续依赖左侧最终外层退出 0。

## 复现与证据

```bash
private=/home/xk/.local/share/dh116-autonomous-3c80166/evidence
f1="$private/flexible-f1-20261112-883RS4"
export CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH="$f1/sdk/lib64"
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
bash scripts/python.sh scripts/verify_dh116_deformx_thumb_contact.py --headless --device cuda:0 \
  --side left --steps 240 --ropesim-python "$f1/build-attempt3/python" \
  --sdf-dir "$private/flex-f4-sdf-register-rK7gw0/bake" \
  --capture-dir "$private/flex-f4-parity-prep-Vu9H0f" \
  --output "$new_output" --execute-native
```

CPU 名义 FK 预检记录在 `flex-a-thumb-preflight-sMaTlr`。本次原生目录为 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-a-thumb-left-QvvmpW`。工作树脚本与本次外层记录的执行 SHA 完全一致。

| 原始证据 | SHA256 |
|---|---|
| physics.jsonl | `e3ff29b703f29474222ecfd033eeba97a6ac2ad7e36be6713fd5e6653d6faf7b` |
| result.json | `7e235bf55576c37cc372635888349dccc383a72452c31a59e698548b01dac24d` |
| ownership.jsonl | `0fbf29ba868ee322b01491d5faf96c5d14fd766f6e6c18f6c9a4bfb869249fb3` |
| 独立审计 | `72601a242b72646e0bf0e9bbb2b96308fd227fd0d0d2d526bcb368d50e564ef3` |
| OpenUSD CPU 回归，8 通过 | `5ebea2affe88ad358e97d5c734756ce5ba2afb0e744383e096b507f614e0e59a` |

528 个保护文件和 2 个历史 Checkpoint SHA 通过核验。无正式 Teacher、抓取或训练资格。
