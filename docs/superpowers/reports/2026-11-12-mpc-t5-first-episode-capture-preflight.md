# T5 原 MPC 完整首回合采集预检

原始 T5 启动记录 SHA256 为 `d232f5be63367aac58daa805f9af32ad025e9cda3a2d575faf9d3c9a0eed41c5`。历史记录为 8 个完整首回合，二指 4/4、三指 0/4，共执行 1800 个 60 Hz 策略步。历史结果见 [T5 原生验证](2026-11-08-stage5-native-validation.md)。

新入口 `--t5-first-episodes` 仅接受固定 SHA 的原启动记录，以 Stage7 原始计划的六个输入和四个 USD SHA 校验路径及内容。保留原 `pressure_geometry_constrained`、seed 42、8 环境、相机、物理设置、控制合同及 600 步 hold tail。历史命令的硬上限为 1900，实际轨迹运行 1800 步，新入口以 1800 为显式上限。输出目录必须新建。原启动命令中的旧 Teacher 物理审计观察器替换为现有 sequence context 和 MPC state 两个采集观察器，这两种观察器在当前原生运行验证中不可同时启用。无新动作变换。原 2×32 诊断入口继续使用原预算。

## 归档合同

完整 371D 观测与 24D 实际发出动作按 env、episode、tick、相机帧、SDK／World 时钟和每个有效 480 Hz 物理子步对齐。成功、失败、未完成三个归档单独写入，每一组保留原行索引。成功归档要求原生完整首回合、原任务成功终止、failure=0、无恢复、原执行区间完整和对应抓法的原生接触样式审计。最终行与带 SHA 的原生 terminal snapshot 动作、时钟和归属一致。其余已终止首回合归入失败，达到运行预算仍无终止归入未完成。实际 PPO 更新数为 0。

当前可交叉检查 24 个关节位置和速度字段。其余 347 个原值保留，逐字段有效性仍未通过独立核验。即使采集到原生成功首回合，`qualified_for_bc` 仍为 false，待完整 371D 与现有力学准入证据。沿用原任务 30 mm、5 s、0.5 mm 成功条件。

## 原生资源预算

原 T5 8 环境 1800 步采集计时 2510.846 s，输出 3.026 GB。其中 RGB-D 2.172 GB。最近 2 环境 32 步同钟诊断计时 44.162 s，输出 30.424 MB，其中双份 MPC 状态及原始传感器日志 9.588 MB。按环境物理样本比 225 倍外推，整套短诊断格式约 6.85 GB，单独 MPC 物理记录约 2.16 GB。预留 12 GB 私有实验空间，运行时间先按 1–3 小时安排。实际耗时由渲染、序列化和首回合结束时刻决定。运行盘预检可用 3.814 TB。新增观测器尚无 8 环境原生显存实测。

## Ubuntu 验证

当前源码 `9a74290b5acc4660d103f6028b360b1046228739`。用实际原始记录完成 CPU 预检，十个输入 SHA 均通过。预检 JSON SHA256 为 `15434c7b275cb8d36db10c242ea4ea92407cce82ec3ce2addcd7db1d115e847d`。本包 CPU 回归 37 项通过，0 失败。新增用例覆盖实际模式预算、旧模式不扩容、输入篡改、成功与失败归档、缺失 terminal、未审计成功、策略缺步及 terminal 时间错误。复核了 528 个保护文件和 2 个历史 Checkpoint。实际原生执行次数为 0。

```bash
export PYTHONDONTWRITEBYTECODE=1 DH116_NUM_THREADS=2
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES=3
bash scripts/python.sh scripts/collect_dh116_mpc_privileged.py \
  --baseline-plan "$stage7_input_plan" \
  --baseline-plan-sha256 0561d7fa70845e7ea85b386d7befa35f35febb16ab44f384d0c85ac65adf1c31 \
  --t5-first-episodes "$t5_reference_launch" --output "$new_native_output"
```

预检执行不创建原生输出目录。正式运行时从冻结源码核对资产路径与 SHA、当前单实例资源、原生预算与独立输出，再使用同一命令追加 `--execute-native`。核验实际原生结束状态、归档和每种抓法的终止结果后更新 readiness。
