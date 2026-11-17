# DH116 柔性预算 v2 计入 SDK 初始化

本次从 `d4dad93fb2612cbeb98c56781ec5a761cbb59d76` 增加柔性任务 CPU 预算 v2。已有 v1 配置保留。v1 的 `physics_steps_per_env`、`total_physics_rows` 和 `solver_substeps_per_env` 继续表示测量阶段，返回 `phase_only_setup_unaccounted`，初始化和实际总步数为 `null`。

左右拇指历史运行各有 240 个接触账本对象，SDK tick 为 3–242。SDK reset 路径在账本开始前实际运行 2 个物理步/侧。历史两侧合计 484 个物理步，超出批准的 480 步。本工作包只修订独立预算计划接口，未重跑历史实验。

| v2 草案例 | 环境数 | 控制步/环境 | 测量物理步/环境 | SDK 初始化步/环境 | 总物理步/环境 | 测量账本行总数 | 墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 单拇指接触标定 | 1 | 0 | 238 | 2 | 240 | 238 | 180 s |
| Teacher 单次有界诊断 | 2 | 32 | 256 | 未验证 | 未知 | 512 | 300 s 示例 |
| 冻结评估示例 | 1 | 360 | 2880 | 未验证 | 未知 | 2880 | 900 s 示例 |

`plan_flexible_experiment()` 在 v2 强制显式提供 `setup_physics_steps_per_env`。可验证的初始化步数以整数输入，尚无证据时写 `null`。只有显式设置初始化步数后才计算 `total_physics_steps_per_env` 和按环境累计的 `total_physics_steps_across_envs`。后者为工作量合计，不代表同一并行场景的 SDK 时间线。测量账本行独立计数，不把初始化步加入接触记录。时钟保持 480 Hz 物理、60 Hz 控制，冻结评估保留 2400 个连续 Hold 物理步，另计接近和抬升。拇指标定 v2 预算固定 240 总步，初始化 2 步，测量 238 步。

冻结评估示例的 360 个控制步由 60 步接近与抬升预算、300 步 Hold 预算组成，仅用于说明 60 Hz 与 480 Hz 的换算。自由杆达到 30 mm 的可达性和实际抬升速度尚未验证，无法确认 6 秒窗口足以完成接近、抬升和保持。正式冻结评估预算须依据真实物理接口和轨迹验证结果确定。

v1/v2 所有计划均返回 `requires_separate_native_approval=true` 和 `native_qualification=false`。任何版本的冻结评估计划都会在初始化预算未知时被范围检查拒绝，即使已有非零来源 SHA 和已配置 solver 子步预算。v1 配置和旧字段继续用于读取测量阶段预算。计划接口只校验配置字段。真实物理步数、参数来源和 30 mm / 5 s / 0.5 mm 资格须用独立原生证据复核。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_deformable_task_budget.py tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_deformx_bridge_lifecycle.py
```

新增五项回归先观察到 v1 标定错误地无需独立批准、v2 不被识别。另用 v1 冻结计划复现初始化未知时误通过范围检查，修复后该计划被拒绝。共 24 项 CPU 测试通过，并核验 528 个受保护文件和两个历史 Checkpoint。本包未改原生调度器、冻结拇指脚本、配方、训练入口或物理参数。
