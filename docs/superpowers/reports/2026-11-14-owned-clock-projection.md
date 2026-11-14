# 原验证器时钟投影：只读完整账本 A/B

基线 `98e995af8541300f52f6309f92e2da7005ee09c5`；开发 Ubuntu 独立 Worktree。旧 SDK、物理、动作、原始数据和时钟校验器均未改。新模块仅 opt-in，不进入默认训练。

## 改动及验证

- `tabletop_owned_clock_projection.project_clock` 只读取一次真实 clock reader。普通字典的十个内置标量时钟字段先选取，再调用原始 `tabletop_bimanual_table_contacts._clock` 校验。自定义字典/字段保留原 deepcopy 路径。没有缓存、填零、减少采样或跳过校验。
- `clock_projection_scope` 仅替换明确的已导入消费者及原 source；不主动导入 SDK。异常也还原，拒绝嵌套或未知覆盖；作用域内新导入的同源别名也还原。
- `scripts/replay_dh116_owned_clock_projection.py` 对同一完整账本执行原始/投影/投影/原始顺序；报告精确结果 SHA 与耗时，拒绝输入变动。
- 红测试：19 项因为模块不存在失败；后续补充缓存、异常、嵌套、延迟导入、自定义容器回归。运行命令：`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_owned_clock_projection.py tests/test_dh116_bimanual_table_contacts.py tests/test_dh116_bimanual_table_contacts_v2.py tests/test_dh116_material_director_readback.py tests/test_dh116_native_parameter_endpoints.py tests/test_flexible_parameter_readback.py tests/test_dh116_native_parameter_task.py -q --tb=short`。

## 实际数据

来源为已耗、超180秒失败的 `attempt-_vy7ya1j`，不能因离线分析重新晋级。
完整 physical SHA：`1a8acccbd58536848d046475da39293267da575732efb57cfc5b8e8144a3b0c5`，232 个测量物理样本、29 个控制窗口。

| 顺序 | 模式 | CPU 秒 | 完整审计结果 SHA |
|---|---|---:|---|
| 1 | 原始 | 32.230929 | d9549869eadbd0f9c8c172cab86dd1e5265f35383e14df371a96e412ab0d5571 |
| 2 | 投影 | 27.804224 | 同上 |
| 3 | 投影 | 28.508115 | 同上 |
| 4 | 原始 | 32.002046 | 同上 |

输出逐项完全一致，输入 SHA 未变。均值减少约3.96秒（12.3%）；仅 CPU 只读审计，不是新的原生运行或 GPU 性能认证。结果 JSON SHA：`0564b85d847e6c8a8cc5ce70610610a2ec964b2ea9bc62708771eb37daa94a53`。证据位于服务器私有 `evidence/clock-projection-20261114/`，未上传原始数据。

第一轮在独立 Worktree 回放被既有 `parameter identity/source differs` 拒绝：历史参数绑定原采集工程绝对路径。失败日志保留；之后从原 `integration` 目录只读加载原审计器，以文件加载新投影和回放脚本，未修改来源字段或放宽源 SHA。四次对照均使用同一实际来源。

## 限制与后续

本包 0 新物理步、0 PPO；不授予 Teacher、抓取或新原生资格。四秒改善不足以证明194.48秒失败原生能进入180秒预算。后续建立独立“两阶段审计”：GPU owner 只做逐步原始记录和封存，正常退出后原验证器完整重审；封存必须明确 pending，任何缺失/篡改/审计失败阻断，不能用占位通过值。原生配方必须新冻结，旧耗尽配方不重跑。

最终 CPU 回归：154 passed，165.89s，0 failed。原生执行 0；Teacher 资格仍未通过。
