# Pinch v8 调度启动预检

新配方使用现有 `Repository.start_native` 构造解释器与调度器进程参数。禁用字节码写入通过环境变量 `PYTHONDONTWRITEBYTECODE=1` 完成，进程参数符合冻结校验。真实 CPU 子进程直接与 GDB 握手回归 4 项通过。

依据 v7 已封存的 Reset 与训练合同生成 v8 的独立输出路径。训练配置的字节 SHA 仍为 `763a46f39355a8870e4d336812e9a4be588b11f220239453d89e4385175b566d`。运行原 `prepare_dh116_pinch_ppo_draft` 的 `_cpu_reference` 完成状态、后端、动作及训练合同预检。1 环境、32 控制步、最多 258 SDK 步和 3072 求解器子步、1 次 PPO 更新，内外 180/240 秒。单 Isaac 和现有锁适用。

新工作包 `teacher.pinch-ppo-fresh-smoke32-v8` 状态 pending，冻结指纹 `e4f597aecbf764ebf97fd149b7fcc175f832aca79973fec4b1ab86e24219f13a`。v7 失败记录保留。v8 为待独立授权的原生诊断，已执行的 SDK、求解器和 PPO 步数均为 0。
