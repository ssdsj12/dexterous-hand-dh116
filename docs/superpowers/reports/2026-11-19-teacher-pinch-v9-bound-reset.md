# Pinch 新右手 Reset 原生读回与 v9 配方

CPU 候选只更新右手 Reset。第一次准备发现我在发布之前冻结了包含 README 关联发布合同的源清单。随后更新 Registry、发布清单与 readiness，原冻结指纹在 CPU 预检被拒，Isaac 子进程没有启动。发布完成后从 `3e6334d` 重新冻结并在新授权指纹下运行一次。

| 原生验证 | 结果 |
| --- | --- |
| 配方 | GPU 2，1 环境，2 个 SDK 初始化步，1 次 Reset，零控制、杆求解器和 PPO |
| 左右手原始几何 | 两侧 source 与 locked 指腹初始读回均通过 |
| 关闭 | 43.962 s，外层自然退出 0，正常关闭，owned lifetime release 通过 |
| 独立验收 | 核对八项读回文件及外层回执，封存 Reset manifest |
| 原生 Reset 指纹 | `45c9109e6216432616341205d1a3b25aed6b18847e66856dd02491f338a48225` |
| 外层回执 SHA | `50c2f4ac6abc11921db5092b9470368e50e3a430bee70ef69ab07e6c722f4efe` |
| 新 manifest SHA | `488c8220bf7c66ac60aafd4ec3c779c1b0af0334c073420b2162877523a5c3f0` |

v9 引用此次封存 Reset 而生成，保持原相机关闭、1 环境、32 控制步、至多 258 SDK 步和 3072 求解器子步、1 次 PPO 更新、180/240 秒。冻结指纹 `1484e90ac79677f02d965750b1cf2b1cc40886a6155ceee3bad2365d9322c7a2`。使用 `Repository.start_native` 的原有子进程调用方式。新配方具有独立证据和输出路径。

17 项目标 CPU 回归通过，33 项未选中。Reset 资格已封存，PPO、接触力、Lift/Hold 与教师资格仍需各自验收。历史 v8 签名与物理终止记录保留。
