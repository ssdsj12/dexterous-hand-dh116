# Pinch v12 实测 Reset 和 PPO 计时配方

GPU 2 单 Isaac、1 环境、2 个 SDK 初始化物理步、1 次 Reset。左右手的实际 source geometry 和 locked pad 读回均通过。记录零控制步和零杆求解器步，关闭前资源释放验证为真，`close_mode=normal`。外层进程自然退出码 0，耗时 48.599 秒。没有 Teacher 查询或 PPO 更新。

- 原生 Reset 冻结指纹 `dafcd8c1324780bfe5bd6d4502c92ac3b2c8b0f368fcbc26cabbfd63c379169c`
- 独立外层回执 SHA `d35689abb67d793bce464cb92013203fc93bc931209ca75e098691caf89d528a`
- 已封存 Reset manifest SHA `a860e7b3e7799fc9328e84d7cc7ad9e2fde8d9d476ced93cfc65de34cf3eb35b`

新的计时版 PPO 指纹 `ad9a5f4c74b2e06fbea6c8ce0dae27129f1d483646be9698421731bbfed7a6f2` 绑定该真实 Reset、原 R0 smoke 配置、已验证的 CPU 随机序列和原物理保护。GPU 3 单实例、1 环境、32×8 个测量 SDK 步、2 个初始化 SDK 步、最多 3072 求解器子步和 1 次 PPO 更新，180/240 秒内外墙钟。每个控制窗记录分段耗时和真实终止。还用独立 CPU 参考模型和 Adam 核验了两个配方的完整训练语义。旧 matched-Reset 草案的来源清单更新后留下了旧语义 SHA，已重新推导，未启动旧草案。

83 项 CPU 回归通过。当前实证覆盖新初态和自然关闭，抓持稳定性、离地抬升和 PPO 更新仍待原生控制验证。
