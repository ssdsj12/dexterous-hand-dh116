# Pinch v11 原生 Reset 与固定种子 PPO 预检

右手配方完成单次 GPU 2 原生验证，1 环境、2 个 SDK 初始化物理步、1 次 Reset、0 控制步、0 杆求解器步、0 PPO 更新。左右手实际 source collider 与 locked pads 均通过原来源和净空校验。清理报告记录 `lifetime_released_before_close=true`、`close_mode=normal`。外层自然退出码 0，耗时 47.343285 s。Reset 轮次未验证持续控制、对向载荷或抬升。

- 候选配对 SHA `48d007fcdffa16cdc0fd3a1c711a708067de0d5cd584c8e670afe83d04927cfb`
- Reset 冻结指纹 `c3f75ffafb9d9870339356bab80c54d920ee2aa1bad940c0a7d566268490c4e0`
- 独立外层回执 SHA `3ba8b7d237be8cbe3d7121e449ec02b51fcdbe1af6dee17e5ef8a0f943da7eb8`
- 封存 Reset manifest SHA `6f68015fe317ecdc28468f9483cd24361c1d894ae316bd02c3f6117ccaf73eef`

v11 实际 Worker 配方绑定该 manifest、CPU 初始化及探索 seed 和现有原始动作保护。本次仍为 GPU 3、单实例、1 环境、32 个控制步、最多 258 个 SDK 步和 3072 个求解器子步、最多 1 次 PPO 更新，内层 180 s、外层 240 s。独立指纹 `f62e1849dc0dbcbfb5a3309eb521c9f683bbeecc3a62f5491c2f5938a525edd5`。最终发布后再核对启动指纹与资源。v11 的 CPU reference 和 124 项 CPU 回归通过。尚未执行 v11 PPO，无抓取或 Teacher 准入结果。
