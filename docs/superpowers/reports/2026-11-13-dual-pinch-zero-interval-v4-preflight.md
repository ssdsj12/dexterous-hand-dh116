# 双手 Pinch 原台面碰撞节点与原生复验

v3 已执行两个初始化 SDK 步，测量 0 步。左右手实际挂载和自由杆 Reset 完成，场景配置交接通过。台面接触读取器在旧的根节点检查处终止。app.close 返回，外层和 GDB inferior 均自然退出 1，无信号，69.440 秒。原始物理账本 SHA256 `36cff198922a48525cedfcbf5bd68c3fff7d2cdbbe44eec2ca1ba9c948bb0168`，回执 `b70888833bab2e1a1cbddff2fe9595b8953d31565d2ff7dc5c1c1a49da71c64c`。

安装版 Isaac Lab `spawn_cuboid` 创建 Table Xform，并在 `Table/geometry/mesh` 挂载 CollisionAPI。CPU 使用原 spawner 方法的源码及实际 USD 骨架复现旧根节点失败。独立读取器在现有 Table 子树识别启用的 collider，核对刚体祖先及左右实际 17DOF、23 body、同一 SDK 和杆时钟。原 USD、物理、Reset、动作和安全路径沿用。静态手台面接触继续记 NA。旧 GPU 警告使用了 Table 根节点路径，实际 mesh filter 能否返回有效的逐接触数据仍未知。

独立台面包新增 18 项，含原行为在内 59 项 CPU 通过。继任 worker、台面模块和原后置审计整合 38 项通过。测试使用安装版源码的 USD 方法、CPU SDK doubles 及真实上一轮账本，不启动 Isaac 或 PPO。

- 配方 SHA256 `f1419d81ffacec115dfa086e1421f5b77d583a1c6cdc81ae6b704835bb095135`
- 调度器代码配置 SHA256 `9e95d33e8b0565133c05d8fae5eb5c5eb68de269806ae561e4c5bd022bd9e202`
- worker 前检 SHA256 `51470123f470624664c86b0b5845b7c416af395630e7516f28d3bce0ce8203e2`

GPU 3、单实例、1 环境、2 初始化加 8 测量步、180 秒、0 PPO 更新、一次尝试。双手物理执行、同杆接触、抓取和 Teacher 准入分开验收。
