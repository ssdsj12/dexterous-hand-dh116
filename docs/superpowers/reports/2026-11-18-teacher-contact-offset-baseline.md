# DH116 柔性场景碰撞偏移基线

在独立的 stat-cache 诊断中新增 --contact-offset-baseline。它在保存原始 cooked 数据后、SDK 第 2 步的同一物理边界，使用现有 Physics Tensor SimulationView 为三个指定 link 分别尝试创建一个 RigidBodyView。三处为左手 finger12_link、左手 finger14_link、右手 finger14_link。仅在 prim_paths 精确匹配、count 与 max_shapes 都为 1、两组读回均为有限且合法的 1×1 数组时记录数值。

使用安装版 api.py 的 get_contact_offsets 和 get_rest_offsets，源码路径及 SHA 纳入冻结计划。单位来自当前 USD stage。数据表的主体是 link 唯一 shape slot，原生 shape prim 路径留空。它不会把 cooked polygon 索引映射到运行时 shape。单个 link 拒绝、找不到 view 或时钟跨步时按不可用记录。跨步会丢弃本轮全部测量。原始几何收据先写盘，原严格几何失败仍按原路径报告。单次诊断额外创建最多 3 个临时刚体视图，不建第二个 SimulationView。无控制、求解器、Teacher 或 PPO 步。

CPU 测试覆盖精确路径、单 shape、读回形状与数值、时钟、冻结指纹与 CLI 一致性，以及原严格拒绝语义。实机数值和视图生命周期尚待一次独立原生验证。已有 34 份 cooked 数据的偏移字段只是未解析的 USD 属性，不能替代本次张量读回。

若后续比较偏移对接触的影响，应先确认目标接触由 PhysX 处理，并取得两侧 shape 的读回值。在一项独立的柔性接触实验中，同初态只改变接触偏移，保持静止偏移及其余配置一致，比较首次接触间距、法向载荷、稳定间距和滑移。对于 DeformX SDF 接触，应使用其自身的距离和载荷账本。偏移量取值需要建立在读回和约束上。初始化诊断可独立复核退出，不能作为接触实验结果。PhysX 对两个偏移的语义见 [官方说明](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/AdvancedCollisionDetection.html)。

专项 99 项通过，联合 CPU 561 项通过，测试套件有重叠。
