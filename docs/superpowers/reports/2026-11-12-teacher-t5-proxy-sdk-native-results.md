# 原 T5 R0 任务实例代理与 Wire SDK 调用原生复核

新探针在 07:01:40–07:02:21 UTC 运行，外层自然退出 0。二指、三指各 1 个环境，16 个 60 Hz 控制步，128 个连续 480 Hz 物理步。World 步号 1–128，SDK 步号 19–146，256 条环境记录的时间、身份、策略步与终止和原生物理账本逐值一致。沿用原 T5 任务和已核验的 R0 update-1 Checkpoint `71ea209f6710db9edc9457e5ed3bfde8cecc9daede4e0dd5eff0167e905e6638`。无额外训练或完整原生成功回合。

| 核验项 | 实测 |
|---|---|
| 完整组合 USD 路径 | 普通遍历为 8 个 CollisionAPI prim，实例代理遍历为 76 个，其中 68 个实例代理。全部启用且有声明的精确路线，其中 70 个归属实际刚体 actor、6 个归属静态形状。当前组合场景未发现启用或启用状态未知的未归类路径。 |
| Authored 外部来源 | 组合场景未发现 PhysxForceAPI 或 ForceField schema。92 个 joint/attachment 候选的关系中未发现指向 Wire 的路径。 |
| Wire root SDK 调用 | 实际绑定的 Python view 上捕获 0 次 `apply_forces_and_torques_at_position` 调用。同一步、较早未匹配、窗口后未匹配各 0。逐物理步和回执计数闭合。 |
| 对象侧与手侧 | 法向反力最大残差 5.96×10⁻⁸ N，摩擦逐元素相消。无过滤合力减已枚举法向最大残差 2.46×10⁻⁷ N。 |
| 同 Checkpoint 前后观测 | 与先前原 T5 探针 DsBg 比较，手部和三个静态组的法向和摩擦、无过滤合力和两种残差、下发 24D 动作、物体净抬升与实际掌位全部逐值一致。 |

此次记录覆盖既有冻结任务的约 0.267 秒受支撑窗口。实例代理归属有真实组合场景证据，碰撞组间交互、上游容量、无过滤摩擦及完整成功回合仍须独立核验。SDK tap 覆盖这个具体 Python Wire root view 的指定方法。其他 Tensor alias、插件或 C++ 写入及未观测的力应用路径仍在资格范围之外。原生力学、完整外部支撑排除、正式专家准入和 Teacher 成功资格保持关闭。

## 独立审计

Ubuntu 私有原始目录 `teacher-t5-proxy-sdk-20261112-jaYqX3`，独立只读审计 `/tmp/dh116-t5-proxy-sdk-audit-BPUXST`。冻结快照探针入口和实现与任务提交 `7d90456d7f423dfba2bf22233b06a99d471231eb` 逐字节一致。

| 证据 | SHA256 |
|---|---|
| Wire Tensor gzip | `903a2673228a19620399684c1172105231692bd5cc376611173779155524b312` |
| 实例代理清单 | `df7b076c4d3523af75be261f5e07d65f561515af2d927be4f6e25331ed699f87` |
| Wire root SDK 未关联调用文件 | `28ab80e01865a689e144111235b88505658d0de4c36a55e808c4b1f2a8d97ffa` |
| 回调完成文件 | `586405e5fce6181c7de23c7c7c238492c3b11a478db9a83eebc1123e485919ed` |
| 原生物理账本 | `b10fe2f226f6a40142c0f2547d47a8f18b2b0a0ce26546ae480e7df4ffb5bb76` |
| 外层日志 | `cb106e3514c32b3dd4543ef57426f0fa3ca09971308d5a88ae4d09210af3de5f` |
| 独立 SDK／实例代理审计 | `4501c6e5b4526d0f7a3b4ff462b08de99c2764bb2607ab7564088eb40d27bb94` |

Ubuntu CPU 回归 107 项通过、0 失败，日志 SHA256 `b466959656fee7f08c804843cf0c8be6e3d317e353b41c9ea334103da67932be`。本报告复核已有运行，无新增原生启动或资格改变。
