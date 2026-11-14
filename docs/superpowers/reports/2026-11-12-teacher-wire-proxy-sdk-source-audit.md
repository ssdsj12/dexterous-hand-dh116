# Wire 实例代理与原生 SDK 外力调用审计

上一次原 T5 GPU 探针给出了对象侧法向／摩擦闭合。常规组合场景遍历仅发现 8 个 CollisionAPI prim，Wire wrench composer 仅在物理步后读取。本包将这两个缺口分别接入既有有限 Wire 探针，适用同一任务和同一检查点。

- 通过 `Usd.TraverseInstanceProxies()` 读取实际组合场景，包括实例代理下的 CollisionAPI prim。逐 shape 保存启用状态、来源 layer、是否为实例代理，以及最近的 RigidBodyAPI actor。对照已核验的精确手部 actor、Wire actor 与三个静态 collider filter。未列入 filter 且启用状态为 true 或未知的路径保留。记录 PhysxForceAPI、ForceField schema、joint 和 attachment 及其关系。
- 在首次正常 reset 之后，对真实 Wire root PhysX view 的 Python `apply_forces_and_torques_at_position` 调用建立独立 tap。原始参数和返回／异常语义沿用原方法。记录实际 force、torque、position、indices、global 标记、目标 Wire actor、调用前后参数值和 World/SDK 时钟。只有 SDK 步号和时间确实对应下一物理步的调用才会归入同一步。旧时钟及未关联调用单独保留，回执核对调用总数闭合。
- 原探针继续保存原生碰撞力残差、步后 wrench composers 和原物理日志。SDK 调用 tap 覆盖所绑定的具体 Python view。其他 view alias、插件、C++ 直接写入、力场与上游物理完整性仍需独立证据。即便没有记录到该方法调用，外力缺失和正式 force/support 资格仍保持关闭。

Ubuntu CPU 回归 **107 项通过、0 失败**，日志 SHA256 `9858ae2bcac3d1ca7548a73538e8504303bf1ed0493d9016657e1941a3e23d1b`。新增测试覆盖实例代理 actor 归属、启用状态未知或未列入路径、参数对象原样传递、严格下一步关联及 SDK 异常保留。独立使用已安装的 USD 在内存构建内部引用及实例代理，确认实际 `/World/TestActor/collider` 解析至正确 actor；无 Isaac 进程。运行期 PhysXSchema 及 SDK 写入源需要原生短实验复核。

继续使用 `scripts/collect_dh116_teacher_wire_tensor_probe.py` 的同一有限命令，使用已核验的 T5 R0 Checkpoint、独立新目录、2 环境及最多 32 控制步。本提交未启动原生，未改变控制器、USD、物理设置、保护或成功门槛。
