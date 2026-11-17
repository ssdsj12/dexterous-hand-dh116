# 原 GPU 任务的 Wire 对象侧 PhysX Tensor 探针

任务接触 v02 在首次物理观测前退出。原场景已经绑定，但首次 VecEnv reset 调用 IsaacLab Articulation 的 WrenchComposer，产生 Warp 索引 cuda:0 与数据 CPU 不一致。原 full report v01 的 394 条非零 Tensor 行没有对应完整报告。新探针沿用原 CUDA reset 和已存在的 Torch SimulationView，保持完整报告设置原值。

## 已有原生证据与安装接口

安装的 Isaac Sim PhysX Tensors 实现位于 `isaacsim/extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py`，SHA256 `bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b`。263 行附近定义精确 sensor 列表与逐 sensor 嵌套 filter，4592 行附近是无 filter 汇总 `get_net_contact_forces(dt)`，4624 是配对矩阵，4660 和 4744 是法向与摩擦原始缓冲。法向与摩擦 getter 共享 count/start 缓冲，必须先复制法向数据。dt 为 1/480 秒。无 filter 接口不给出逐 partner 源。

历史 GPU `stage7_mpc_aux_capture06/smoke16_v01` 外层退出 0，128 个物理步 × 8 个环境。独立重放原 gzip，SHA256 `488b655955ee565271ec6fc973292c63341e75a177e1bc11dfdc96ae49602664`。三个静态 view 在 1024 条逐环境记录全部可读。两个端支架各有 1024 条非零法向和摩擦；桌面为 0；full report header 总数为 0。该证据确认原 GPU 任务的静态 filter 可产生非零原始值。

## 新诊断及边界

首次正常 reset 后创建 Wire 对原 34 个手部真实 PhysX actor 的逐环境精确 filter、原有三种静态 filter 及 Wire 无 filter 汇总 view。检查 backend 返回的 sensor/filter 路径顺序、capacity、count/start 所有权、数值有限性和法向矩阵闭合。将对象侧与原手侧独立读取的法向和摩擦分别比较，保留无 filter 汇总减已枚举法向，以及减法向与摩擦的两种候选残差。保留合成场景中未列入 filter 的碰撞 prim、authored force prim、joint 与 attachment 关系，以及 post-step Wire wrench composer。末端 view 或容量验证失败的记录保存错误，不填零。

残差、支持名称、post-step 零 wrench 均无法证明完整 partner 覆盖、上游未截断、无外加力或正确方向与坐标系。保留原资格关闭。最终支持证据还需要精确全场景碰撞来源、原 GPU 模式下的独立反力方向和容量对照、Wire pre-step 到 SDK force 写入的溯源。探针不修改任务、控制器、物理设置、USD 或成功门槛。

## 命令与验证

```bash
bash scripts/python.sh scripts/collect_dh116_teacher_wire_tensor_probe.py \
  --config configs/dh116_training/privileged_teacher_ppo_guarded_v2.json \
  --checkpoint "$frozen_teacher" --checkpoint-sha256 "$checkpoint_sha" \
  --native-task-config "$matching_native_task" --output "$new_probe" --policy-steps 16
```

上式执行 CPU 预检查。经单实例资源确认后，追加 `--enable-experiment --execute-native` 执行有限原生采集。默认 2 环境 ×16 控制步，最多 128 个全局 480 Hz 子步，0 PPO 更新。回调在 App close 前持久化实际结果，外层退出由独立观察器记录。此提交未启动原生。

Ubuntu CPU 回归 104 项通过，日志 SHA256 `d2bb98ecf34adab85cf7571c51c5b5ea2fde61481aaea398a3869be2e68d19ac`。使用实际历史 Teacher checkpoint 验证 CPU 计划通过，未创建原生输出目录。CPU 计划 SHA256 `63f9c4d4baa90ba017471987fe398ad41227b49a6238c16fae4fe8792aef495c`。
