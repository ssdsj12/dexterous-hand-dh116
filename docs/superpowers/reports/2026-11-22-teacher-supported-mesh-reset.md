# Tripod 17 槽零步 Mesh Reset 与 Readback

## 范围

基线 `86b61f6dbf8b309e9d19f1c1f739cae01d92cf0e`。为现有柔性三指任务增加独立的17槽构造计划、零步 clear/add 重建和实际 mesh_state_snapshot 核验。不是新增资产或改建物理场景，不启动 Isaac，不授予抓取资格。

## 根因与实际代码

旧的延迟 `set_mesh_poses(teleport=True)` 只登记 pending target；首次移动目标可能覆盖该标志。因此不能靠“做一次额外静置物理步”冒充正确的 fresh Reset。已有11槽 clear/add 已证明底层 `add_mesh` 会提交本槽，但旧读写器和容量固定为11/5412。

新增 `tabletop_supported_mesh_reset.py`：

- `prepare_supported_mesh_reset_plan` / `verify_plan_sources`：独立校验原10个 distal 网格和桌面；按原生捕获、锁定 SDF 和原 USD 来源加载掌面/近端4网格；重算原支架网格，保留摩擦、原点、间距、位姿和严格17槽顺序。
- `MeasuredSupportedMeshReset._rebuild`：复用原完整手部 Reset 事务与已核验的 clear/add 算法，逐槽提交14个实际手部 link 加桌面和两支架。原SDK/solver时钟和41节点参数、位置、速度、director 状态必须不变。
- `capture_supported_mesh_state` / `validate_supported_capture`：读取真实同一 RopeBatch 的 current/target/solver 位姿、pending flags、pair capacity=8364和所有权，不用目标值替代测量值。
- `validate_supported_readback`：逐14个 link 对齐实际 post-Reset SDK body 副本；三静态槽对齐封存构造位姿；核验版本、源SHA、重建前后数值状态、时钟、身份、完整零步计数。current/target/solver 全部一致且无 pending 才能通过本读取合同。
- 失败时保留已完成的手部 Reset 和取得的原始 readback，poison rod、拒绝重复事务。未知的失败阶段步数保持 null。原异常不会被二次记录失败替换。

旧 `tabletop_mesh_reset_order.py` 和 `tabletop_mesh_reset_readback.py` 保持已发布原始字节与SHA：

- order: `94394cd6a10c528422bda28f7ee8813587ba1ca970dff54d9eabcc41ab19b50a`
- readback: `53c362e365df241280caa8d89ebd821a8a541d7c62c845d4956a6c12530ca0cc`

旧11槽接口继续拒绝17槽数据；未改 BackendProfile，未降低其 SHA 校验。

## TDD 与失败证据

1. 新17槽入口缺失的红测先失败。
2. 末近端追加失败测试发现原已完成 Reset 记录丢失；实现独立 owned receipt 保存后修复。
3. 最初尝试共享重构旧源文件，导致 BackendProfile 的10项来源装载错误。已还原两份旧源文件原始字节，改为独立17槽版本；保留失败日志，不改旧清单SHA。
4. 追加省略 implementation inventory、伪造 plan/rebuild version 和零步计数的负例，确认原实现未拒绝，再补齐严格校验。
5. 独立版本曾缺少 math import，3项测试失败；修复并完成最终回归。所有中间失败日志仅保留在服务器。

四个新增测试分别覆盖真实17槽首次移动起点、readback/计数篡改、局部重建失败、真实锁定构造来源。CPU双桩只模拟 SDK/solver 状态机，真实 SDF/捕获字节使用 SHA 核验。支架数组另与原有 `make_support_sdf_grids` 逐值比较；未用测试桩证明 GPU 或接触成功。

最终命令、数量、耗时和日志 SHA 见 readiness JSON。原生运行0、PPO更新0；未执行全仓库测试。

## 尚未完成

- fresh supported episode、七路反作用力 exchange 与本17槽 Reset 的正式接线。
- 独立 Tripod source/factory/preflight；复用既有真实 PPO 的 update1/resume2。
- 本版本 Reset 仍明确限定原有 SDK≤240 的时钟范围，不宣称支持完整长时训练 Reset。
- 新17槽原生重建、接触、自然退出和三抓法成功均待验证。当前通过的是源码/CPU合同，不是原生资格。
- Pinch/Tripod/Power 必须分别达到实际对向接触、固定腕保持、完全离架后 Lift≥30mm、Hold≥5s、Slip≤0.5mm。
- 双固定 RGB、物理参数、原 USD、24D/371D、Mimic、关节保护、R2关闭状态与历史模型均保持不变。

Antigravity 的既有R1最小独立接线建议作为设计依据；本补丁没有获得新的外部审查通过回执。Codex 独立核对依赖、旧 SHA 和回归证据。


## 最终执行结果

- Reset/构造来源/旧读写器/三指距离：**81 passed / 0 failed，140.98s**。
- Teacher preflight/session：**91 passed / 0 failed，47.12s**。
  命令：`python -m pytest -q tests/test_native_teacher_preflight.py tests/test_native_teacher_session.py`。
- 上述9个文件为不重叠的172项相关 CPU 测试；没有声称全仓库通过。
- Registry：**60 passed / 0 failed，15.31s**。
- 4个新增测试；全部中间失败与修正后日志 SHA 见 readiness，原始日志留在服务器。
- 本包无原生实验，无新的求解器或 PPO 更新。
