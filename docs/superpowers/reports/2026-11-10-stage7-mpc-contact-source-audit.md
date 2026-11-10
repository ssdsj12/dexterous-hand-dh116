# Stage 7：场景与接触来源测量扩展（MPC 尚无替换资格）

本报告延续 `804214e26fe94db8ef2285dffa4820578a361c88` 的只读统一状态采集。
这是 Commit 2 测量基础设施的后续证据，不是新 Teacher/MPC、接触动力学或视觉 Student 的成功报告。
M0、旧 Teacher、3844D Actor、371D Critic、canonical24 Tanh、原材料/物理/执行器和
30 mm / 5 s / 0.5 mm 成功标准保持不变。PR 继续 Draft，不合并、不更换默认 Teacher、不启动 B3/PPO。

## 本次实现的范围

### 实际场景与被动表面来源

`tabletop_bimanual_mpc_scene_measurements.py` 从已有 composed stage 读取 Wire 实际
Cylinder collider 的尺寸、局部轴、单位和 local-to-body 变换，结合已有 Wire rigid view
的实际 pose、COM、twist、mass 和原始 inertia getter。派生的是**当前刚性圆柱代理**中心线，
不是柔性线缆形变。动态 world pose 不从可能滞后的 USD/Fabric 变换猜测。

Table 的 Xform 根不是 collider；实际 mesh 和两个 EndSupport Cube 分别绑定。
复用已有 ContactSensor 的低层 SimulationView，仅建立三个 Wire→静态表面 view，
不调用高层 sensor 初始化，不写 sleep、stabilization、contact threshold、schema 或物理属性。
实际 `ContactSensor._initialize_impl` 第 260 行确实保存
`self._physics_sim_view = SimulationManager.get_physics_sim_view()`；此处没有新建仿真上下文。

normal/friction 的 count/start、所有权、buffer capacity、finite、matrix 一致性及实际
sensor/filter 路径均分别审计。原始 tuple 包含未拥有 padding 的非有限值时保留原证据，
只有所拥有数据才能参与数值归约。capacity 饱和、重叠和来源缺失不能被解释为零反力。
被动表面的 delivery、作用侧、符号、frame 和完整性尚需独立 Native 资格验证，
因此所有 qualified support/reaction/airborne 字段仍为 NA。

SDK 文本对 inertia 的 COM/actor frame 描述冲突，保留原始 9 列及两种解释。
仅在实际 COM 旋转下两者数值相同的条件下提供共同 world inertia，不能普遍授予惯量 frame 资格。
SDK SI 派生量要求实际 stage meters/kilograms per unit 都为 1；非 SI 不静默转换成有效 SI。

### 真实 contact report 来源

`tabletop_bimanual_mpc_contact_provenance.py` 延迟读取已安装 PhysX full contact report API，
校验其源文件 SHA 与包来源。actual enum 是 `CONTACT_PERSIST`，不沿用过时文档拼写。
header 的 actor/collider uint64 通过 SDK `PhysicsSchemaTools` 做 SdfPath 转换及往返检查，
不将 tensor 行号、Wire filter 索引或手体 body id 当作 partner body id。

全局 report 保留 header、contact 和 friction anchor 原始向量。
异步 callback 没有可证明的 frame，始终不能凭到达时刻授予时钟关联。
同步 poll 则要求完整原物理一步的 World/SDK step 和 dt、episode/reset 身份一致。
原 SDK 尚未证明的单位、作用侧、frame、完整性和 per-env pair 归属保持未知；
全局 report 的 representative env 时钟不表示所有 pair 属于该 env。
原始 impulse、派生 impulse/dt 和合格接触力明确分开。

### 480 Hz 只读辅助采集

`--vision-mpc-aux-capture` 默认关闭，必须同时启用原冻结 Teacher/MPC 的 primary state
capture 和 sidecar，复用原有预 Kit 参数验证。
两路使用同一 run UUID、actually issued 24D action 和 policy tick。

实际安装顺序是 **原 Sequence → aux → state**。原 Sequence 先在未包装的方法上验证原
Shadow 合同，aux 再严格检查该实例的 World/task、保存的原方法、当前控制方法、未关闭状态和
native readback 标志；任意未知或嵌套包装仍拒绝。没有放松原 Shadow 或修改原 Sequence 源。
辅助采集括号包住整个原 `TensorWorld.advance` 调用链的入口，而不是只包住 `sim.step`：原方法先增加
World 计数，再执行唯一物理一步。同步 report 在 task consume 和真实 terminal 判定后、
物理 reset 前读取。没有增加 step/refresh、命令提交或 Actor 输入。
World elapsed 与 SDK time 保留不同原点，检查每物理步与每 8 子步 policy period 的增量。

每个首回合单独闭合；其他环境继续或已闭合槽 reset 后，不覆写其首 terminal。
scene getter 失败仍保存已复制 contact receipt 和 terminal 的 partial frame，异常继续上抛；
raw sink 写入失败则用独立 terminal archive 保留首终止，明确记录 archive-only env，
不增加成功写入的 raw frame/record 计数。close/abort 错误不能跳过其他清理。
raw 非有限值通过 path/kind/捕获的 Python float64 bits 编码，不冒充有效测量。

## 验证与复核

### CPU 实际执行

以下结果来自独立运行，不能相加为独立测试或原生次数。

| 范围 | 实际结果 | 资格边界 |
|---|---:|---|
| Scene + Contact + State，远端独立 CPU02 | 140 passed，41 subtests，1.00 s | SDK 形状 fixture / 纯 CPU |
| Aux hook、来源链与异常持久化，stdlib | 21 passed | CPU；真实 collector 的空 report fixture 也不是 Native |
| CLI / collector 接线与真实 Shadow、Sequence 类 AST | 15 passed | CPU；无 SDK 实例 |
| 新 PRIVATE finite launcher | 38 passed | 所有进程/锁均 mock，不执行 Native |
| 独立 source05 组合回归 | 331 passed，1 skipped，74 subtests，10.87 s | CPU；之后的原生初始化仍失败，见下文 |
| 修正后的独立 source06 组合回归 | 336 passed，1 skipped，93 subtests，8.29 s | CPU；CUDA 不可用项跳过 |
| 关闭归档辅助分析器 | 34 passed；独立复核另跑 34 passed | 仅临时合成 gzip，不计原生次数 |

source05 不是声称与整个 Git checkout 字节一致：它从仍只读的 source04 派生，保留已绑定实际
USD 路径的 asset manifest，叠加 11 个明确代码/测试文件；旧文档不作为本次进度证据。
源与独立 CPU 副本在运行前后逐文件 SHA 一致，四个原 controller 源文件完全不变。

远端 source05 实际命令的测试部分：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
bash /tmp/dh116_stage7_mpc_aux_capture05_cpu/scripts/python.sh \
  -m pytest -q -p pytest_mock -p no:cacheprovider \
  --junitxml=/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_aux_capture05/cpu_validation/pytest.xml \
  tests/test_bimanual_mpc_state.py tests/test_bimanual_mpc_measurements.py \
  tests/test_bimanual_mpc_capture.py tests/test_tripod_native_evidence.py \
  tests/test_tripod_raw_contact_capture.py tests/test_native_sensor_batch.py \
  tests/test_vision_teacher_collection.py tests/test_bimanual_mpc_scene_measurements.py \
  tests/test_bimanual_mpc_contact_provenance.py tests/test_bimanual_mpc_aux_capture.py \
  tests/test_bimanual_mpc_aux_integration.py
```

cwd 为上述独立 CPU 源目录；`cpu_validation/command.json` 绑定实际 argv/cwd。
列示的显式 CPU 环境覆盖来自准备脚本，不是继承环境的完整快照。
CPU 不能验证低层 view 在当前 GPU/静态 collider 配置中的实际 delivery。

| source05 证据 | SHA-256 |
|---|---|
| 完整 source manifest，1486 文件 | `f189ec20fa19fa59b11909316202fe7ddabb258766f550a2f1efba1fb9a8a4fe` |
| CPU validation.json | `cb067691b2731b33ef6e3e1dedc72c3adb90e1a02ac260cf362ca2b5a5cd05bc` |
| CPU pytest.log | `d608757d881c87e20b58fd9136889ffd35e745b94cd9a04e2624658bba700595` |

三份 source05 文件均已下载并核对 SHA；source06 采用同一受限文件清单，另建独立源与 CPU 副本。
source06 完整 command.json 使用对应 `/tmp/dh116_stage7_mpc_aux_capture06_cpu` 和
`stage7_mpc_aux_capture06/cpu_validation` 路径，测试列表不变。
原始 run、模型、archive、运行日志不提交 Git。
发布补充 `.gitattributes` 的 LF 规则，防止 Windows checkout 改变采集器/入口的已绑定源码字节；
这不改运行代码或已冻结 source06。README/CHANGELOG 仅新增进度前缀，原历史尾部字节完整保留。

| source06 证据 | SHA-256 |
|---|---|
| 完整 source manifest，1486 文件 | `36d93e163e2866e24a085179b3ccce224b5510a80fcdd5df0448e07187a747e5` |
| CPU validation.json | `5cf8841e5e75767ed682ce3898eaf98b9b18442b7b791ed4629963de17623410` |
| CPU pytest.log | `58fc2dfdac06b3f889f0843504ade4fa7a91112c1cb797a007f371313c79d4ad` |

### 独立复核发现及修复

1. **P1：真实 sidecar 与原 Shadow 双重检查冲突。** 先装 Sequence 会使旧 aux 的原方法检查失败；
   先装 aux 则使真正的 `Shadow.controller_contract` 拒绝包装。初始测试将后者 stub 为无条件返回，
   未覆盖这一点，source05 原生实验据此失败。补充真实 Shadow 常量、callback 检查及 SHA 函数的
   RED 后，使用 Sequence→aux→core，并仅允许来源明确的既有只读 Sequence 链。
   原 Sequence 和四个原 controller 源不变。15/15 独立测试覆盖正常、core 构造失败、
   core raw close 失败和 aux raw close 失败后的完整恢复；source06 原生复测才实际通过。
2. **P2：scene 异常丢失 terminal/contact receipt。** 先复现 RED，再归档 partial frame，
   错误保留并上抛；独立内存复核通过。
3. **P2：raw 写入异常丢失首 terminal。** 先 pin 独立终止证据，再写 stream，失败明示并不计 raw；
   独立内存复核验证 frame=0、terminal=1、archive-only=[0]。
4. **P2：finite launcher 的异常 outcome 缺 censored 范围。** busy lock、已有进程与 child 启动异常
   分支均新增 RED，再在 outcome 初值固定 bounded/censored=True，38/38 及独立复核通过。

这些修复不表示运行失败会变成任务成功，也不改变任何接触、执行器或物理阈值。

## 实际 Isaac 结果与失败证据

### 原 source04 完整首回合采集

`stage7_mpc_state_capture04/full1900_v01/native` 自然退出，process/launcher 均为 0；
请求上限 1900，原算法有效 cap=min(1900,1272+600)=1872，实际 1800 控制步即全部结束。
14,400 个原物理子步、84,461 条活跃首回合状态、8 个真实首 terminal，无截断回合。
关闭后的两份 gzip、SHA/CRC、DTO、实际动作、时钟、UID 和 reset/no-op 检查通过，failures=[]；
30,739 个已闭合槽未被重新纳入首回合。

| 原 Teacher 抓法 | 首回合结果 | 实际结束 |
|---|---:|---|
| 二指，env0–3 | 4/4 原生成功 | 第 1174/1175 控制步，原成功判据不变 |
| 三指，env4–5 | 0/2 | 第 1800 步原生 timeout，failure code8 |
| 三指，env6–7 | 0/2 | 第 1092/1169 步 object_dropped，failure code7 |

这四份每抓法案例仍来自相同 nominal 来源的克隆，不是至少 100 个独立初态；
也不是新 MPC 或 Student 的成功。分析器没有独立的物理 reset 高度参考，因此其 net Lift 为 NA，
不把首个 post-step 样本或绝对 bottom 减 0.30 当作净抬升（后者会包含支架预抬）。

| source04 完整证据 | SHA-256 |
|---|---|
| teacher_collection_report.json | `6201908c49c23d578f16dc980a961b35bacfed337f994aa25d823c7f496770ee` |
| native_state_capture_report.json | `bf1d60f168136ee21e6a2ce2a6122c40ee7c1e2bad6e4a47f473ce1fd534656f` |
| full1900_analysis_v01/analysis_summary.json | `36f7ccce364a3557daa010047c0066cb6806079c90968c526c7deaca70a24bab` |

### source05 失败与 source06 独立复测

source05 的 `smoke16_v01` 已显式执行一次，process/launcher exit2；真实 SDK 和三个低层 view
构造之后、Teacher 控制采集前，因原 Shadow 拒绝替换 callback 退出。物理采集帧=0、terminal=0。
其源、输入、失败目录与 summary 原样保留，没有在同目录重试或覆盖。
失败 aux summary SHA 为 `d790f6edfb29edce8c571b849d0b88b13d687cb60edf82e67c8a6a8b27767ce2`。

修复后另建 source06，通过上述独立 CPU336 检查后显式执行一次：
`stage7_mpc_aux_capture06/smoke16_v01/native`，process/launcher exit0，
16 个实际 RGB-D/控制批次、128 个原 480 Hz 子步、每环境 128 条状态，共 1024 条。
全部 8 回合在短窗口内仍开放，不能写成完整回合 0/8 失败或成功率验证。
PPO updates=0；collection wall time=166.074 s，**没有 60 Hz wall-clock/deadline 资格**。

关闭后使用独立 matching-source 分析 capsule；core 和 aux 完整性、共享 UUID、实际 issued24、
每物理子步、采集实现 SHA 和两份 gzip 校验均通过，无 provider/raw write error。
分析器的退出码 0 只代表归档检查通过，绝不代替 runner 的 SDK 身份或测量/任务资格。

| source06 实际读取 | 128 帧 / 1024 活跃环境槽的结果 |
|---|---|
| Cylinder 尺寸/轴、刚性中心线/速度、COM、mass、pose、raw/world inertia | 每字段 1024 个可用观测；不是柔性测量或一般惯量 frame 资格 |
| 三个静态 view 的路径/normal/friction/count/start/matrix | 每 view 1024 个读取及所有权检查通过 |
| 两个 EndSupport | 每 view 的所有活跃槽均有非零 observed matrix；不授合格承托力 |
| Table | 1024 个 observed matrix 为零；不能单凭零证明无反力或无支撑 |
| 全局 full contact report | 128 个同步 receipt，但 header/contact/anchor 全部为空；不能据此声称没有实际接触 |
| qualified support、partner body ID、ON_OBJECT sign/frame/SI/completeness | 继续 NA / 未资格化 |

原生辅助采集创建链正确报告 `SequenceContextCapture->original`。
全局空 report 与端支架实际非零低层观测并存，说明其真实交付/报告缓冲条件还需调查；
当前不补造 pair IDs、接触年龄、逐指合格摩擦或支撑值。

| source06 原生/分析证据 | SHA-256 |
|---|---|
| 16 步 command.json | `9900a27ea6ce76752a6fc39822ea543ac0cdc7be596a4151c790e68aad123688` |
| aux_capture_summary.json | `7fb1a0a33d9b2a3ec35d1b20f6ba5fb1f4d4824d5cf9d1551f3d80cb5e27e398` |
| aux_480hz.jsonl.gz | `488b655955ee565271ec6fc973292c63341e75a177e1bc11dfdc96ae49602664` |
| smoke16_analysis_v01/analysis_summary.json | `2a6407dd7fded70ea0e58cb9015e511adf78be518c7d7019ae1ba6b807d309df` |
| smoke16_analysis_v01/analysis_details.json | `6fe77199061602e05e50ccd21dde18cf4e05c06491fcbf75e68267b749f43ac5` |

各 Native 作业均在前一实例自然退出后显式执行；全局非阻塞锁和 preflight 生效，未建立自动队列。
实际 root 命令（完整输入/source/CPU SHA 在各 command.json 中，旧结果未被后一次替代）：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -B \
  /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_aux_capture06/run_stage7_native_aux_capture.py \
  --execute --plan /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_aux_capture06/smoke16_v01/command.json

/mnt/mydisk/xk/envs/dh116/bin/python -B \
  /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_aux_capture06/analysis_capsule01/artifacts/analyze_stage7_mpc_aux_capture.py \
  --run /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_aux_capture06/smoke16_v01/native \
  --output /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_aux_capture06/smoke16_analysis_v01
```

## 未完成与下一步

- source04 的首 terminal/closed cohort/reset/no-op 与 source06 的短窗口测量归档已通过；
  进一步调查 full contact report 的 buffer/subscription/step 交付条件及实际 pair 绑定。
- 完成低层静态 filter、ON_OBJECT frame/sign/SI/completeness 和合格承托的独立 Native 门槛。
- 后续补齐全 D articulation / backend drive / M/C/G 的独立实际读取；它们未进入 source05/06。
  PD estimate 不等于隔离的 implicit motor torque；当前状态 getter 不是未来 candidate dynamics。
- H8 / 64 个原始物理子步、candidate-dependent 接触/刚体预测、完整非线性/swept 硬约束、
  求解器和总 16.67 ms deadline 尚未实现/资格化；不可由只读采集器冒称完成。
- 双侧逐指恢复、载荷转移、共同暂停完整预算、新旧 Teacher 每抓法至少 100 个真正不同的配对初态
  尚未完成。没有新的专家池、B3、Student PPO 或默认 Teacher 替换资格。
