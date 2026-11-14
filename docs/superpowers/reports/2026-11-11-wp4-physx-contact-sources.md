# WP4：PhysX 接触来源、完整采样与退出问题审计

日期：2026-11-11。基线：`ae307a55917519138b6da5179e8254ce2e9a7c8f`。

## 结论与资格边界

两次已有标定都完成了要求的 **240 个物理子步**，原始 JSONL 完整。挂起发生在采样、汇总之后的应用关停阶段；没有证据表明它们在第 240 步以前退出。两次最终 launcher `exit_code=0` 均是在操作者对该标定子进程发送定向 SIGTERM 后取得，**不能作为无干预自然退出通过**。

这两次实验仅使用左 `finger14_link.STL` 的固定运动学安装座与重力下落刚性球体，没有完整 DH116 Articulation、右手或双手抓取。Tensor `dt` 缩放和该单对刚体的动量残差已有可复核数值，但 `get_full_contact_report()` / full callback 没有提供实际 pair 数据，作用侧、逐 collider 身份及 full-report 摩擦语义未通过资格。**R2 与 Force Proxy 继续关闭。**

本次只修改代码、运行 CPU 测试、读取已有原生数据。新增 ordinary/full 双 callback、公开生命周期清理和跨 run episode 身份修复都 **尚未进行新的 Isaac 原生复验**。没有启动新的同步采集或训练。

## 实际安装 API

读取服务器现有 Isaac Sim 4.5 安装源码，不借用其他版本的接口声明：

| API | 实际来源 / 结果 |
|---|---|
| Tensor `get_contact_data(dt)` | 同一物理步分别读取 `dt=1` 与 `dt=1/480`，复制拥有的 counts/starts 范围，再比较 `force = impulse / dt`；无夹入额外 simulation step |
| Tensor `get_friction_data(dt)` | 独立读 friction anchors；保留其数据和 capacity，不能将 anchor 数等同普通接触点数 |
| `get_full_contact_report()` | installed `_physx.pyi:1292`；三数组 headers / contact data / friction anchors |
| `subscribe_contact_report_events(fn)` | `_physx.pyi:1380`，**两数组** headers / contact data；没有提供摩擦数组 |
| `subscribe_full_contact_report_events(fn)` | `_physx.pyi:1392`，**三数组** headers / contact data / friction anchors |

`_physx.pyi` 完整路径：`/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physx/omni/physx/bindings/_physx.pyi`。

SHA256：`b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9`。

安装的 `ContactReportDemo.py` 同样区分 ordinary 两数组和 full 三数组。两类 impulse 的文档描述不能单独证明 contact impulse 与 friction anchor impulse 可直接相加，因此代码分别保留 `contact_only`、`friction_only`、`contact_plus_friction` 假设；未经动量与分量验证，不授予任何假设真实力来源资格。

## 既有原生数据完整性

远端独立目录：

- `/tmp/dh116-stereo-wp4-contact-v02`
- `/tmp/dh116-stereo-wp4-contact-v04`

本地原样副本：`artifacts/stereo-wp-20261111/contact_failures/` 下同名目录。原始文件未重写。独立重新统计保存在 `artifacts/stereo-wp-20261111/teacher/wp4_existing_raw_audit.json`。

| 检查 | v02 | v04 |
|---|---:|---:|
| 完整 JSONL 行数 | 240 | 240 |
| 不可解析行 / 缺少结尾换行 | 0 / 否 | 0 / 否 |
| 实验物理步 | 1–240，连续 | 1–240，连续 |
| SDK 物理步 | 3–242，连续 | 3–242，连续 |
| 首个记录 SDK 时间 | 0.006250000325962901 s | 相同 |
| 最后记录 SDK 时间 | 0.5041666929610074 s | 相同 |
| 相邻时间增量对 1/480 s 的最大残差 | 1.0865430040071788e-10 s | 相同 |
| 有 Tensor contact range 的步数 | 36 | 36 |
| Tensor 合力非零步数 | 30 | 30 |
| contact data 条数 / 单步最大 | 36 / 1 | 36 / 1 |
| friction anchor 条数 / 单步最大 | 38 / 2 | 38 / 2 |
| 保存的 full callback receipts | 0 | 0 |
| poll 输出 | legacy 展平 pair 0 | headers / contacts / anchors 均 0 |
| launcher 返回码 | 0，SIGTERM 辅助 | 0，SIGTERM 辅助 |
| launcher 总墙钟 | 202.495024 s | 223.364625 s |

SDK 起始非零来自 reset / 初始化阶段已经推进的两步。采样覆盖 240×1/480=0.5 s；时间戳为 SDK 浮点累计值，不应因约 1e-10 s 的增量误差误判时钟断裂。实验 `physics_step` 是采样序号，不能冒充 SDK 原始 step。两次逐步 pose、velocity_before、velocity_after、sdk_step 与 timestamp 都相同。

旧 run 的 `episode_uid` 都是 `left:left:0`，**跨 run 会冲突**。解释旧数据必须同时带 run 路径 / raw SHA；禁止把两个文件仅按该 UID 合并。本次新脚本采用独立完整输出路径 SHA + side + episode 序号，既有数据保持原样。

### 原始文件 SHA256

| 文件 | SHA256 |
|---|---|
| v02 `left/raw_contact_480hz.jsonl`，443686 bytes | `981a40e256a9bb810799ebdbd30c8bbdeed432b6d1e2ef1be75f4c1954903877` |
| v04 `left/raw_contact_480hz.jsonl`，568726 bytes | `f70a7d8fec08093a2c6a11e634b2b117162af5ec2bc61c2eed7ecd37b3b23064` |
| v04 `left/shutdown_stacks.log`，9306 bytes | `80f21b72179ea1ef1492530d9b9a14d779855a16a857e9f648eed2f942ee94e9` |
| v02 `left_launcher.log` | `5923e25011b819ced745eb7e4ca820cfcf33f33d3ed270bbcf82b415fd3335b6` |
| v04 `left_launcher.log` | `e90aeaf6b3bd50f4ccff38213c9420358cbd27689de0d71839d2dafea7983ce9` |

v02 的旧展平格式不能追回当时可能遗漏的 header-only / anchor-only 事件；因此不能把 `pair_reports=[]` 声明为完整底层三数组为空。v04 保存的是完整三数组，三者实际均空。两次 callback receipts 均未返回记录，不能推断普通 callback 已得到验证，也不能推断零 callback 就等于零接触。

## 关停时序与根因证据

UTC 时间来自服务器文件时间和 v04 明确记录的 Unix 时间；v02 缺少内嵌关停时间，mtime 仅作为时序证据，不冒充精确采样墙钟。

| 事件 | v02 | v04 |
|---|---|---|
| raw 最后写入 mtime | 10:25:59.290329 | 10:33:07.995479 |
| summary 最后写入 mtime | 10:25:59.296329 | 10:33:08.011479 |
| `shutdown_started.json` | 旧版未记录 | time_unix_s=1794393188.012352（10:33:08.012352） |
| SIGTERM 干预记录 | pid=1488384；`shutdown_timeout.json` mtime 10:29:09.971186 | pid=1494916；time=1794393395.4443996（10:36:35.444400） |
| exit 文件写入 mtime | 10:29:10.346182 | 10:36:35.808095 |
| `shutdown_complete.json` | 不存在 | 不存在 |

v04 在 `shutdown_started` 后约 **207.432 s** 才记录定向 SIGTERM。重复 faulthandler 栈指向：

```text
SimulationApp.close(...), simulation_app.py:581
  -> IsaacLab SimulationContext STOP callback, simulation_context.py:252
  -> _app_control_on_stop_handle_fn, simulation_context.py:1029
  -> self.render()
```

现有安装 `simulation_context.py:1027–1029` 在 STOP 后循环 render 直到 timeline 再次 playing。`wait_for_replicator=False` 没有解除这个 STOP 回调，因此 v04 **已反证**“只跳过 Replicator 等待即可解决关停”的推测。

v02 的旧 summary 含 `native_exit_complete=true`，写入发生在 `app.close()` 之前；该字段命名和含义错误，不能作为进程已退出证据。原文件不修改，新版改为 `sampling_complete=true`、`process_exit_verified=false`，真实 exit 由外层 launcher 单独记录。`exit_code=0` 必须与 `shutdown_timeout.json` 一起解读。

日志另有大量 `Failed to create change watch ... errno=28/No space left on device`；这是 watcher 创建错误，不能仅据字符串认定数据盘写满，也不能据此认定是 STOP render loop 根因。原始日志完整保留。

## Tensor 测量与仍缺失的资格

两次实验的最大 `|force(dt)-impulse(dt=1)/dt|` 为 **1.4901161193847656e-8 N**。该单对刚体 sensor-side 合力的最大动量残差为 **4.348754785594155e-9 N·s**。v04 正反传感器分别累积 normal / friction / total 的 action-reaction 残差均为 0；这是每步合量对照，没有逐 contact 跨 view 配对。

v04 `/physics/disableContactProcessing` 从 `true` 显式设为 `false`，`/physics/suppressReadback` 实测仍为 `true`。poll 和 full callback 仍为空，不能声称 reporter 开关已经修复 pair 来源；也不能未经实验就断言 suppressReadback 是唯一根因。

以下仍未证明：

1. 原生 pair headers 的 body A/B、actor 与 collider 的逐侧身份。
2. full-report impulse 的 on-object 符号，以及 contact impulse 是否包含 friction 的真实语义。
3. callback 真实发射条件、采样时钟归属、ordinary/full 能力各自的实际返回。
4. 右手、活动手 Articulation、逐指、双手接触的来源资格。
5. 抓取 episode 的桌面/支架承托排除、attachment 和外力完整审计。

标定的 fixed mount 明确存在；无 authored force 属性不等于已经证明真实抓取没有外部承托。尚无该来源下的抓取成功或 Teacher/Student 成功证据。

## 本次实际代码修复

### `tabletop_stereo_contact_calibration.py`

- `capture_tensor_pair`：保存每个数组容量、是否触达边界、未知 overflow 状态；同一步 dt 审计检查 ownership / geometry 未变。
- `decode_contact_report`：复制完整 header/contact/friction arrays，保留 offsets、actor/collider/stage 与事件类型；支持旧 full 三数组和 ordinary 两数组。ordinary `friction_anchors=null`、`friction_source_available=false`，不能补成测得的零摩擦。
- `summarize_pair_calibration`：保留旧 `pair_reports` 解析；side-correct actor→collider 匹配；分别输出三种 impulse 假设、normal/tangent 分量和 reverse Tensor 合量。callback 只单独计数，绝不与 poll 或 Tensor 加总。
- `ContactReportReceiptBuffer`：拷贝 SDK callback 数据避免缓冲区复用；分别标识 ordinary/full、arrival SDK clock、当前 sim.step 请求窗口。窗口关联不冒充 contact 内嵌时间；reset/outside-step callback 保留未知请求 step。
- `close_calibration_app`：在确认 singleton 属于本标定后，调用安装版本公开 `SimulationContext.clear_instance()` 释放 STOP callback，再 `app.close(wait_for_replicator=False)`；写开始/清理/返回标志和 faulthandler 栈；拒绝清理其他 owner；保留并重抛 `ISAACLAB_CALLBACK_EXCEPTION`。不修改安装源码、不调用会先触发 STOP 回调的 `stop()`。
- `calibration_episode_uid`：完整独立输出目录 SHA + side + episode，消除同名 `left` 目录跨 run 冲突。

安装的 `SimulationContext.clear_instance()` 673–680 行先 unsubscribe STOP handler，再调用父类公开清理。安装源码 SHA256：`d95f2eaa138b4fff845ac72439caa2eca5deab4f0d5609e06f163d20eef4df93`。这支持修复路径，但仍需原生自然退出复验，CPU 模拟不能替代该门槛。

### `scripts/calibrate_dh116_stereo_contacts.py`

- `--full-contact-callback` 与 `--ordinary-contact-callback` 均为显式 opt-in，分别调用正确 API，独立 receipt channel。
- `--enable-full-report-processing` 独立控制 reporter 开关，只作用于隔离标定。
- inventory 记录请求的 callback 通道及 episode 身份；每步保持 Tensor / poll / callback 三类证据独立。
- `finally` 先释放本地 callback subscription，再以自身 `SimulationContext` 调用清理函数。
- 不将采样结束等同进程退出。

## CPU 验证

`tests/test_dh116_stereo_contact_calibration.py`：**26 passed in 0.31 s**。新增失败测试先于实现，private logs 位于 `artifacts/stereo-wp-20261111/teacher/wp4_*.log`。

覆盖范围包括：dt scaling/ownership、capacity 边界、anchor-only/header-only、接触/摩擦重复计算假设、actor-collider 侧别、reverse action-reaction、旧 raw 兼容、close 失败不写完成、正确释放 owned STOP callback、其他 owner 拒绝、callback 错误不吞掉、ordinary friction 缺失、callback buffer 拷贝及窗口、callback 不双计数、跨 run episode 身份。

此次 CPU 验证不是新原生接触结果，也不是 hand/force qualification。按用户要求本轮不继续原生采集；下一次获授权后，需先验证独立自然退出，再检查 ordinary/full/poll 返回、body/collider、方向和单位，资格通过前不得推进完整手或开启 R2 / Force Proxy。
