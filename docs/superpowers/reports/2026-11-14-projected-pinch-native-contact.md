# 原安全投影 Pinch：真实接触与自然退出结果

## 实际执行与边界

本轮只执行一次新冻结配方。GPU3、单Isaac、1环境，**234总SDK步（2初始化+232测量）、0PPO**。外层与GDB inferior均自然exit0，无信号；用时**163.496670751s**，小于180秒预算。app.close正常返回。父调度等待先返回pending，随后只接收同一次worker的退出后审计，没有重启、重复运行或扩预算。

- 执行HEAD：`24eef09c230b3ee5ed50ba855b51a6584400e521`。
- recipe：`129e0b0e3455e276733554d15c1ccbd431772dc04572e168943ed3d327109254`。
- scheduler code/config：`d2b175b7654b4b30c67028a20a6b0e1a50eda98b3804414554c6d8c9fd3f7d58`。
- receipt：`04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3`。
- physical.jsonl：`9cec6ed43c06f23cf23b327a22009c7f1b3a98574de27883ba89c60b0bf255d4`。
- post-exit audit：`d36d9c35b2cda842e3ce264130dd7a3b085747bb53dd7063df11711d8642be5c`。
- 资格仅为 `pinch_original_safety_projection_2_setup_232_diagnostic_only`，不是抓取/Teacher成功。

## 原动作与安全投影

原动作、原初始姿态、原材料、原cooked碰撞形状、原joint/mimic保护完全不改。新实验接受且逐条重算原控制器合法投影，未放宽物理门槛。

首次投影仍在SDK176、policy21/substep5，env0、reset1、
episode `dual-pinch-555cbae972d144ba9b8a9458579dfc26:one`，
时间0.36666666666666664s。36条测量记录有左侧safe_z投影；最大单步Z修正0.002325894mm。右侧未投影。
末步投影目标World偏移：左Z -1.435799535mm、右Z -0.500000000mm。这是控制目标偏移，不能当实际掌心位移。

上次SDK176的实验校验中断已被修复。此次从实测源数据重算投影并完成全程；原safe_z自身没有被修改。

## 接触结果：左食指仍未进入有效区域

| 指腹 | raw pair样本 | 原有效patch内样本 |
|---|---:|---:|
| 左拇指 finger14_link | 314 | 300 |
| 左食指 finger23_link | 428 | 0 |
| 右拇指 finger14_link | 1100 | 276 |
| 右食指 finger23_link | 1116 | 312 |

合计2958个raw pair，888个在原patch内，2070个因 `outside_inner_ellipse` 被拒。
左食指ellipse值1.122299996～1.128947406，始终大于原阈值1；不能把它的428个raw pair计入有效接触。
右侧两指同时在patch内连续276个实际solver子步，区间slot2508～2783，对应SDK212～234，时长0.047916667s。
左侧同时有效窗口0，双侧同时有效窗口0。

这里统计原区域归属，不证明法向对向、重量承托或固定材料锚滑移。求解器力为仿真测量，不能冒充硬件传感。
仍无30mm/5s/0.5mm原生抓取成功，无Teacher更新，无Student成功。

## 实际cooked hull桌面间隙

下表为整段最小World间隙，均正值、测得穿透0；原保护未取消：

| 左指腹 | 最小间隙mm | SDK步 |
|---|---:|---:|
| finger14 | 3.003336 | 232 |
| finger23 | 3.298033 | 224 |
| finger33 | 38.831943 | 232 |
| finger43 | 55.864464 | 229 |
| finger53 | 72.739563 | 229 |

右finger14/23最小间隙分别3.127406/3.269874mm。每项在分析JSON中绑定同一个env、episode、policy、physics、timestamp，不拼接为同一双手轨迹。此轮未再次调整初始姿态。

## 所有权与退出证据

before_stop：左右Articulation对象、各自root Tensor View、WrenchComposer和Rope/runtime弱引用已消失；共享asset SimulationView仍存活。
after_stop：两侧asset SimulationView弱引用也消失；SimulationContext仍有构造帧引用。显式subscription unsubscribe与原clear_instance返回，extra_gc_calls=0。
这些弱引用和函数返回记录本身不等于native析构/GPU释放已被逐项证明，原 `native_release_verified=false` 标记保留。

外层自然exit0、app.close返回和GDB正常Py_FinalizeEx停止点独立保存；GDB在该点停止136线程并保存栈、maps、libraries和registers，无SIGSEGV。
本结果只覆盖当前无相机owner，不能宣称历史C5相机入口已修复；未重跑C5。

## 可复现只读分析及测试

新增 `scripts/audit_dh116_projected_pinch_contact.py`：
- `run`验证真实receipt SHA、配方/代码、退出审计和原生账本SHA。
- `projection_summary`核验完整232步身份与投影，保留首次发生时刻。
- `summarize`复用旧contact_summary与连续窗口计算，分别报告两侧patch、间隙和限制。
没有新物理执行，不提高native/grasp/Teacher资格。

先7项失败（缺模块），随后14项单元通过；再加真实冻结账本回放与旧回归，**21项CPU测试通过，29.93秒**。
一次调用写错旧测试文件名，0项运行，日志保留；改为实际存在的 `test_tracking_pinch_contact_replay.py` 后通过。

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_projected_pinch_contact_audit.py tests/test_tracking_pinch_contact_replay.py
```

分析结果SHA：`1c64155ff79e6c327dec6dfb95150a78659597c590b53772c981d04fcaf65c65`。
私有目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/projected-pinch-20261114/` 保留命令、失败日志、结果和恢复点；不上传原始账本、模型或Checkpoint。

## 下一批及独立阻断

1. 按本次左finger23实际link、关节、杆节点和接触点分解目标/FK/跟踪误差，再做保留所有保护的候选筛选，不扩大patch。
2. Pinch仍缺左食指与双侧承托。Tripod仍有全几何路径约束冲突，Power仍缺合格闭合；独立推进，不复制本姿态。
3. Teacher继续补柔性稳定性来源、R0训练系数/Reset与奖励资格；无相机Teacher不依赖WireSeg。不得把本短诊断用于成功BC池。
4. 本轮WireSeg新增训练步0；此前10/148完整官方分片、360精确配对状态不晋级，持出误差问题仍在。
5. 本次原生预算已消耗，新实验须新的冻结代码/配置与本聊天持续授权范围绑定；不自动启动长PPO或扩展5s完整成功预算。
