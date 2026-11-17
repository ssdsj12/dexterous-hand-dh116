# 同一 episode 双手 Pinch：实际接触与自然退出

执行HEAD：0b0f79e1adc831468725205c09f261e3bdcc2d80。一次新配方，234/234 SDK步（2初始化+232测量）、29控制窗口、2784 Solver子步、1环境、0PPO，GPU3单实例。实际原生过程166.289762秒，外层exit0、GDB inferior exit0、signal=null；app.close返回。没有重跑旧预算，没有强制退出。

## 独立身份与原始来源
- scope：bilateral_owned_pinch_2_setup_232_guarded_contact_diagnostic_only。
- recipe SHA：78f1a9f79eb5f0a795396e4284c85d95ba068f0dab8f7b59670616c6ff5df7f2。
- scheduler/source SHA：cc57fbb782979afea2148c066aef33020926ed8385ff229f064f4bb7e1441d2e。
- worker SHA：4ee7dcf283f3e6ed9cb60cd6c5e27fd51665f41a92ca8aa75f414d42d499ee02。
- 原生receipt SHA：8ff5b25afb5a46a1cb57899b5aedf42c1e0e766a5bf30d4628e229bd62850db8。
- physical.jsonl SHA：d53b446425640fbbbf59ba952c52ec84ff952f0eca8b539131f9197bc899158d。
- 独立postexit SHA：c361d17d2c49efb0675fcfa51dbe2305cca3683ff949aa56cba2ad7017747492。
- env_id=0，reset_id=1，rod_uid=rod0；episode_uid=dual-pinch-76ef2a55c15642c8a40a405096485601:one。
原始输出：/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-bilateral-owned-20261114-xizgjb7a/native-runs/flex.f8-bilateral-owned-pinch-native/attempt-w0i1tg3o/native。
此批已耗尽，不允许重跑同配方。scheduler父进程180秒等待返回null发生于原生已自然退出之后的CPU复核期间；真实外层耗时166.29秒，不把CPU后处理误记为额外物理步，也不因此启动第二次。

## 同一时刻接触证据
| 原生量 | 实际结果 |
|---|---:|
| 左拇指 / 食指 pair记录 | 526 / 495 |
| 右拇指 / 食指 pair记录 | 530 / 560 |
| 左必需两指连续有载荷 | 85.9375ms |
| 右必需两指连续有载荷 | 92.013889ms |
| 同一Solver子步四个必需指腹均有载荷 | 495子步，85.9375ms |
| 对向法向符号候选（左 / 右） | 85.9375 / 91.666667ms |
| 桌面有载荷 | 全部2784子步；109870条pair |
| 有效力闭合 / 稳定性 / 材料锚滑移 | 尚未资格化，NA |
| 抓取成功 / Teacher成功 | 0 / 0 |

双侧同载荷首次发生于SDK193、policy23/substep6、solver_substep9、solver_tick2290，重建子步末时刻0.401736111s；持续至SDK234。不能将法向符号候选等同力闭合，不能将本次0.08594秒称为5秒Hold。
杆median Z相对实际Reset为-0.000089407mm，最低节点相对Reset最小-0.001758337mm；没有形成净抬升。这里只报告同钟实测节点统计，成功仍要求30mm/5s/0.5mm与无外部支撑。

## 姿态、碰撞和跟踪
初态复用已独立验证的左姿态与右自身姿态，不是左右镜像；与上次右单手实验相比，初始Reset不变，新增原左闭合动作并发执行。右根位移请求仍为-0.5mm、其他腕平移/旋转不变。原物理、材料、速度、关节、Mimic和深度保护均不改。
完整232步没有guard failure。全部source17+cooked5最小桌面净空：左2.117664mm、右2.205674mm。整手分离下界最小318.166895mm。

四个关注指腹全程桌面穿透均为0；各自最小净空如下（source / cooked，mm）：
| 指腹 | 左手 | 右手 |
|---|---:|---:|
| finger14 | 2.992700 / 3.086673 | 3.274423 / 3.319090 |
| finger23 | 2.833199 / 3.172788 | 2.671451 / 2.925908 |
| finger43 | 55.876630 / 56.028752 | 47.524810 / 47.559858 |
| finger53 | 73.363064 / 73.508966 | 61.358938 / 61.611527 |

非目标碰撞形状至杆的保守下界最小：左0.00005833mm、右0.00345352mm，余量极小，不能据此放行后续更大运动。
实际base_link相对SDK2 Reset的最低Z位移：左-0.000625849mm，右-0.498205423mm。这是实测base_link，不冒充另一个palm link或命令位置。
主动关节目标-实际最大绝对误差：左0.003324151rad（finger21）、右0.005631417rad（finger11），均在同一episode SDK210、policy25/substep7、timestamp0.4375s。原始目标及实测值保留physics.jsonl。
右finger12实测范围0.008912619～0.010057464rad，无越限；被动耦合残差最大左9.466836e-6rad、右3.730163e-6rad。均为仿真实测/估算，不是真实电机测量。

## 生命周期所有权
释放后左右Articulation对象、articulation view、permanent/instantaneous composer、runtime、rope、episode和reader均无存活引用。
共享asset SimulationView在释放手对象后仍由SimulationManager持有，STOP后消失。
SimulationContext仍有__init__ frame引用，直到after_app_close快照仍alive；native_release_verified保持false，不能据此声称所有C++对象已在close前销毁。
本次真实自然退出0且无SIGSEGV；保留gdb_finalize.bt.txt及libraries/mappings，未产生崩溃栈。此结论仅覆盖本次有界配置，不替代旧C5相机失败，也未用gc或强制退出制造通过。

## 新增只读诊断与测试
audit_dh116_bilateral_native_evidence.run绑定唯一实际receipt路径+SHA、双手scope、recipe和源码SHA；逐artifact核验。validate_receipt_scope拒绝超预算、非有限时间、单手scope和失败回执；require_episode禁止跨episode混合。
reset_base_position使用原SDK2 getter实际body-link位姿；按同一env/episode/physics/policy时间保存掌根峰值和关节跟踪峰值，未使用启动期名义姿态。
先4项模块缺失RED；补充非有限duration及自声明fixture门控RED（4失败5通过）；修复后新旧诊断16 passed。发布追加既有contact-source基线回归。
只读分析新物理步0、训练0，run02耗时7.906308秒。
输出 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-native-evidence-20261114-run02/result.json，SHA52e99c16914a1ce95849b93a1a8e05a9be06e2cae4f68ebd5cd1fc3e53e22066；run01保留未覆盖。
命令：bash scripts/python.sh -m scripts.audit_dh116_bilateral_native_evidence --output <新目录> --receipt <本次实际receipt.json> --receipt-sha256 8ff5b25afb5a46a1cb57899b5aedf42c1e0e766a5bf30d4628e229bd62850db8。

## 下一工作包
保留本次原生成功执行回执，但Teacher准入仍false。继续同钟有效对向接触/稳定性、固定材料锚滑移、R0训练系数和371D字段来源，不填NA为零。
Tripod/Power分别做独立碰撞和可达性预检，不复制Pinch成功标志。WireSeg、camera独立推进；本包WireSeg训练步0，历史诊断128seg+128depth不改。正式PPO、BC、DAgger均未启动。
新原生接触配方仍需新的CPU证据、冻结指纹和资源检查，按已有持续批准范围执行一次；更长Hold/训练预算不由本234步结果自动扩展。PR2保持Draft。
