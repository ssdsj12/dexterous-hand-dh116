# Tripod零动作原生对照：沉降取证与自然退出

实际执行HEAD：`1c3668340457c8a25de2125c0248ba9cacdfaa6e`，源码提交1254014210882ce0d090fdc33ab64fca77411524。此前150项最终CPU回归和Windows同commit发布已通过。沿现有“完整访问、有限独立诊断”持续授权，仅执行一次新冻结配方，不复用旧67步预算。GPU2开始前仅9MiB/0%使用，原单实例锁可用；单Isaac、1环境，无Actor、模型、Teacher查询、PPO或双目采集。

## 实际预算与独立验证

2setup＋1独立prime＋240零指令测量，共243SDK、241solver calls、2892内部子步；30个原8子步control窗口，policy_windows=0。实际242.498320s，内层360s软截止、外层480s，未超限、未重试、未强制结束。冻结SHA6c8b0a88943a9ca7635e0e5b4e2c2bc127861ead60380daefb07b7ff0aa9e418，外层scope SHA84e72056c6be8ef95eb047b052eaa120c88073f7f3c025a575077e118adfaa81。

独立CPU读回核验28个原生artifact与2073个冻结文件，重算全部241行原始clearance guard、全部241行几何packet、原contact reducer和连续SDK/rod/action时钟；所有实际issued request为有限float零24D。raw与enriched两份各241行，返回求解行、持久化行和来源验证行均241。SDK3起、243止，rod_tick12到2892，没有伪policy token、插帧或隐藏步。详细源SHA与不可变回执见readiness。

冻结时先拒绝了/home别名路径，改用resolve后的真实/mnt路径后通过，没有放宽来源校验，也尚未启动Isaac。另保存了分析v1的限制证据：其seed比较曾对缺失旧字段使用fallback，不能作身份依据；最终v2改为核验旧core固定SHA→c123 manifest→本次实际seed，并重新完整重算241行。原始数据未修改，未追加原生。

## 新证据：杆在零动作时也持续下沉

零动作SDK3到243，平均Z从364.983567降至354.384604mm，下沉10.598963mm；最终最大节点速度12.626074mm/s。末48行约0.1s内平均Z仍变化0.891880mm，最大节点速度18.446211mm/s，所以不能把0.5s末端宣布为静平衡，也不生成“合格静置task_start”。

与此前Contact8的同SDK4～67区间作逐材料节点对照：两段全部节点坐标最大差0m，平均高度变化均-4.671093894mm。两段SDK3 prime的41节点、双手q、qd、backend targets最大差都为0；旧core reset_sha256同为c123，当前实际Resetseed SHA也匹配。因此这个受控初态中，短窗口杆下沉在无策略动作时同样出现；不将其泛化为所有初态、所有训练失败的唯一原因。

本次零动作最后左三pad gap分别4.025/4.390/3.330mm，右三pad为7.372/9.594/11.019mm；平均距离从5.805214增至6.621500mm。手实际target保持，杆与指腹相对几何仍改变，说明不能把“零动作”解释为线缆/相对状态冻结。

## 接触和关闭仍严格分层

241/241行左右手法向载荷均0N，bilateral、support-free和table接触行均0。两支架峰值0.157355/0.157580N，末端两支架约各0.150664N。没有建立抓持，没有Lift/Hold成功，不授予Teacher或ForceProxy资格。

App.close正常返回，GDB inferior和外层均exit0、无signal；14个被追踪weakref均已死，GPU2回到9MiB/0%，GDB和driver PID已不存在，单实例锁重新获得并释放。R06内部general native_release标志仍false，结果只证明这一次诊断的自然退出，不宣称所有生命周期问题已根除。

## 回归与后继

原生后91项CPU回归通过（60.02s），覆盖真实入口冻结/clean-App边界、原contact账本和Registry。这些包含此前测试，不与150项累加。本结果包没有再次启动仿真或训练；原始模型、日志、JSONL、maps均留服务器，不上传。

下一项审查接近阶段distance potential在d>=固定Reset参考时的clamp，以及如何在被动沉降中保留有效学习信号。不能把距离势能导数0解释为PPO参数梯度全0，也不能盲目扩噪声、改速度、移动旧dref或取消安全guard。独立signed potential、独立真实静置task_start都只能经证据和设计核验后选用；现有R0、24D/371D、双RGB、资产/物理和30mm Lift/5s Hold/0.5mm Slip保持。Pinch和Power仍有自己的接触来源和训练门槛；Power必须掌面加五指真实包络。正式BC/DAgger/Student继续blocked，R2关闭。

登记结果后Registry另60项通过（14.70s，与91项重叠，不累加）。报告和readiness记录的是本次执行的归档HEAD，后续发布元数据变更不改写原生frozen_plan和原始JSONL。
