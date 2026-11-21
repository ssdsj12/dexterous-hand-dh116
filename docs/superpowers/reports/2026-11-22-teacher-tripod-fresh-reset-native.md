# 三指 fresh17 Reset：真实原生通过

## 实际执行
源码/Windows/GitHub一致提交 b925cc1012428ea94f1052c2a0ad157e2b93a17c。
冻结指纹 71a60cdc5a90873f80afa571a8a528261401abe17dbc5f8c53568ed22afaf2dc。
GPU2、单Isaac、1环境，scripts/python.sh启动；GDB外层自然退出0，105.711470s<180s软预算。
实际仅2个SDK初始化步、1次零步episode Reset，控制0、求解器0、Teacher查询0、PPO更新0。
没有SIGSEGV、SIGTERM、SIGKILL或os._exit。单实例锁已重新取得并释放，GPU2恢复9MiB。

## 真实接口与独立复核
- 在首次Reset前已直接建立17槽，14个左右独立DH116 link来源；容量8364只分配一次。
- current/target/solver三套17槽GPU回读一致，pending/moving/pair-ready均关闭，原生容量8364保留。
- Reset后14个手碰撞槽位逐项对应本轮实际23-link SDK位姿；table和两个support保持固定原输入。
- 数值Reset审计2份完整样本逐字节相等；杆位置/速度/四元数和真实参数不被mesh重建改变。
- common状态由实际source_packets重建：原371槽中66项有效，305项保持NaN和false validity。
- 34个原始形状与10个cooked包络保留，左右手最低桌面间隙57.382352/46.537862mm；
  44个包络对支架最小间隙5.655938mm。此为初始无穿透几何，不是抓取力或稳定保持。
- 全部native文件SHA、GDB日志和冻结文件逐项重读；真实生产validate_supported_readback、
  validate_numeric_reset_audit、validate_mounted_observation和guard_hand_supports独立复核通过。
- GPU枚举阶段记录了CUDA bad state/跳过GPU警告；随后表格显示GPU2 active并完成任务、自然退出0。
  这不代表全部图形/CUDA路径稳定；未因此更改Driver/SDK或重启。
- 首次入口在App创建前因USD导入被拒绝，自然退出1；对应失败回执保留，没有抹去或重用旧指纹。

## 测试与证据
本轮源码修复的7项选定CPU回归已通过（42.66s），另有此前构造/回归74项与Registry60项记录，范围分开。
原生退出后真实数据独立审计1项通过，非新的原生实验。
新增源码 scripts/audit_dh116_tripod_fresh_reset.py::audit_run，返回实际Reset组件证据，绝不授予抓取资格。
对应3项CPU回归通过（13.07s），使用真实native数据；强制退出或把exit code改成布尔False再重算SHA均拒绝。
TDD保留缺入口与布尔退出码红灯日志；生产拒绝使用显式ValueError，python -O不会绕过。
仅Registry/Release这两份可变发布账本核对冻结commit的精确Git blob；
所有物理代码、配置、资产、编译来源仍逐项严格重读当前SHA，不能用历史Git字节绕过物理来源变化。

复核命令（输出必须使用新的Git外路径，失败不覆盖旧证据）：
python -B -m scripts.audit_dh116_tripod_fresh_reset --run <ref902/fresh-supported-tripod-reset-native-v2>
--receipt <同目录/outer.json>
--receipt-sha256 421e698dd5ba6604dd19a43af14f00875a974bc086c92b575e12d6ef41e5d466
--output <新的Git外审计结果.json>
raw JSON（含约199MB的mesh/SDF来源展开）、log、maps均留服务器Git外，只发布源码与摘要/SHA。

## 下一步与资格边界
三指冷启动/Reset已得到真实证据，可供独立Tripod manifest与NativeSources消费；
旧Pinch-only static/live gate尚未修改，不能只改grasp字符串放行。
接下来接通独立三指六指腹距离、七路线源、任务context和R0到原PPO collector，
再做有界1次更新与恢复。完整接触、固定腕保持、离架Lift/Hold/Slip仍未达标。
Pinch已有历史短PPO更新/恢复证据，不等于完整抓取；Power仍需五指与掌面全握的独立来源与真实力资格。
本次未加载策略，不是Teacher训练或成功演示。R2/ForceProxy、正式BC/DAgger保持原门控。
相机、24D/371D原合同、USD、材料/物理、关节保护及30mm/5s/0.5mm标准均未修改。
Antigravity当前额度不足，未取得新外部审查，不将CPU/原生Reset证据冒充对方确认。

最终Registry：60通过、0失败，13.50s。528个保护文件和3个原best.pt保持原SHA。
