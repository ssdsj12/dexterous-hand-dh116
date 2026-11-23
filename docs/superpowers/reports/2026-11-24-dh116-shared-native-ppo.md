# Shared native Teacher PPO: actual two-lane updates

基线：346730e53ea1eaabc2773f763ddbe779d9cbdeb6。Ubuntu独立执行树，GPU3串行；
GPU0/1为其他用户占用，未启动三张GPU并行。PR #2仍为Draft。

## 实际运行与范围
Pinch原生两环境、64控制窗、522 SDK步、2次accepted PPO update，natural exit0。
两次迭代26.2722s / 25.7917s；优化0.1496s / 0.0454s；Checkpoint往返核验均通过。
Power同预算完成2次更新，natural exit0；18.8775s / 18.5790s。
全部运行均没有强制退出。峰值torch allocated仅约26–29MB，**不是全进程/全GPU峰值显存**。

这是approach_wiring_smoke_v1 / finger-goal数据链路诊断，不是完整抓取训练完成。
Pinch/Power本次hand reaction非零步均为0，不宣称对向接触、离架、Lift/Hold/Slip通过。
128/512环境实际Isaac PPO尚未验证。所有新Checkpoint独立保存、禁止进入成功专家池。

## 实际修改
- train_dh116_shared_native_teacher.NativeSharedPPO：真实pre-action采样→原Bimanual24D执行→每8SDK步原TrainingReward exactly-once结算→原record_group_window/GAE/KL Guard→CUDA优化。
- 窗末bootstrap先于局部Reset；延续原shared反力所有权，清空本路奖励/上下文后下一窗继续；保留last2 SDK内存黑匣子。
- 正常训练不写逐SDK全量JSONL/fsync，只写update/Checkpoint及失败证据。旧完整审计路径保留。
- SharedParameterRadius：完整N环境原生parameter_snapshot来源，不冒充N=1；原几何数学、阈值和默认旧reader保持。
- make_live_reader：十原始inspected pad＋完整contact_bindings；Tripod/Power新增四体不丢失，46个原始碰撞形状仍检查。
- 原始pair capacity/attempted/substeps等元数据透传，17-slot检查不造字段。
- 独立height seed：Pinch双腕+5mm，原q/qd、杆、资产、物理、关节范围及保护线不变；失败SHA和新seed绑定。
- 原生reset_env_mesh_poses在初始零时间提交current/target/solver三态，并检查零motion；距离基准从实际新Reset重算。
- Shared checkpoint使用原Adam/RNG校验，包含CUDA RNG、recipe/模型/backend/N身份、独立SHA与exclusive写入；恢复要求新的已核验物理Reset。
- verify_roundtrip逐项比较policy/Adam tensor树，保留Adam整数参数键；不使用只接受字符串键的账本JSON hash。
- 修复三个已有Registry条目的非法executor=offline为prepared，状态/证据范围不变；Registry回归由4失败恢复至60通过。

## 失败证据保留
Pinch原初态在SDK61净空2.0898608mm，小于2.1mm保护线；未降低保护线/噪声。
Tripod首次14 contact误进10 pad入口；修复后真实15窗，在SDK132左finger33源几何报contact envelope collision。
离线原数据重放：原间隙lower=-0.230255mm，subset upper=-0.162355mm，原允许穿入0.15mm；2mm平移假设通过原几何检查，仅为候选依据。
首次新初态提交代码误取collision_clearance tuple，随后CLI未转交height参数，两项均修复并新增实际CLI转发回归。
第三次Pinch在SDK266已完成一次优化/写Checkpoint，但核验错误地对Adam整数键使用JSON hash；已修复，最新两次更新完整通过。
所有失败的原始日志、黑匣子、Checkpoint留在Ubuntu证据目录，未上传GitHub。

## 回归与命令
- 最终发布联合回归：104 passed / 2 CUDA skipped，36.55s，shared-ppo-release-cpu.log；涵盖Registry与本包源码。
- shared-ppo-integration-cpu.log：44 passed / 2 CUDA skipped，20.27s；与其他测试重叠，不相加。
- shared-ppo-preflight-cpu.log：原奖励/geometry/rollout联合54 passed / 1 skipped，651.89s。
- shared-checkpoint-adam-green.log：CPU＋真实CUDA checkpoint/RNG/Adam往返2 passed，6.06s；人为损坏Adam exp_avg必须拒绝。
- shared-ppo-registry-fixed.log：60 passed；先前4 fail / 56 pass日志保留。
- python -m scripts.train_dh116_shared_native_teacher --grasp <grasp> --receipt <actual-reset-outer> --receipt-sha256 <sha> --backend <source-locked-so> --backend-sha256 19de772f73beb082a1f20e9b2337d2d593bfc6021121db18e4a3e15dfc0ff6bb --gpu 3 --num-envs 2 --updates 2 --output <exclusive-directory>
Pinch另加其原始smoke来源和独立height-candidate的路径/SHA。完整argv和冻结源码SHA在每轮attempt.json。
原保护528文件全部SHA不变；两个历史/tmp Checkpoint运行前后均不存在，不能声称已核验其内容。

## Antigravity与下一包
通过指定desktop-review角色讨论R6/R7；它的MCP仍固定旧HEAD125f2664，意见仅基于当前摘录，不能充当本树实读证据。
采用：保留mount/target/seed同步、新Reset重算距离、静态table不随手升高、独立身份、128短测及分项计时。
未采用：将窗中硬守卫改为负样本准入；保留用户要求的停机/拒收。
其“90%耗在几何”属于未证实推测；实际Pinch guard_and_r0=9.49s/26.03s，尚有其他阶段。
继续Tripod独立候选；随后128环境短测，补齐真实总显存/分项耗时，再启动各抓法128 PPO。
R2、正式成功BC/DAgger和Student资格继续关闭，成功标准30mm/5s/0.5mm不变。


## Tripod本轮最终回执
独立+2mm候选在SDK190、22个完整窗后再次触发左finger33原contact envelope检查：
lower=-0.180977mm，subset upper=-0.151031mm，对比不变的0.15mm允许穿入；
所有来源完整，非Source缺失。0次update，natural exit1，159.774s，无强制结束。
冻结文件核对不变。只阻断Tripod自身后继，Pinch/Power已验证结果保持。
下一候选需要增加初始可探索间隙，保持原关节、接触/桌面保护线和探索std不变；
不能把无接触更新解释成三指抓握成功。
