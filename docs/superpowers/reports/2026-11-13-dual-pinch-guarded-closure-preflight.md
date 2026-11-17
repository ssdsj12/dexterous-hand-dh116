# 双手 Pinch 一次手指动作区间：原生前完整链路核验

本包基于已发布双手 v4 的真实 2+8 零动作账本，只准备独立 2+8+8 物理步骤配方。当前包原生执行0步、PPO更新0、Teacher/抓取资格均未取得。生命周期测试使用 SDK-shaped CPU doubles，不计入原生接触次数。冻结原父receipt、实际reset、自由41节点杆、原关节/Mimic/力/2.1mm台面保护均保留。旧环境、资产、模型和Checkpoint未改。

## 实际实现

- prepare_dh116_dual_pinch_closure_interval.py：复用原App/Reset/SDK/rod owner，以ClosureExchange显式扩展原ClearanceGuardedExchange。先完成原SDK/rod子步，再同钟保存实际17DOF、23-link位姿/速度、source17+locked5几何及真实杆节点。
- 第一个60Hz区间动作全零；第二个区间只有原action indices 1/2/7/8为0.1/0.1/0.05/0.1，其余20维为零。每区间八个480Hz物理子步，不能重复、延长或自动Lift。
- geometry/table旁路读取异常转成独立error字符串、缺失packet和终止原因。原完成物理样本先写入physical.jsonl，再终止后继动作。原joint/Mimic/30N守卫优先；缺失不伪装通过。文件I/O失败、强杀等不能保证完整落盘，外层必须失败。
- sdk_clock.jsonl每行来自实际SDK step、原sim.current_time、同钟杆及post-action身份。保存policy_step、substep、timestamp_s、sdk_time_s，逐行flush/fsync，包括失败前缀。没有在成功后按range伪造完整时钟。
- inspect_interval无SDK重放原序列/动作/净空守卫，逐包重新计算真实几何；原guard_failure通过独立字段恢复后核验。geometry/table错误行、NA行和终止原因分别计数。
- worker只接受专属GPU3/GDB标记、精确父回执与profile SHA。旧checkpoint、全局控制器、旧worker入口不变。
- validate_dh116_dual_pinch_closure_interval.py与独立lifetime helper保留原STOP、非Sim弱引用释放、SimulationContext清理、App close返回、外层退出码和GDB全栈/映射门槛。Simulation实际存活阶段据实记录。
- 原静态GPU hand-table contact仍不可读取，保持NA。几何正净空不冒充原生零接触；没有手杆接触、抓取、原生R0或Teacher晋级。

## 本次修复的真实软件失败

1. execute绑定旧run_existing_app后，新wrapper又经相同可变模块属性调用，CPU完整路由实际复现RecursionError。保存原函数引用后修复，并新增execute级回归。
2. 原旁路table reader在journal.record前抛错会丢掉已完成SDK行。现在两个附加读取均在新exchange完整子步后捕获，保留原物理行，以守卫结果先写后停。
3. 原退出审计只比对SDK两字段。现在全部时钟字段逐行相等，时间/策略步/substep/bool-counter篡改均被拒。
4. 配方构造曾将受保护资产列入可发布源码，调度器正确拒绝。改为原绝对路径只读inputs/SHA。末次测试修正了资产已包含在父external sources时的重复计数假设，生产门槛未放宽。失败日志保留，未启动失败配方。

## 最终CPU证据

以下使用最终production preparation SHA 686a5bcc56ffdcb9cceb1eb24ba41d0f2c686c4f574fbc6615023f18ff77c03e；几何helper SHA 3e90dca3dfc2448ad2a1e82f545c29e7229fbda7510b90fba45bf88a5a7132de。

| 范围 | 实际结果 | 日志SHA256 |
|---|---|---|
| 完整动作链、真实几何helper、SDK11 geometry/table失败、旧guard优先 | 23 passed，221.88s | 058ad047a7f70dfab102a57930402cb767e496269a5b044e4ed4232dde6c7155 |
| 最终独立退出审计与全部时钟篡改负例 | 26 passed，199.32s | 718944c67e3505c4104d90493589d2ee1cbdfb658bb426881fe45aa4b4206208 |
| pre-App导入阻断、实际CLI、冻结源/配方/父依赖/预算 | 52 passed，13.76s | 4db254ebb2e63fa751f4e6093df2408b315f25f5255166c967a599320220124e |

合计101项不同测试；前置几何包95项已单独发布，不重复计入本包。CPU路径实际执行原builder/episode/SDK doubles，几何校验使用真实MeasuredClosureGeometry（原USD与锁定SDF），未以永远passed的几何stub绕过。成功路径16条测量；SDK11故障前缀9条物理行与9条SDK行、两次end、终止后无后续步。

统一复现命令：

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_pinch_closure_interval.py tests/test_dh116_dual_pinch_closure_postexit.py tests/test_dh116_dual_pinch_closure_worker.py

原件位于私有evidence的closure-observation-cpu-64ksg2g5、closure-final-postexit-DKaFZQ、closure-clock-4pRU6c。未上传原始实验数据。集成工作区发布前会再核对相同代码和回归，结果写入Registry/readiness。

## 冻结范围和后继执行

- 配方SHA：8c377e2e7aae5539597f9df08ee38a9a0ac7319f7bef98affa25976a02b8c200。
- 调度code/config SHA：74f8378ee4648f30c8ecb2ba8f0aa5f2caf9ef396a8e49b9c532cbf44ff7fad6。
- worker内部源指纹：4e6dd4284c433253cddbab4c1ed610501dc13155f6e2e0d2756d50458389b463。
- profile SHA：67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498。
- 父回执SHA：aa1a6877c29b460e8f12589674d4ba09381faf8aea200400a7cd0e71d1a5a1ec。
- GPU3、单实例、1环境、2初始化+16测量=18总物理步、0PPO、180秒、精确指纹最多一次。调度依赖真实v4 scoped receipt；恢复不得重复父实例。
- 发布后依据用户“批准，后续自动批准”的既有小规模范围绑定本新指纹，重新检查资源锁，只执行一次。不复用已耗尽的旧配方、不自动加步。完整接触/Lift/Teacher资格不由本动作回读获得。

下一步取得新动作的真实追踪、台面净空、手杆pair、双侧wrench与自然退出账本。失败只阻断该候选后继，保留前缀和GDB。成功也只证明这一短区间执行，仍需真实对向接触、连续材料滑移、R0有效测量/独立系数资格和三种抓法各自的训练准入。
