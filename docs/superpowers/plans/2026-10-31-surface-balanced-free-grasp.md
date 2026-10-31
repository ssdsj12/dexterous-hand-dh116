# 实施顺序

对应已确认设计：`docs/superpowers/specs/2026-10-31-surface-balanced-free-grasp-design.md`。继续当前10-30工作目录，保留现有未提交修改，不新建用户任务、不删除运行数据。

1. 独立任务A：`tabletop_end_supported_scene.py`、`tabletop_gpu_isaac.py`及专属测试。纯函数先验证端点支撑与60 mm高度变换，随后接入native GPU scene及reset；默认0保持旧场景。不得改训练CLI、TensorTask和分面模块。
2. 独立任务B：`tabletop_effective_pad.py`、`tabletop_effective_pad_runtime.py`、`tabletop_gravity_feasibility.py`及专属测试。先观察背面/非指腹、超力矩/缺托举方向测试红，再实现源CAD指腹区、原生接触采样与静态六维wrench筛选。不得改共有训练/世界文件，不启动Isaac。
3. 主线程：先写`tests/test_surface_force_sharing.py`验证F/2/k、不固定指名、不固定F、单面不合格，运行看红；实现`tabletop_surface_force_sharing.py`，再测试绿。
4. 主线程集成24维新模式：训练CLI显式开关，`tabletop_free_wrist_grasp.py`、`tabletop_tensor_training.py`、`tabletop_tensor_world.py`和`tabletop_tensor_task.py`。保持旧26维入口；新合同拒绝静默读取旧权重。原始有效指腹特征、面间/面内误差进入观测和TensorBoard，不保留旧固定力带及拇指食指专用均衡惩罚。无lift启动门。
5. 新模式全部高度使用初始支撑面，加入架高零动作不成功的回归；在明确的物理限幅内加入零总目标增量的面分配反馈。
6. 源码审查和定向回归，冻结运行源与配置后部署；全程仅一个Isaac。执行架高场景短物理诊断与原始记录，再静态wrench复算。接口可靠后仅短PPO冒烟，成功/失败如实分开。更新README、CHANGELOG、实际报告，下载证据并核验SHA；本轮无自动GitHub推送。

服务器测试统一使用`DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q <本任务测试>`，物理场景使用480 Hz、原PGS64/16、mimic400、0.2 Nm与0.3 rad/s上限，质量、材料和重力不修改。新场景高度和支撑明确作为场景变化记录。

## 成功示范后的PPO就绪验证

沿用用户已确认的连续自由探索、分指误差软目标、真实lift成功标准。AK示范不是策略权重，缺少策略观测，不静默用于actor初始化。

1. 合并原生每控制周期均值配对、逐子步净高/深度/速度有效性和5 s三点稳定窗口，部分reset不污染其它环境；先纯CPU反例，再重放AK真实记录。
2. 训练端均力只保留软奖励；策略零动作能保留关节指令，物理mimic继续执行，默认不叠加自动均力/主目标回退。surface回合30 s覆盖成功示范21.33 s，旧模式仍15 s。
3. 接入可选原生批量读回，保持原物理参数；以已成功静态预抓姿作为reset seed，仍从原支架高度开始，无空中reset/no object forces。
4. 单Isaac完成16环境、16步rollout、2次PPO更新：检查有限采样/优化/权重与优化器保存/TensorBoard事件。该短测不用于声称策略已会lift，不自动沿用历史2048轮预算启动长训。
5. 保存短测证据与新源码冻结、更新就绪报告，明确教师控制成功与PPO策略成功的区别。
