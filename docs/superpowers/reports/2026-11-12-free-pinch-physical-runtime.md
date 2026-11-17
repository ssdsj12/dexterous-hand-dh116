# 自由连续杆单手 Pinch 原生候选实现

本包新增实际 DeformX/Isaac 运行绑定和有界 CLI，未启动 Isaac、CUDA 求解或训练。P0 双拇指校准及冻结代码/配方保持不变。本候选只准备下一次单手接触试验；grasp_qualified、full_state371_bound、teacher_admission 均为 false，原 flexible task 的未知字段不变。

## 实际物理入口

tabletop_deformable_grasp_runtime.py 直接调用锁定扩展 RopeBatch.from_positions，显式传入全 False pinned mask 和 fix_root_orientation=False，仅使用 step_free。直接读取 pose_reference.json 的左手初态、41 个材质节点、0.5 m 连续中心线、桌面中心 [0.6,0,0.25] 和尺寸 [0.8,1.2,0.1]，不引入 root driving、杆外力伺服、附件或两端固定。

独立候选继承 P0 的 rho=0.065 kg/m、EA=800 N、EI=0.018 N·m²、radius=0.005 m、12 子步/物理步和8求解迭代；质量按相邻静止段长度各分一半读回，预期总质量0.0325 kg，两个端点质量均大于零。重力采用 [0,0,-9.81]，新增有限尺寸 box SDF 桌面。不能继承零重力固定端校准的资格。

五个原始 distal pad 各使用冻结 SDF 与 cooked hull/source SHA，通过实际 PhysX link_paths、body index、原始几何 SHA 映射到五个独立 mesh_wrenches 槽。桌面使用第六槽单独记账，不向 hand 施加桌面力。手上的上一物理步反作用力经真实 link→COM 转换后写回各自 SDK body，实际 composer 读回与来源步逐行校验。

保留原 USD、mimic、active joint 限位、0.3 rad/s速度限制与0.15 Nm驱动力上限；SDK getter读取实际stiffness/damping/limits/max_forces/max_velocities。原 grip_compliant 材质被绑定并通过 USD 解析验证；DeformX 接触使用自身 kappa 接触律，不能把 PhysX pad compliance 说成已经映射。

## 真实源码证据及材料歧义

后端目录 /mnt/mydisk/xk/projects/deformx-f0-04f6a11/backend，commit 96b8514c7c55c724f1abba1fbf6ed6642b7db72b。CPU preflight 按已有backend lock逐文件校验真实源码SHA，并核对已构建扩展、五个pad SDF和原USD。

| 实际文件/位置 | API事实 |
| --- | --- |
| csrc/bindings.cpp:889–934 | from_positions 默认固定root；本候选显式全False mask |
| csrc/bindings.cpp:951–959 | step_free，无root轨迹程序 |
| csrc/bindings.cpp:860–888 | drag/damping/num_itr getter/setter |
| csrc/bindings.cpp:1097–1103 | RopeBatch只有gravity setter，没有getter |
| csrc/bindings.cpp:1109–1120 | inv_masses通过真实solver download读取 |
| csrc/bindings.cpp:973–1008 | 接触材料和逐SDF mesh注册，有效配对摩擦 |
| csrc/bindings.cpp:1058–1096 | 节点力是所有表面合力，mesh wrench才有逐mesh身份 |
| third_party/yarnball/KittenEngine/YarnBall/YarnBall.cu:112–174 | rho单位kg/m，端点按半段质量，非kg/m³ |
| YarnBall.cu:237–258 | 独立EA/EI作用于stretch/frame bending；无独立G参数 |
| ropesim/sdf.py | 网格外无接触；本候选有限桌面margin=25mm、spacing=5mm |
| 本机IsaacLab articulation.py:1610–1628 | SDK实际关节属性getter用法 |

P0参数不是测量材料。按5mm圆截面倒算，EA/A约10.19MPa，EI/I约36.67MPa，不能冒称同一个Young modulus，也不能由它们确定shear modulus。后续来源选项是：批准明确标注的P0系数候选；另提供可追溯实测rho/EA/EI/摩擦；或明确选定一致截面、E、nu形成新配方。既有刚性杆资料不能补出弹性参数。重力/接触/EA/EI明确是setter配置，不伪造SDK getter读回。

## 身份、时钟、状态与原始R0

每次运行生成UUID episode。reset_envs复位后核对节点和零速度；solver全局时钟不清零，记录独立solver_time_origin。每行绑定env_id、episode_uid、reset_id、rod_uid、真实SDK physics_step/timestamp、policy_step/substep和rod_tick。初始化2次SDK tick，加238个测量tick，总计240；重力沉降包含在238步内，没有额外warmup。

read_rod_snapshot使用真实solver数组适配既有DeformableRodStateProvider，包括节点、速度、合力和几何派生切线/曲率/应变；CLI实际调用provider验证。完整371D policy观测仍未绑定。动作只启用左手finger12/finger21的规范24D通道，每步核对SDK target。R0保留required-contact fraction和table contact原始读数；净wrench点积只叫wrench_opposition_candidate。有效对向法线、材质锚滑移和reward权重继续不可用。

## 生命周期与独立验收

run_owned_call在成功/异常路径都先reset composer并同步，再清除已结束异常frame及原traceback，按provider→runtime→view→composer→hand释放。run_app完全离开异常块后执行STOP/App close，关闭后才抛出不持有原native frame的字符串异常。CPU故障注入覆盖step/provider/validator及实际run_app边界SimulationContext/closure弱引用，确认释放在关闭之前。

validate_pinch_artifacts(output, outer_exit_path, expected_code_config_sha256)从磁盘重读ledger/plan/result/ownership/shutdown，验证238行、多个pad与桌面分账、前步实际反作用力、时钟、自由端、GDB自然外部exit0及native.log SHA。CLI的--validate-output与--outer-exit可独立CPU调用。

配方冻结入口、验证器、运行时和专属 scheduler wrapper 的 SHA。run_dh116_pinch_work_packages.py 的 PinchRepository 在旧 _native_child 成功后、receipt写入前，先将 native_qualification 撤回，再执行独立磁盘验收；任何异常记录 failed_native 与具体reason。仅验收成功并将 pinch-post-exit-audit.json 写入attempt、绑定audit SHA和三份源码SHA后才恢复候选范围资格。后续 _verified_native_receipt 会重新读取并比较audit及真实物理证据，缺失或改变均拒绝。

专属wrapper的main临时将scheduler.__file__和Repository指向自身/PinchRepository，并在finally恢复。旧start_native会真正spawn该wrapper；子进程CLI重新安装相同的审计门槛，项目root保持不变。真实CPU子进程测试保留start_native、worker CLI、registered-worker检查及receipt写入，仅替换私有夹具内的native body与资源/仓库环境，覆盖审计通过和拒绝两条路径，且明确native_execution_performed=false。所有试验都没有启动Isaac或GPU。

专属worker在调用base _native_child前临时设置DH116_PINCH_WORKER_V1，标记绑定version、wrapper SHA、candidate SHA和output绝对路径，并在finally恢复原环境；recipe.environment不包含此标记。native CLI在创建输出目录、导入Isaac/AppLauncher之前校验标记，缺失或不匹配立即拒绝。新增真实CPU子进程回归先复现generic入口可错误到达SDK边界，再确认generic入口被拒绝、专属入口可到达明确CPU stub；另验证base异常时标记也恢复。标记是专属调度路径的误用防护，不是额外实验授权，也不是对同一用户主动伪造环境的安全隔离。

本候选必须通过该专属wrapper调度；基础scheduler未修改，也不能替代专属多pad审计。CLI参数与recipe冻结同一个候选指纹。项目内USD的哈希键使用项目相对路径，外部后端/SDF保留真实绝对路径；同字节复制worktree的preflight指纹移植测试通过。

## CPU验证

前一提交命令：DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_deformable_grasp_runtime.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_task_budget.py tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_contact_material.py

前一提交90 passed，8.66s，exit0；528保护源码/脚本/配置SHA全匹配。新增测试先观察缺实现、异常帧持有、CLI未分派等对应失败，再修正通过。之前扩展回归83passed/1failed：旧material测试直接引用manifest已迁移绝对路径。该测试不在528保护清单；经root审查改为已有_sources解析本地同SHA USD，保留17collider和材质断言。另一个CPU测试验证本地原USD全部collider和五个compliant pad材质绑定。

私人日志：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-physical-runtime-cpu；原始失败日志、最终回归和preflight不进入Git。

scheduler收尾额外纳入 tests/test_dh116_pinch_scheduler.py 和旧 tests/test_dh116_work_packages.py；156 passed，29.96s，exit0，528保护SHA全匹配。私人日志 scheduler-gate-cpu-green.log，SHA256=b2cdf2eba3b2c421e388c780e098740f5455f76e67baeeafcf9d94eadc1193a4。

最终完整命令：DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_dh116_pinch_scheduler.py tests/test_deformable_grasp_runtime.py tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_task_budget.py tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_contact_material.py tests/test_dh116_work_packages.py

recipe command_spec_sha256=49084203742e4ff3487eeb1305fde2a342204edc06ecd34d7b4157bec38685d4；scheduler code_config_sha256=d0ec5e2f6146c82e637331acd8dac173fd46cc80425e769578d36e7ce98273de。此scheduler指纹使用package.tests=[]；recipe.source_files已冻结test_dh116_pinch_scheduler、test_deformable_grasp_runtime、test_dh116_contact_material三份测试，因此package.tests设为这三项或其子集不改变指纹。根集成若额外加入其他测试文件，按最终registry重算。wrapper SHA256=3668bf86a02067366a258ba1ac7bed97a0d973b1172f72cc03f9fe19d9c36793。

当前CLI code/config指纹：e43cdadcbf7d5a02b7b0def7abe79c0dce5c56d5720f10e9fa3349e48a02e217。recipe CPU schema校验通过。尚无本候选native执行、接触成功、Lift/Hold或训练结果，需要新冻结SHA明确批准。任何共享代码/配置变化后，必须重新生成preflight指纹与recipe approval参数。

## Integration verification

The first bare-Python integration invocation exited 2 during collection because
the existing USD Python path had not been loaded (pxr unavailable). Its log SHA
is e687ea2c010e4c5060c5ea44ace5d4ec3b09c7c1b980742a21e973ca23d10e15.
No package or environment was installed. The existing scripts/python.sh launcher
with the actual DH116_PYTHON restored the intended environment: 156 passed,
0 failed, 0 skipped; log SHA
2266d838ea217c674ab4ab257f2a162f6d933f6e6ce2f11e7f506761c2b98eb0.

Publication schema validation then rejected string dependency_scopes in the new
registry entry. The prepared transaction is retained privately as failure
evidence. Entries now use the existing scope/native object schema, binding the
two previously qualified thumb receipts. No native receipt, recipe, runtime
source, physics parameter or authorization was changed. Registry validation,
protection checks and frozen recipe verification run before publication.
