# DH116 Semi-Implicit Contact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. User selected Inline Execution; no subagents.

**Goal:** 保持原摩擦及作用反作用，完成稳定接触积分并最终验证双手夹持、抬升。

**Architecture:** 独立纯数值求解器不修改输入；后端适配器提供实际动态响应与接触几何，一次提交结果；运行入口显式选择模式并记录身份。先通过解析及细步长对照，再启用 Isaac。

**Tech Stack:** NumPy、PyElastica/DeformX、pytest、Isaac Sim。

## Global Constraints

- 不修改 k=1e4、nu=10、mu=.5、c=1000 或既有接触积分权重；不绑定杆。
- 外层100µs不变，内部候选步长必须通过收敛测试；一次更新、一次冲量记账。
- 只运行一个 Isaac，保持全部碰撞保护；用户最新CPU全核授权有效。
- 保留所有已有dirty改动；根目录为用户指定独立项目，不另建目录。
- 新算法通过前不热替换原实例，不篡改状态pin，不把离线测试当抓取成功。

## Task 1: 独立联合接触速度求解器

Files: create `vendor/DeformX/co_sim/implicit_contact.py`; create `tests/test_dh116_implicit_contact.py`。

接口 `solve_contact_velocity(free_velocity, mobility, jacobians, boundary_velocity, normals, penetration, *, stiffness, normal_damping, tangential_damping, friction, weights, dt, max_iterations=50)`。
输入维度分别为(d,)、(d,d)、(n,3,d)、(n,3)、(n,3)、(n,)；参数为(n,)或scalar。
返回 frozen `ContactVelocityResult(velocity, forces, iterations, residual)`；forces为已乘quadrature权重的(n,3)世界坐标杆侧力。

- [ ] 先写解析测试：法向单点 `u=(u_free+h*k*delta/m)/(1+h*nu/m)`；切向未饱和 `v=v_free/(1+h*c/m)`；共享两个同向接触的阻尼分母为 `1+2*h*c/m`。加入滑动饱和、负间隙、零接触、运动边界、权重与旋转。
- [ ] 执行 `bash scripts/python.sh -m pytest tests/test_dh116_implicit_contact.py -q`，先确认缺失实现导致失败。
- [ ] 实现确定性 damped Newton 核心，计算残差：

```python
relative = np.einsum('nij,j->ni', jacobians, velocity) - boundary_velocity
vn = np.sum(relative * normals, axis=1)
fn = np.maximum(0., stiffness * penetration - normal_damping * vn)
fn = np.where(penetration > 0., fn, 0.)
vt = relative - vn[:, None] * normals
speed = np.linalg.norm(vt, axis=1)
ft = -np.minimum(tangential_damping * speed, friction * fn)[:, None] * vt / (speed[:, None] + 1e-14)
force = weights[:, None] * (fn[:, None] * normals + ft)
residual = velocity - free_velocity - dt * mobility @ np.einsum('nij,ni->j', jacobians, force)
```

每个未知速度中央差分求残差Jacobian；线搜索alpha=1,1/2,...最多24次，必须降低固定尺度残差。最多50迭代；非有限/奇异/不下降抛RuntimeError。初始及最终残差以 `1e-8+1e-6*max(abs(u),abs(u_free))` 逐分量检查。输入有限、形状、单位法向、正步长、非负系数、正权重、对称正定mobility验证，输入不原地修改，返回副本只读。
- [ ] 重新运行同测试并补充非法输入、强刚性、非收敛拒绝、输入不变；针对性及全量回归通过后更新README/CHANGELOG并独立提交本任务文件。

## Task 2: 实际杆动态响应及几何适配

Files: create `vendor/DeformX/co_sim/implicit_contact_adapter.py`; test `tests/test_dh116_implicit_contact_adapter.py`; inspect `vendor/DeformX/elastica/rod/cosserat_rod.py`, `_contact_functions.py`, `co_sim/table_rod.py`。

- [ ] 将实际动态更新的加速度对外部节点力、材料力矩的线性响应与有限差分逐列核对，包含伸长因子；先建立失败测试。
- [ ] 构造每个quadrature接触的 `B`：节点 `(1-t)I,tI` 与转动 `-skew(arm)*director.T`，验证 `B*u` 和现有核中的点速度一致；验证 `B.T*f` 与原节点力/材料力矩一致。
- [ ] 桌面原有力律单独提取并核对；不得未经检查把桌面改为指腹模型。几何/响应适配完成才进入Task3。
- [ ] 执行 `bash scripts/python.sh -m pytest tests/test_dh116_implicit_contact_adapter.py -q`；对零伸长/非单位伸长、转动和共享节点分别验收，更新文档并提交限定文件。

## Task 3: 一次步进与反作用记账

Files: modify `vendor/DeformX/co_sim/moving_contact.py`, `free_rod.py`; create `vendor/DeformX/co_sim/implicit_stepper.py`; test `tests/test_dh116_implicit_stepper.py`。

- [ ] 先测试零接触与原Verlet结果相同、接触只施加一次、失败不返回合格统计。
- [ ] 使用Task2产生的自由速度/响应/接触输入调用Task1；动态子步仅提交一次解，保留解析阻尼/约束/后半步更新顺序。原显式接触回调不得再施力。
- [ ] 用 `impulse=dt*force` 计算每个通道相反线冲量和质心角冲量，桌面单独记录；测试原点总角动量平衡及运动边界功。
- [ ] 执行 `bash scripts/python.sh -m pytest tests/test_dh116_implicit_stepper.py -q`，再全回归，更新文档后独立提交。

## Task 4: 数值及性能资格

Files: create `scripts/benchmark_dh116_implicit_contact.py`; test `tests/test_dh116_implicit_benchmark.py`。原始证据写新runs目录，不覆盖旧报告。

- [ ] 先测试报告对位置/力/冲量越界必须FAIL，缺少工况不得标为通过。
- [ ] 复用已有静态及移动探针初态，双方均固定原参数。执行100ms静态、10mm/s运动、桌面接触/离开；1.25µs显式为参考，候选2.5/5/10µs逐级比较并各自减半。
- [ ] 报告逐通道位置≤.2mm、力≤max(.05N,5%)、冲量≤max(1e-4Ns,5%)、残差及角冲量平衡。预热后3组交替计时；失败时保留显式模式，不能为性能放宽容差。
- [ ] 更新文档只记录实测命令/结果，限定文件提交，不提交runs。

## Task 5: 显式选择、恢复及真实抓取

Files: modify `scripts/verify_fixed_dual_panda_dh116.py`, `src/go2_pvcnn/control/dh116/contact_runtime.py`, `stage_state.py`; test `tests/test_dh116_runtime_config_static.py`及阶段恢复测试。

- [ ] 先测试新算法/参数加入pin，旧状态不能直载；显式默认不变，失败不自动降摩擦。
- [ ] 新模式只在Task4合格后开放；状态迁移保留外部SHA/源身份且另存，清空载荷累计和成功资格。
- [ ] 经切换确认协作停止旧实例，确认进程退出后才恢复唯一新实例；重新验证接近，再闭指/保持/同步抬升。恢复运行不标连续端到端成功。
- [ ] 记录实际四路载荷、滑移、桌面余量、杆高度及稳定保持证据；没有这些不能完成目标。最后更新README/CHANGELOG与CODEX_RESUME，提交明确阶段，不擅自推送。

## 自查与执行状态

最新Task2已实现 `rod_mobility(mass, inverse_inertia, dilatation)` 与 `contact_jacobians(positions, directors, contact_points, subsamples=...)`。广义坐标为node-major世界平移，随后element-major材料转动。8项先RED后GREEN：逐列对照真实加速度核、伸长1/1.7，1/4/16子采样对照原接触核节点力/力矩与作用反作用，独立点速度和非法输入。全量719 passed in26.09s。桌面确认与指腹使用同一RodMeshContact力律，当前引擎共享contact参数，必须从实例读取而非硬编码默认值。Task3–5未实施，尚未进行新积分的细步长对照。未提交推送。

Task3接线检查：可复用原PositionVerlet，在synchronize末尾求解包含非接触kick的u_free，将接触力一次加入external，由原dynamic_rates只更新一次速度。必须用无接触等价、单次冲量、失败拒绝测试证明；禁止同时直接写速度。保留原阻尼和后半步顺序，通过独立实验MovingContact子类实现，旧显式contact回调不得同时注册。下面Task1段落为此前检查记录。

设计要求均映射到五个任务；后端适配与实际抓取仍需独立验收。Task2必须先核对动态算子和桌面原律，不能以简单点质量代替整根杆。

Task1代码/测试已完成：19项RED，实施后GREEN，补充3项回归共22通过；全量711 passed in25.90s。Inline自审未发现核心契约不符，但尚无实际杆适配/性能资格。Task1提交尚未进行，避免连带暂存现有混合dirty更改。Task2–5未实施。已确认实际转动响应为 inv_mass_second_moment_of_inertia*dilatation；桌面也使用RodMeshContact，适配时必须保持它自己的系数。
