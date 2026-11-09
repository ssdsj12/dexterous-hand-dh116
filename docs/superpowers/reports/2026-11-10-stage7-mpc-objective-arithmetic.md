# Stage 7：独立八项MPC代价算术

本单元实现 `tabletop_bimanual_mpc_objective.py` 的纯CPU算术和证据合同。它不生成动作、不安装控制器、不预测物理、不执行求解器；不能称为完整H8 MPC或抓取修复。没有新增Isaac实验或原生成功，已有二指Teacher与失败Student的归属保持。

## 输入与输出

输入候选为8个60Hz控制周期的canonical24序列，时域为8/60s，不能把8个480Hz物理子步叫H8。依赖项通过同env/episode/reset/clock的不可变 `ResidualEvidence` 明确给出单位、frame、有效mask、来源和资格；未来残差额外绑定候选digest和控制周期。`candidate_dependent`只是输入声明，不证明模型正确。

输出八项的raw、weight、weighted、逐元素cost、实际required/valid mask、来源/单位/缺失原因，以及总和与 `full_spec_cost_valid`。权重必须显式提供，是独立实验配置，不宣称已验证最优。

| 项目 | 算术与边界 |
|---|---|
| geometry | 指腹到已有目标区域的距离平方；不在本模块计算FK或接触 |
| contact | 显式accepted/required-hold mask中的误差和；几何候选不成为真实接触 |
| support | `max(0, required_support - sum(selected signed support))²`；缺失支撑不补0 |
| slip | 实际接触切向速度平方；不冒充固定锚滑移或未来滑移模型 |
| bimanual | 共同抓取相对位姿残差，translation和SO(3) log分别用显式m/rad尺度归一化 |
| joint | 每命名DOF的同单位参考尺度与正限位margin形成barrier；支持显式m/rad混合，不能替代硬限位约束 |
| smooth | 相邻候选24D及上一条实际issued24的变化平方；不修改原速度尺度 |
| risk | 明确归一化来源的collision、overload、tracking残差；不把电机预算当逐指力预测 |

当前测得的support/slip只有一个当前样本，作为常量诊断明确标记为不影响候选优化。只有真实上游预测器才能提供未来8步；本模块没有该预测器。缺失/NA、零值和非选中项严格区分；任何激活cost缺失，总cost为None；零权重也不能令NA变有效。终止、reset、跨episode、过期/错单位/错候选的输入拒绝。

## 实际CPU验证

源码SHA-256：

- objective：`14b2976664817ba2b6171da596d102105dfc9704726b1823601a02ee4972b2ab`
- objective tests：`d74cc057442d3dea4358a7e7bb4ea92680d0f65954d5938c8140258863ac0001`
- 原state：`517a41ab64bf8e719f3d6c5d33ad2db0150b85b892a72b9fc3ba1e50a0f82cc8`

2026-11-10在Windows独立CPU执行，实际73 passed、41 subtests passed、0.38s、exit0，其中48项objective和25项state。CUDA隔离，源码前后SHA一致，Native作业数0。命令为：

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES=''
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider --junitxml=artifacts/stage7_objective_cpu_public_v01/pytest.xml tests/test_bimanual_mpc_state.py tests/test_bimanual_mpc_objective.py
```

覆盖独立手/指mask、非零/负向支撑、已测常量与候选相关项区分、单位与候选/时间身份、缺值传播、混合关节单位、raw×weight及总和重算、不可变输入、溢出拒绝。合成fixture只证明这些算术和拒绝行为，不计原生验证次数。该源码亦在source11较大CPU组合中测试过，本次明确范围由上述独立实跑核定。

## 原生状态与未完成项

本提交没有使用该目标函数控制Isaac，未改变原奖励或默认Teacher。此前epoch/F1的8/8/128帧短测只证明数据完整性；它们0 terminal、0非零force观测，不构成本模块的动力学或任务验证。原Teacher的独立180步数值观测继续运行，本报告不提前引用其结果。

全17D实际FK/J与frame校准、候选动力学、64子步连续约束、求解与16.67ms准入、逐指恢复/载荷转移、共同暂停完整回合、每抓法100个唯一配对初态仍待完成。`native_qualified`、`prediction_validated`、`solve_qualified`、`action_admissible`均固定False。M0、3844D/371D/24D Tanh、原PPO/Teacher、物理/材料/执行器及30mm/5s/0.5mm标准不变；不启动B3/PPO、不替换默认Teacher，PR保持Draft。
