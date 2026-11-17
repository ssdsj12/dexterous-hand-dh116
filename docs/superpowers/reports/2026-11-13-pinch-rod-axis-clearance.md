# Pinch 杆轴静态姿态：左侧桌面约束出现可行区间，右侧接触仍不合格

## 基线、实现与原始证据

基线 Git 5d159d93a1119d3c11b9fa13b30ca7f57622c02e。

新增 scripts/audit_dh116_pinch_rod_axis_clearance.py：
- feasible_intervals 解析求全部顶点的正弦不等式交集；
- rotate_about_axis 明确 World Y 杆轴、WXYZ Reset姿态、列向量组合；
- table_coefficients 加入连续关节路径位移界，不以端点代替路径；
- audit_side 复用原USD/17碰撞几何、locked5 cooked SDF、主动/被动关节耦合，分别验证两手；
- qualification 独立保留桌面、非目标、初态、双指SDF深度和对向法向门槛。

原始输出（服务器本地，不上传）：
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/evidence-rod-axis-20261113-dohO4O/
含 red.log、frozen-files.sha256、geometry.log、outer.exit、pytest.log、run01/progress.jsonl 和 run01/result.json。
result SHA256：752d367a9caebe470575816d60750cd0765eaf90d5b9ebfbf06481a1ce0aee39。

原始输入：
- 左候选 SHA 3d04ccca63263b4480e1831693b8d7c70f1336170f2006b5c6a3a1f414553fe0。
- 右独立450候选历史报告 SHA 1a271578961192096d3d4aa5ec03785807c3940908bd9444fd0c285be86dff1f；只读已有 original_right-r0-p-10-b0-x0，不重跑网格。
- 已完成参数任务原生回执 SHA f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901，物理账本SHA 1981d68067f6554e63a27b190b9eb55f8663501ebdc9daca4700c73929e4ba2a。只读最后一行实际41节点，非新原生采集。
- 各侧USD、原FK、Coupling、SDF与原候选的SHA严格相等；两手身份独立。

## 有界结果

644/644 CPU几何状态；20.975秒/90秒；程序退出0。
左可行角度区间[-0.019241032,-0.016831597]rad，固定规则选−0.016841597rad（约−0.965°）。
右区间[-0.155226524,-0.009331073]rad，选−0.009341073rad。
这些区间只证明本冻结关节路径、原几何表示与指定范围内的桌面约束；不是动态或全局可行性证明。

| 独立检查 | 左 | 右 |
|---|---:|---:|
| 原连续桌面净空最小mm | 1.958800 | 1.986869 |
| 新连续桌面净空最小mm | 2.100079 | 2.100122 |
| 非目标杆间距下界mm | +0.025449 | -0.165409 |
| 初态完整沉降包络间距下界mm | +0.539662 | +1.196947 |
| 旋转前拇指节点SDF gap mm | -0.208249 | +0.580455 |
| 旋转后拇指节点SDF gap mm | -0.208242 | +0.580464 |
| 旋转后食指节点SDF gap mm | -0.250916 | +0.025559 |
| 双指预测梯度法向 dot | -0.776300 | -0.116726 |
| 本次静态检查 | 通过 | 阻断 |
| Native / Teacher资格 | 否 / 否 | 否 / 否 |

负 SDF gap仅是预测几何重叠，不是真实接触载荷。右侧负的保守非目标间距不能单独宣称实测穿透；但不能据此授权自动运动。右拇指仍有0.580mm间隙，两个预测法向不满足原−0.2对向条件，故不生成原生配方。

四个历史重点左侧指腹的**本候选全路径**净空（source与locked cooked取更低值）：
finger14 2.974742mm；finger23 2.837874mm；finger43 55.906170mm；finger53 73.391130mm。
限制项实际为非指腹source finger13，2.100079mm。以上不是重写历史失败实验的四个穿透数值。

## 初始姿态差异

左base平移变化[-1.897598,0,+1.119667]mm，绕杆World Y −0.016841597rad；右[-0.774529,0,+0.977436]mm、−0.009341073rad。
手指初始角和全部闭合增量未改：
左delta[0,.01855539082010854,.0234063279676594,0,0,0]rad；
右delta[0,.0375,.0625,0,0,0]rad。
这是绕杆轴的独立静态Reset候选，不是放大腕部动作，不改变控制器或旧姿态文件。

左侧桌面余量仅约0.079µm，不能直接当作可靠动态裕量。仍必须验证实际Reset偏差、连续必需指腹穿透深度、手臂可达性、自碰撞及新的受限原生接触。脚本保持native_ready=false，未生成或执行原生配方。

## 测试与运行命令

RED：新测试先因模块不存在而失败；证据red.log。
CPU：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_pinch_rod_axis_clearance.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_pinch_dynamic_fk.py tests/test_dh116_measured_closure_geometry.py
```
**86 passed / 0 failed，36.10秒**。新测试27项。
几何：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.audit_dh116_pinch_rod_axis_clearance --output <新的独立目录>
```
该批次已耗尽，不按相同代码配置重复运行。

原生：本包0步、0新接触样本、0新自然退出验证。
训练：0更新、0完整episode、Teacher仍0/3。WireSeg未追加训练，保持既有128步seg / 128步depth诊断记录，不作正式资格。
Tensor View所有权、已有GDB和自然退出证据未被本次CPU结果覆盖；本包没有新C++栈。

## 后继与保护

1. 左侧先解决实际可执行裕量与全路径指腹深度证据，再冻结新的原生接触批次；既有授权不代表可复用耗尽配方。
2. 右侧独立处理离散节点接触与finger13，不镜像复制左侧，也不取消碰撞。
3. R0仍缺hold_progress/instability真实资格；公共状态仍66/371。Tripod、Power各有独立几何阻断。
4. 现有528保护文件、两个历史Checkpoint、旧控制器、USD/材料/关节限制均只读；由集成发布保护器再次核验。PR2保持Draft。
