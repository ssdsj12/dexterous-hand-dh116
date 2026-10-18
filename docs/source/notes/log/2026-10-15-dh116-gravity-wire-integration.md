# DH116 重力保持与桌面 wire 接入（进行中）

父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)，D3 的子节点；Baseline/Candidate：5e3bcc4 + dirty worktree。继续 Inline Execution，不重新选择策略。用户目标为真实闭指、夹持、抬升，当前未完成。

## 重力保持

新增 `Go2Pvcnn/go2_pvcnn/control/dh116/gravity.py`：读取当前组合模型 PhysX 广义重力，按关节名解析，手部按 Cᵀg 折算到六主动，从动施加力矩为零。所有值有限、形状与名称严格检查；超过显式合成仿真限值即拒绝，不静默截断。7 项 TDD 通过。

Panda 合成仿真最大力矩显式为40 N·m，手仍为0.1 N·m；不是厂家额定值，不用于硬件。原资产和 O6 参数没有修改。

GPU0 / taskset core0 / OMP、OPENBLAS、MKL=1：

- gravity_hold_trial1：第一步右手异常接触力30724N，安全停止，未继续。保留失败证据。
- gravity_hold_control_nogravity40：相同40 N·m上限、无重力1920步通过，排除单独提高上限造成故障。
- gravity_hold_trial2_diagnostic：添加初始body pose读取后1920步通过；IsaacLab body_link_pose_w会调用update_articulations_kinematic，提示原先teleport关节后未刷新link。
- gravity_hold_trial3_explicit_fk：显式刷新两侧运动学，移除初始body pose读取，独立运行1920步通过，最大补偿23.919897N·m。180点对照最大位置3.64235e-7m/Jacobian4.17310e-7m·rad⁻¹/速度2.92085e-8m·s⁻¹，全部门通过。

以上证据位于主仓库 `assets/dh116_v2/` 下对应目录，summary/trajectory不可覆盖。当前 verifier 记录逐body净接触力；净力为零不代表完整碰撞间隙证明。

## 桌面 wire

新增wire_table纯状态验收：41节点有限性、桌面XY范围、表面距台面<=1mm、速度<=.03m/s、没有固定边界。7 项测试通过，覆盖浮空/穿透/运动/出界/绑定/NaN拒绝。

verifier新增 --table-wire（必须 --gravity-hold）、--steps。使用本地DeformX TableRodEngine，40元素、长.5m、半径.008m、E=1e6Pa、密度1000、细步1e-5s、Isaac dt=1/240；静态有限桌面中心(.6,0,.25)、尺寸(.8,1.2,.1)。wire初始中心z=.318，落定预期z=.308。显示来自Cosserat状态网格，不添加刚性线缆或端点绑定。当前仅桌面接触，不启用手-wire接触，报告明确grasp_qualified=false。

table_wire_trial1（240步/1s）summary passed=true：最低线缆表面z=.2999972726m，支撑误差2.7274e-6m，最大速度6.79893e-5m/s，XY余量.222m，active_constraints=[]。双臂重力保持/净接触监测同时通过。全DH116专项127 passed，含真实表格/USD；无skip。

新增build_dh116_contact_meshes.py：实际四个源collision凸包、COM局部闭合PLY、源层/geometry/reference/派生SHA。初版contact_hulls_v1通过Trimesh闭合，但后端Open3D拒绝左拇指；增加同后端测试后RED，保留失败资产不用作接触。

几何排查：两个被判自交的面片近共面，距离约1e-10m、投影重心坐标在三角形外；改double PLY仍失败，排除仅导出float32根因。独立比较1e-9/1e-8/1e-7/1e-6m量化，100nm网格的派生凸包四侧全部被Open3D接受。仅派生凸包量化，不改源USD；凸包集合位移上界由逐顶点量化位移记录，<=sqrt(3)*50nm，远小于.1mm门。重新质心对齐、double PLY、导出前执行后端几何门；新contact_hulls_v2已生成。

实际DeformX MovingContactRodEngine._prepare_assets四网格接受，正体积约3.60e-6/4.32e-6m³；右拇指后端重算COM偏差约11.5nm，其他接近机器精度，均<1µm。2项实际测试通过。当前仅几何解析，不是PhysX实际cooked面或机器人接触/抓持资格。

最终本轮DH116专项129 passed（10.96s），含后端Open3D四实际网格门；没有跳过真实资料测试。

## 下一步

确认桌面落定结果；派生并核验DH116接触网格/COM变换（不能复用O6索引/局部点），接入运动接触冲量及一次性机器人反力，非目标全臂/桌面保护，然后接近→闭指→保持→抬升。全流程尚未完成，goal保持active。

Key Files：gravity.py、wire_table.py、verify_fixed_dual_panda_dh116.py及对应tests。没有提交或上传。
