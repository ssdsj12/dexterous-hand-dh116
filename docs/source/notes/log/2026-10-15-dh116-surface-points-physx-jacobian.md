# DH116 指腹参考点、掌心工作帧与真实PhysX点Jacobian

Baseline/Candidate：5e3bcc4 + dirty worktree；Inline Execution。父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。沿用[已批准设计](../../docs/superpowers/specs/2026-10-15-dh116-pad-jacobian-wire-grasp-design.md)，继续先上方对捏，不重做方案选择。

## D1：仿真几何参考确定

新增surface_pick.py、calibrate_dh116_surfaces.py、plot_dh116_references.py。先观察左右真实源网格，显式分别指定射线，使用首个朝外三角面相交点，保存原始面索引、重心坐标、局部点/法向、完整prim/mesh到body变换、几何SHA及5个加载文件层SHA。生成主仓库assets/dh116_v2/surface_references_v1/references.json，文件SHA为7fb840275b4a9ad3b26975ce530fce01059d3749ed6e11d1f1c9c7e048b75235。

双侧各五指独立参考加掌心定位点，共12点。四指从局部-Y侧向+Y选取；左拇指从-Y侧，右拇指从-Z侧，未镜像复制。左右掌心原点分别约(0,-.00838479,.075)/(0,-.00830538,.075)m（base_link局部）。工作帧+Z=base -Y，+Y=base +Z，+X=base +X；这是显式仿真工作帧，不是厂家实物测量。掌面局部三角法向独立保存，不用凹面法向定义整掌朝向。

左右参考标记图views/left_references.png、right_references.png已打开检查：红点在对应光滑指腹侧，红线为法向；掌体采用稀疏三角仅示意，RGB为工作帧。掌心坐标不参与本阶段接触求解。

源面到理想凸包的同射线距离：十个指腹最大6.14825e-6m（.00615mm），小于预定.1mm核验门。左右掌心点位于凹面，距理想凸包分别6.847/6.916mm；只作定位，不作为接触点，不忽略掌体碰撞。理想ConvexHull不是PhysX实际cooked凸包/接触offset验证，后者留在D3接触资格，不把本结果宣传为真实抓持。

## D2：真实USD离线与PhysX交叉验证

新增usd_kinematics.SideKinematics从实际组合USD的关节帧、轴、fixed mount和根变换建立每侧FK；按名称解析7臂+11手，主动输入13D，线性耦合后求每点3×13 Jacobian。没有沿用O6帧或索引。

离线：全部12个实际参考点、3个主动手状态(.1/.3/.7rad)、非零臂姿态、3个差分步长(1e-4/1e-5/1e-6)，共108组矩阵对照。最大绝对误差9.587096672092343e-10 m/rad，小于1e-6门。已纳入实际资产测试，不仅命令行一次性计算。

新增point_probe.compare_point并给已有verify_fixed_dual_panda_dh116.py加可选--point-references。默认无参考文件仍为原空载探针。校验参考资产/源层SHA；明确PhysX固定根省略第一body行，输入COM世界Jacobian平移到实际link+局部参考点，再折叠手耦合。实际q输入独立USD模型，用实际COM速度+角速度叉偏置对比J*qdot，不复制目标值。

GPU0无重力、固定基座、原合成PD、core0及单线程。双侧1920真实物理步；丢弃起始120步，每120步抽取12点，共180组。实际运行是Panda7与食指小幅往返，不覆盖全臂全指工作空间；所有点均接受位置/J/速度检查。

证据：主仓库assets/dh116_v2/point_probe_trial1/{summary.json,trajectory.jsonl,point_probe.json}；进程退出0、summary passed=true、physics_stepped=true、point_probe passed=true。

| 最大指标 | 实测 |
| --- | --- |
| 点位置误差 | 4.4906012701790565e-7 m（.000449mm） |
| 点Jacobian误差 | 4.268468188003971e-7 m/rad |
| 点速度误差 | 4.667267520977498e-8 m/s |

阈值分别1e-4m/1e-4m·rad⁻¹/1e-3m·s⁻¹。原臂/手跟踪、mimic、安装误差及净力门也通过。本实验没有重力负载或wire接触，不能替代完整几何碰撞保护。

## TDD与回归

ray_pick：缺模块RED→4通过；实际标定：缺脚本RED→1通过；USD点Jacobian：缺模块RED→2通过；COM测量合同：缺模块RED→1通过。最终DH116全专项113项通过（含真实表格、USD、实际选点差分），无skip。原源资产未改。

运行环境同前：go2 Python、当前目录及omni.usd.libs的PYTHONPATH、相应lib/bin/kit的LD_LIBRARY_PATH、真实Excel目录；taskset -c 0、OMP/OPENBLAS/MKL=1。

```bash
python scripts/calibrate_dh116_surfaces.py \
 --asset /home/xk/coding/M1/assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd \
 --output /home/xk/coding/M1/assets/dh116_v2/surface_references_v1
python scripts/verify_fixed_dual_panda_dh116.py --headless --device cuda:0 \
 --asset-dir /home/xk/coding/M1/assets/fixed_dual_panda_dh116_v1 \
 --point-references /home/xk/coding/M1/assets/dh116_v2/surface_references_v1/references.json \
 --output /home/xk/coding/M1/assets/dh116_v2/point_probe_trial1
python -m pytest tests/test_dh116*.py -q
```

目录已存在，复跑必须换新输出目录，不能覆盖旧证据。

## 下一步

D1仿真参考点/工作帧DONE；D2实际点Jacobian局部无负载探针DONE，非全域资格。下一步D3重力保持、桌面wire与有效碰撞表面/DeformX接触反力，之后D4接近→闭指→夹持→抬升。实物转接/驱动能力、实际cooked凸包接触、全路径碰撞及抓握仍未验证。
