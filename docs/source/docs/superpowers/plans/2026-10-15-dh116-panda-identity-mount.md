# DH116 双Panda同轴安装实施

用户已同意两侧DH116 base_link与Panda link8零偏置同轴仿真基准。Inline Execution，保留旧O6与独立手资产，不声称实物转接标定。

1. 新build_fixed_dual_panda_dh116.py：固定四源SHA；在内存新建两臂、将手基座世界变换对齐link8，禁用手root_joint，创建identity fixed mount。两臂基座复用原同侧并排位置y=±.25m。静态资产所有drive保持0。
2. 真实USD结构测试先RED，验证2 articulation根/36物理DOF/26主动DOF，手世界根禁用，腕手世界矩阵一致，无O6 prim，拒绝覆盖输出。
3. 派生资产输出主仓库assets/fixed_dual_panda_dh116_v1，记录manifest与源/目标SHA。外部引用当前本地路径，不称独立portable包。
4. 新verify_fixed_dual_panda_dh116.py：无重力、固定基座、双侧18关节名称验证。Panda显式合成PD=80/4、力矩5Nm、速度.15rad/s；DH116沿用1/.05/.1Nm/.3rad/s。这些是无负载仿真值。Panda7小幅.02rad与食指.1rad同步8秒。
5. 每步检查手耦合/跟踪、臂跟踪<1度、腕手位移<.1mm/四元数差<.001及臂/手全刚体接触净力<=.01N。任何失败停止保存，禁止扩大阈值或忽略碰撞。记录实际结果，不把拓扑检查当作完整动力学成功。
6. 更新notes与恢复入口。测试仅覆盖该初始姿态小动作，不覆盖桌面、wire、全臂路径或抓握。
