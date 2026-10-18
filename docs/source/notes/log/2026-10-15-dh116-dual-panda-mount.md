# DH116 双Panda同轴仿真安装及物理联动

Baseline/Candidate：b5666c2 + dirty worktree，未提交。用户已明确同意Panda link8到DH116 base_link零偏置同轴仿真安装，不代表真实转接标定。Inline Execution。

## 资产与合同

新增build_fixed_dual_panda_dh116.py，四源USD SHA检查后创建独立资产；左右臂保持同侧并排y=±.25m，两侧手世界root_joint禁用，创建腕手identity固定关节。每侧7臂+11手=18物理DOF，7臂+6手=13主动DOF；整机36物理/26主动、2 articulation根。旧O6资产及运行代码不改，手名按所属侧articulation解析，不假设PhysX关节顺序。

新目录：主仓库`assets/fixed_dual_panda_dh116_v1/`，含fixed_dual_panda_dh116.usd和manifest.json。资产SHA256为1ab78836ff1a1cea8b2afc2e5df5795f22973c8013f52e87d39ee6138c3c6ae8。引用本地源资产绝对路径，不是可搬移portable包。静态派生资产驱动全部0，运行验证器显式配置。

## 验证

真实USD结构测试先RED（缺builder）；实现后验证2根、36DOF、手世界根关闭、双侧腕手矩阵相同、无O6 prim、拒绝覆盖。当前DH116全部测试52 passed in 0.38s，含真实Excel和真实USD。

新增verify_fixed_dual_panda_dh116.py，GPU0/headless、core0/数值单线程；固定基座、关闭重力、自碰撞开启。Panda合成驱动80 N m/rad、4 N m s/rad、5Nm、.15rad/s；DH116沿用1/.05/.1Nm/.3rad/s。从动驱动为0，真实mimic约束。初始Panda为[0,-.6,0,-2.2,0,1.8,.7]rad，手主动均.2rad。

双侧同时8秒/1920步：Panda7往返.02rad、食指往返.1rad。仅初始化写状态，后续位置目标+PhysX积分。实际每侧18关节集合检查，臂/手传感器覆盖全部刚体。手跟踪<=1度/耦合<=.5度、臂跟踪<1度、安装误差<.1mm/四元数差<.001、接触净力<=.01N，否则停止。

最终证据：主仓库assets/dh116_v2/mounted_trial1/{summary.json,trajectory.jsonl}；summary passed=true，1920步，进程快速退出码0。

| 指标 | 左 | 右 |
| --- | --- | --- |
| 最大臂跟踪rad | .000752564466 | .000752309358 |
| 最大手跟踪rad | .003763869989 | .003763914693 |
| 最大耦合rad | 5.343970776e-6 | 5.013331890e-6 |
| 最大腕手位置差m | 5.960464478e-8 | 6.143906006e-8 |
| 最大腕手四元数差 | 1.892212822e-7 | 1.894045454e-7 |
| 最大接触净力N | 0 | 0 |

实际Panda7达到.71996945/.71996927rad（起始.7），食指达到.29985094/.29985031rad（起始.2），不是只更新目标。传感器为净力门，不证明无抵消接触或完整几何间隙；正对照来自前一独立手测试，不是本次装配场景重新执行。

运行命令从Go2Pvcnn：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD" taskset -c 0 /home/xk/miniconda3/envs/go2/bin/python \
scripts/verify_fixed_dual_panda_dh116.py --headless --device cuda:0 \
--asset-dir /home/xk/coding/M1/assets/fixed_dual_panda_dh116_v1 \
--output /home/xk/coding/M1/assets/dh116_v2/mounted_trial1
```

输出目录须不存在，复跑换新目录。用summary而非单看退出码判断，异常时也会落盘失败结果。

## 下一步与限制

仿真安装及当前无重力小幅联动DONE。实际转接板、重力负载/臂全范围、手全幅、指腹几何与接触点Jacobian、桌面/wire任务、夹持抬升均未验收。未改旧O6 Play让它冒充DH116。下一阶段接入桌面场景并标定DH116指腹/掌心目标；继续保留碰撞保护。

Key files：[builder](../../Go2Pvcnn/scripts/build_fixed_dual_panda_dh116.py)、[Physics probe](../../Go2Pvcnn/scripts/verify_fixed_dual_panda_dh116.py)、[tests](../../Go2Pvcnn/tests/test_dh116_panda_mount.py)。父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。
