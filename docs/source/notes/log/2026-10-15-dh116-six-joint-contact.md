# DH116 六主动逐关节与接触监测

Baseline/Candidate：b5666c2 + dirty worktree，未提交。父节点[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。用户要求持续完成逐关节验证、接触监测，再接双Panda；继续Inline Execution。

## 修改

no_load.reference新增active_index，保留食指默认；验证器新增--sweep，六主动顺序各8秒、0.2→0.3→0.2rad，合计11520步/48秒。驱动、耦合、跟踪、速度阈值与已批准首试一致。只在初始化写关节状态，此后实际PhysX积分。

启用17个手部刚体的ContactReport，ContactSensor覆盖集合必须与articulation刚体集合一致。每步记录净接触力，峰值>.01N、缺失或非有限即拒绝。独立正对照为2米外的10g小球落到静态支撑块，要求初始<=.01N、后续峰值>.01N。对照物不接触手。净法向力可能抵消且不覆盖切向接触/几何间隙，不声称完整碰撞安全；contact_clearance_validated仍false。

## 测试和实际证据

新增2测试先RED（缺参数/函数），实现后全部DH116当前51 passed in 0.50s。覆盖6维目标、非法index、缺失传感器维数/NaN、接触阈值；原49测试保留。

左sweep1已保存11520行且逐行门通过，但在sim.stop阶段滞留，未生成summary；核验PID后SIGTERM退出。保留该证据，不当作完整验收。修正生命周期：所有证据在Kit退出前保存，不先调用会滞留的sim.stop；使用已安装SimulationApp公开的close(wait_for_replicator=False, skip_cleanup=True)。这属于快速退出，不是完整优雅清理验收。

最终左右分别重跑，进程均退出0，summary和逐行数据如下：

| 证据目录（主仓库assets/dh116_v2/下） | 步数 | 最大跟踪rad | 最大耦合rad | 手部最大净力N | 正对照初始/峰值N |
| --- | --- | --- | --- | --- | --- |
| no_load_left_sweep2 | 11520 | 0.003827749464 | 5.178845048e-6 | 0 | 0 / 1.667700291 |
| no_load_right_sweep1 | 11520 | 0.003825381041 | 5.178845048e-6 | 0 | 0 / 1.667700291 |

两侧summary passed=true、physics_stepped=true、hardware_validated=false。跟踪最大约0.2194度，耦合最大约0.000297度。逐关节都有1920步；右手各主动实际极大值0.29985–0.30027rad，末值约0.2rad，确有实际往返。原USD和控制增益未变。

命令（从Go2Pvcnn运行；输出目录须不存在）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="$PWD" taskset -c 0 /home/xk/miniconda3/envs/go2/bin/python \
scripts/verify_dh116_no_load.py --side left --sweep --headless --device cuda:0 \
--output /home/xk/coding/M1/assets/dh116_v2/no_load_left_sweep2
```

右手替换side/output。顺序运行，单核限制，不提高CPU并行度。

## 双Panda前置与边界

只读核实两侧prefixed Panda各7转动关节、root_joint一个articulation；两侧DH116各11转动关节、root_joint独立articulation。合并需要去掉手的世界根约束，保留臂根，再固定连接Panda link8与DH116 base_link，不能保留两重固定或沿用O6掌变换。

尚缺Panda→DH116安装平移/旋转或对仿真零偏置同轴基准的明确选择；已通过异步问题请求用户选择。在得到选择前未创建或宣称完成双臂安装。不把手部无重力小幅通过当作全幅、重力负载、臂手碰撞、指腹、wire接触/抓握资格。

Key files：[探针](../../Go2Pvcnn/scripts/verify_dh116_no_load.py)、[参考和监测](../../Go2Pvcnn/go2_pvcnn/control/dh116/no_load.py)、[测试](../../Go2Pvcnn/tests/test_dh116_no_load.py)。
