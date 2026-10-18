# DH116 V2.0 URDF 线性近似有效域

Baseline/Candidate：b5666c2 + dirty worktree，未提交。阶段：T500/DH116控制耦合；用户明确改选配套URDF角度公式作为仿真近似，不再让Excel坐标歧义阻塞这一路线。不是厂家真实机械标定。

输入：DH116 V2.0手册第3.3节，上限依次80/90/70/70/70/70度。文件SHA256：bac73dfb4264d69970968fcda9bd71238be70e46517eb6bf6b625d5b48aba6bf。旧版DH116手册范围不再作为当前型号约束。手册的力和开合时间不转换成关节力矩/速度。

修改：linear_coupling.py新增active_limits()，返回手册、源URDF主动限位及五个仿射从动限位反推得到的交集，expand使用该交集并严格拒绝越界；无静默夹紧。原MIMIC与Jacobian列折叠不变。没有采用MJCF二次公式或DH2116五阶公式；Excel/PCHIP保持独立未接线。

有效范围（rad）：lower=[0,0.00697725445/0.88178789,0,0,0,0]，upper=[radians(80),1.030,radians(70),radians(70),radians(70),radians(70)]。主动角单位、6→11输出与列命名合同不变；原URDF允许但超过V2手册的输入现在拒绝。

验证：已有7测试基线通过；新增测试RED为7 failed/7 passed（5项超手册范围未拒绝、2项缺active_limits）；实现后实际两份Excel与线性合计45 passed in 0.21s。覆盖边界、所有从动限位、域外拒绝、输入不修改、返回数组无共享状态、上下界邻域的6行Jacobian链式差分（atol1e-8）、原有列重排。命令从Go2Pvcnn执行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
DH116_TABLE_SOURCE_DIR=/home/xk/下载 PYTHONPATH="$PWD" \
taskset -c 0 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
tests/test_dh116_linear_coupling.py tests/test_dh116_table_core.py -q
```

未改USD或驱动，未运行PhysX、空载开合、Panda安装或抓握。当前有效域只在调用expand的路径生效，不代表USD物理限位已同步。下一步审查USD左finger11轴定义（此前空token不能默认为X），在新派生资产中同步限位，再以明确的DH116仿真驱动配置做空载验证。未核实的厂家驱动力矩/速度仍不得沿用O6。

Key files：[模块](../../Go2Pvcnn/go2_pvcnn/control/dh116/linear_coupling.py)、[测试](../../Go2Pvcnn/tests/test_dh116_linear_coupling.py)。父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。
