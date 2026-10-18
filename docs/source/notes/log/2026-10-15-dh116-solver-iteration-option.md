# DH116 位置迭代对照入口

父节点：[T500 / D3-position-rate-consistency](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate：`5e3bcc4` + dirty，Inline，CPU core0。

为验证 pilot8 的位姿—速度不一致，仅将 runner 固定16次位置迭代开放为 `--solver-position-iterations`，默认16，整数范围1–255。实际 articulation spawn 消费该参数，run_config记录位置与固定4次速度迭代。未改步长、增益、接触阈值或碰撞保护。

TDD：新增9项测试先因缺失选项/硬编码16而失败，然后专项与sequence_runtime/runtime_config_static共49项通过（0.34s）。命令：`PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_dh116_solver_iterations.py tests/test_dh116_sequence_runtime.py tests/test_dh116_runtime_config_static.py -q`。首次未隔离插件的pytest被系统ROS插件缺失lark阻止；隔离插件后观察到真实RED再GREEN。

验证范围仅CLI解析、参数接线及既有静态合同，不是PhysX实际迭代读回或物理修复。未启动新Isaac。下一步保存启动manifest/正常停止证据，再对同初态有载16/4/1进行实际位姿差分与报告速度对照。保持与抬升仍未通过。
