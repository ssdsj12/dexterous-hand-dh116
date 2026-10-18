# DH116 启动证据与协作式停止

阶段D3/D4，父节点：[T500位置速度/抓握验证](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty，Inline/core0。

针对pilot8 SIGINT跳过finally导致最终报告丢失，新增run_evidence.py：在SimulationContext创建前以独占文件写入run_manifest.json，包含run_config和asset_manifest、initialized_not_qualified与grasp_qualified=False；先严格JSON序列化再创建文件，flush/fsync文件。不宣称断电级完整事务。

runner每步首先check_stop_request(args.output)，检测输出目录STOP_REQUESTED文件/目录存在时抛出明确RuntimeError，经既有异常/finally路径保存已完成探针和summary、passed=False，不删除停止请求。没有修改控制、夹力、步长或保护。

TDD新增5项先观察全部失败，再实现；54专项通过。完整`tests/test_dh116*.py tests/test_fixed_dual_panda_o6_moving_contact*.py` **438 passed in51.03s**，日志/tmp/dh116-start-stop-full.log。测试使用PYTEST_DISABLE_PLUGIN_AUTOLOAD=1、明确USD库/PYTHONPATH、taskset core0和数值线程1。运行期间pilot9仍旧快照，没有热加载这些改动。

实际Isaac正常停止演练尚未执行；应等pilot9结束后单独验证。不用SIGINT来验证此功能，不把新文件放进当前旧runner目录假定其会响应。当前pilot9 session89681/PID78902仍活，原12000步1次迭代渐进闭指试验继续；保持/抬升未验证。
