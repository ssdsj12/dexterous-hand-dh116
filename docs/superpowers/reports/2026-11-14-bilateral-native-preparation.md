# 双手同杆有界原生入口准备

基线506c337770174179ad09ce6ca27915f3fb93a4f4；源分支接续已发布合同同源提交87d88724b37d96cd2aa63f7e93081dffd981898e。本包无新原生步、无PPO训练。

## 运行逻辑
新增prepare_dh116_bilateral_owned_native_interval.scenario_scope，独立绑定双手配置、动作、实际几何校验与逐步原参数读取。旧margin版本、旧左右操作和其历史SHA不改。
新增verify_dh116_bilateral_owned_pinch.preflight/execute/require_worker，绑定原生owner、单实例worker、冻结指纹和真实父回执。
新增run_dh116_bilateral_owned_work_packages.cleanup_case/audit_outcome，只接受2初始化+232测量=234步、1环境、0更新、GPU3、180秒配方；退出后独立校验所有物理行、参数、生命周期、GDB和外层自然退出。
validate_dh116_bilateral_owned_contact的审计体与原校验AST一致，仅显式导入独立operation；原保护和完整退出条件不削弱。

## 两侧证据
右父回执SHA3e12bb3204f46df748e1de7b35b77306e57be016e6a7f626249f0790d466c83d，scope right_owned_pinch_2_setup_232_guarded_contact_diagnostic_only，actual234、自然exit0。
左独立事后审计SHA57cbc1b6b97eb594368e4702746241c3be3a3b0c562d99293ab2b470b30706cf，原scheduler回执SHA97f83e1e001ea28cfa08ebcf4017d717f7adcc11aad38402280e780f48311746保持failed。读取旧原生文件前重新核对原运行/配方/每个artifact SHA；不把事后审计称作新原生。
两侧接触仍是不同episode；新双手结果只能由新配方产生。

## CPU验证
- 首个新入口测试预期失败：bilateral native worker missing。
- 新worker及实际Python owner合计19 passed，0failed，236.77秒。完整232测量物理接口行、SDK3～234；第11步注入保护时只保留9测量行，未额外推进，原terminal保留。
- 16项worker测试包括真实子进程再进入独立wrapper、作用域恢复、SHA/预算、旧validator拒绝、资产不进入发布清单。
- 3项owner测试使用显式合成SDK/solver，因此仅CPU，不能计入原生。
- 发布前再次运行worker16及新合同/几何/控制审计27项，共43去重测试；不无理由重复已验证的完整合成owner。
日志 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-worker-green01-20261114.log，原失败日志bilateral-worker-red-20261114.log均私有保存。
命令：DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_bilateral_owned_worker.py tests/test_dh116_bilateral_owned_owner.py

## 新配方
configs/dh116_execution/deformx_bilateral_owned_pinch_v1.json。预算234物理步、29完整控制窗口、1环境、0PPO、180秒；首次测量前2初始化计入预算。自由杆、真实重力与桌面、无附着；保持原物理参数、速度、关节、Mimic及碰撞保护。
Windows发布并核对后，由既有“批准，后续自动批准”范围绑定新的配方/代码指纹，资源锁通过再执行一次。不得重跑旧单手预算。
当前未执行该原生配方。预期资格仅bounded contact execution，不是5秒Hold、无支撑抓取或Teacher。
剩余：同一episode双侧真实接触；R0稳定性/固定材料锚滑移及371D来源；Tripod/Power独立资格；旧相机失败不被此测试替代。
