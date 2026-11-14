# 快速 owned 编码的真实 Reset/R0 原生准备

基线：`b0b9f9b91bb4f220857c9c8dfbeda767d0a39430`。原生 v1 自然退出但超180秒的结果保持失败，不复用已消耗预算。本包建立独立入口，仅将已通过同真实账本字节等价对照的编码器作为进程作用域启用。

## 代码与配置

- `verify_dh116_fast_r0_reset.execute/main`：复用原 live R0 execute，仅包裹 codec_scope；成功返回后验证原绑定已恢复并 flush/fsync `codec_scope.json`。异常正常抛出，原所有者继续负责生命周期。
- `run_dh116_fast_r0_work_packages.audit_outcome/check_codec_evidence`：原物理和 R0 同钟退出审计均保留，新增独立编码恢复证据。
- `fast_r0_reset_native_v1.json`：新 entrypoint、worker marker、源指纹；输入、父回执、物理/动作配置不变。
- 新 worker/session 回归覆盖指纹、禁止扩预算、无marker拒绝、异常恢复、完整234步CPU owner/29结算/逐事件SHA。

## 冻结范围

GPU3、单Isaac、1环境、234个SDK步（2初始化+232测量）、29控制窗口、0PPO、180秒、每代码/配置最多一次。
不是5秒Hold评估，不能声称抓取成功。

实验代码/配置 SHA：`687697a0fb9d7c9772463268624ad61cd86c2f744c4e881003176aada2dad57b`。
配方文件 SHA：`3620343e8da966f5641923b6ef49a10800d9980fe3656f67ab33c7228dd68303`。
调度器另在发布后绑定实际command_spec/code_config指纹与用户站立授权，不用旧批准指纹替代。

## CPU 证据

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_fast_r0_native_worker.py tests/test_fast_r0_native_session.py tests/test_live_r0_native_worker.py tests/test_live_r0_native_session.py tests/test_fast_owned_codec.py tests/test_r0_after_physical_reset.py tests/test_r0_live_observer.py -q`

**122 passed in 240.63s**。这是CPU回归墙钟，不是180秒原生预算。
日志：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/fast-r0-native-20261114/regression01.log`；新增入口未实现时的失败日志 `red.log` 保留。

集成时另执行新入口、完整快速CPU owner、原worker与编码器相关回归。原模型、528保护文件及两历史Checkpoint由发布保护检查核验。

## 原生资格与下一步

此报告时原生尚未执行。先Windows发布、核对SHA和Draft PR，再按已授权一次预算启动。CPU使用明确的模拟SDK缓冲，不能计为原生成功。后续以实际外层exit、GDB、时间、账本及R0审计为准。失效仅阻断本后继；无重复、无强制退出、无放宽180秒。

原R0、物理和动作保持；未启用离线必要条件扩展，training_reward仍null、Teacher admission=false。承托/稳定性/approach/slip与正式系数的剩余资格不因编码优化而获得。
