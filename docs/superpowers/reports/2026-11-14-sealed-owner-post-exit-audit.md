# 原生 owner 封存与退出后完整审计

源码父提交：`58bfe72ab15409b79b83518ca17b37fa431eef52`，Ubuntu 独立 Worktree。已有失败原生 `attempt-_vy7ya1j` 的194.482秒结果保留，不能重试或因离线审计改写资格。

## 证据和设计

原生循环最后一个物理样本到资源释放间仍有约35秒；原 owner 在 `_owned_interval` 内完整重读 physical.jsonl。独立 CPU ABBA 回放证明原重审约32.1秒，保留原验证器的时钟投影约28.2秒，输出 SHA 完全一致。仅该优化不足以证明180秒原生预算可满足。

新独立路径仅把**完整账本重审**移到已有退出后审计阶段。实时物理保护、关节/几何 guard、逐子步同步观测与 R0 diagnostic reducer 保持执行；所有原始物理样本先持久化。
1. 关闭物理账本（原 flush/fsync 路径），对普通完整文件计算 SHA256、字节数、JSONL 行数。
2. producer summary 明确写 `status=pending_post_exit`、`full_audit_performed=false`、`native_qualification=false`，不是占位通过。
3. 原 owner 按已验证顺序释放 views/回调/solver，正常 app.close；不使用强制退出。
4. 外层先核对原自然退出/GDB/180秒合同，再由原审计逻辑完整重读全部样本，核对封存 SHA/字节/行数、时钟、动作、参数、限位、几何、安全阈值和完整接触/R0来源。
5. 缺数据、篡改、异常退出、超预算或任何重审失败都不得晋级。原 result.json 不被后处理重写。
6. CPU 后置审计的耗时与 GPU owner 原生墙钟分别记录；不会把端到端 CPU+GPU 总耗时冒充180秒。

## 实际代码

- `scripts/dh116_deferred_physical_audit.py`：`seal_pending`、`validate_seal`；新的 `validate_artifacts` 保留原有效右侧 validator 的完整函数体，只把“producer已重审摘要相等”检查换为封存一致检查。AST 回归固定这个唯一差异；后置完整 inspector 仍执行。
- `scripts/verify_dh116_sealed_r0.py`：独立 worker/versioned marker；只在作用域内将参数 owner 的最终 inspector 改为封存，异常也还原；沿用完整原子接触采集、实际 Reset、稳定来源 R0 和原动作。
- `scripts/run_dh116_sealed_work_packages.py`：后置审计沿用原退出、R0、逐帧原始来源核验，新封存 validator 与 marker 必须同时通过。
- `configs/dh116_execution/sealed_r0_native_v1.json`：GPU3、单实例、1env、0PPO、2初始化+232测量=234总SDK步、180秒、每冻结指纹一次。与旧 origin 配方外部数据输入相同；新 worker、源码指纹和审计范围独立。
- `tests/test_deferred_physical_audit.py`：pending/篡改/尾行/文件/原 validator 全检查保留/自然退出先行。
- `tests/test_sealed_r0_native.py`、`tests/sealed_owner_cpu_fixture.py`：CPU doubles 驱动真实 Python owner，232原子样本、29窗口、原清理顺序、原完整接触和来源审计；不计作 Isaac 原生证据。

## 失败证据

核心红测试14项、独立原生入口红测试9项因缺实现失败，均保留。
首轮集成55通过、1失败：直接复用旧测试夹具时，它要求 producer 已有 `settled_control_windows`，与本包明确 pending 合同冲突。保留旧夹具/旧入口；新增只替换结果断言的独立 CPU setup，继续测试真实 owner、SDK步数、清理回调、完整样本和后置审计。没有让 pending 对象假扮完整审计结果。
私有日志：`evidence/postexit-ledger-20261114/`；不提交原始实验数据、模型或长日志。

## 验证与晋级

CPU 命令：
`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_deferred_physical_audit.py tests/test_sealed_r0_native.py tests/test_owned_clock_projection.py tests/test_origin_r0_native.py -q --tb=short`。

本准备包0新物理步、0PPO、无新的原生或 Teacher 资格。配方 code/config SHA：`866feedb65417e219f8a577a81ab535d927a43daef21fc2b76616fc2ac74d9c2`；配方文件 SHA：`517ac231a7e2757f28ee3209e05b8039b601fb2d5041beb7c8605716b2c97f9a`。发布后按用户“批准，后续自动批准”绑定最终 scheduler 指纹，依赖与单实例锁通过才执行一次。

即使新原生执行通过，也仅证明该234步完整记录/退出/复核范围。有效双侧承托、材料滑移和稳定保持来源仍未齐全，diagnostic系数不是正式训练系数；不授予抓取成功或 Teacher PPO准入。原30mm/5秒/0.5mm、24D、371D、历史模型和528保护文件不变。

修正后完整 CPU 回归：56 passed，216.03s，0 failed。封存 owner 保留232物理样本/29窗口，原 origin 全流程回归也通过；以上均为CPU doubles，不作为原生或抓取证据。
