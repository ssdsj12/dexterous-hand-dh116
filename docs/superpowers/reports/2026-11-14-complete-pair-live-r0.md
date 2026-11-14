# 原子接触包保留与在线负接触 R0 来源

基线：`8b96ec7ac2e5602c5b2aa970cb5061bef507b121`。原快速编码234步原生已通过，但旧投影没有保存solver原始capacity/attempted/overflow；不能在在线R0中凭pair数量重建完整性。本包仅增加独立来源与受控入口，原物理、动作、奖励系数、控制器、模型与已消费配方不变。

## 实际实现

- `tabletop_complete_pair_capture.read_complete_pair_capture`：在原runtime同一个atomic snapshot边界取得一次owned副本，保存全部9个原始字段及边界/源码SHA。保留旧投影。无第二次solver查询、无额外物理步。
- `validate_complete_capture`：原始计数/overflow、唯一索引、单位法向、冲量及步均力矩聚合经原validator核验；投影必须与原始包一致。缺失历史计数不补造。
- `LiveEffectivePadSource.capture`：实际Reset epoch绑定；原11mesh/12substep；原子包与独立solver聚合交叉校验；左右实际SDK link pose、原USD/cooked/SDF来源、原指腹ellipse/planes/normal判据。
- `CompletePairLiveR0Observer`：只将已证明“没有有效接触”的手设置为有效负测量。任何原区域候选仍为unknown，接触深度、固定材料滑移与正向有效承托保持未资格化。stability/slip不伪造为零或真。
- 新独立entry/wrapper恢复绑定，保留原自然关闭和GDB。退出后审计逐条从实际Reset及原始物理数据重算完整接触源，与在线R0比对；连续步号、时间、数量、源SHA和measurement都必须一致。
- `complete_pair_r0_native_v1.json`：GPU3、单Isaac、1环境、234SDK步（2初始化+232测量）、29控制窗口、0PPO、180秒、每指纹一次。不是5秒Hold评估。

## CPU 回归与失败证据

命令：
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_live_effective_pad_source.py tests/test_live_effective_pad_epoch.py tests/test_live_effective_pad_accounting.py tests/test_complete_pair_capture.py tests/test_dh116_deformx_pair_capture.py tests/test_pair_substep_load_audit.py tests/test_r0_effective_pad_negative_source.py tests/test_complete_pair_live_owner.py tests/test_complete_pair_native_worker.py tests/test_complete_pair_native_audit.py tests/test_fast_r0_native_worker.py tests/test_live_r0_native_worker.py -q
```

**120 passed in 174.83s**。CPU测试墙钟不计为原生180秒预算。
日志：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/complete-pair-r0-20261114/final-regression.log`，SHA `7d8add2382cf8dd5bc9bda5bcdeec8760651d6056e465a9fb6d6498a5b8316c0`。

保留失败证据：raw/source缺模块首测；owner-first测试错用diagnostics字段；audit-red证实重复/乱序时钟漏洞；header-red证实未处理真实账本header。均修复并回归。native-prep-red为新配方不存在时失败，随后冻结新配方通过。无失败样本进入成功池。

完整234步CPU原控制owner经过新source，232个contact/R0样本、29个控制结算、退出后独立审计通过。所有CPU doubles与从旧pair重建的测试fixture均明确不是原生证据。原始历史native没有计数的行被拒绝。

小样本CPU profile（同一显式测试fixture20次调用）：删除不使用的重复surface-load归约后1.401971s→0.826850s。原始atomic校验与严格非负normal规则仍保留；不以此预测原生通过。

## 冻结指纹与资格

实验代码/配置 SHA：`fc37cc7022bff89791f139f473cba270d5d12cb3ae42943b9b6dfecfb53e6560`。
配方SHA：`7ccd421db8610cfa93793c7ab8b4b43f802d05f78f949dd06ab4a8b51672c3ee`。

本报告时新原生尚未执行；先Windows发布、校验远端与Draft PR，再按用户持续批准绑定发布后的实际scheduler指纹。
原生scope仅为`retained_atomic_contact_negative_pad_live_R0_234_steps_only`，不能扩写成正向接触、完整R0、Teacher训练或抓取成功。

下一步：一次新有界原生，审计计数、同钟、原指腹拒绝、R0 coverage、自然exit与180秒；失败不重跑同配方。approach、稳定性、材料滑移、正式R0系数和各抓法资格仍需独立证据。Tripod/Power不借Pinch通过。528个保护文件与两个历史Checkpoint由发布器核验；原始数据只读。
