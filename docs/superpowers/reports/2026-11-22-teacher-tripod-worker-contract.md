# Tripod Worker 冻结合同、逐物理步来源和输出账本
基线 db9400f1fb024f79e90e68fdce30321500d34d3a；Ubuntu开发/测试，Windows导入同commit发布。

## 实际修改
- tabletop_native_teacher_worker_contract.validate_teacher_recipe_v2：增加独立Tripod源码闭包要求，保留原Pinch与全部有限步数、环境、更新和JSONL预算；Power不回退到其他抓法。
- training_semantic_config：模型、preflight、独立Tripod Reset manifest的schema/抓法相符；原语义SHA继续绑定实际来源，未改为警告。
- worker_training_target_contract：仅接受对应抓法的真实来源加载manifest类型；仍为原41个材料节点，不重标Ribbon表面点。
- tabletop_native_teacher_worker_artifacts._journal：按真实抓法要求policy_tripod_distance或policy_pinch_distance。混用一个中间操作即拒绝；完整原始行与派生视图不丢失。
- tabletop_native_teacher_actual_artifacts：fresh/resume均传递实际抓法，封存时重新构造同抓法输入审计。
- tabletop_native_teacher_actual_inputs.audit_teacher_input_sources：逐Reset和全部SDK物理行核验17槽/14个源link、PhysX_link_origin、m和左右绑定；中间行错绑也拒绝，不仅核对Actor端点。源pack的手根与common packet对齐；context抓法、clock/owner、原始模型输入重建继续严格。
- tabletop_native_teacher_actual_update：按模型真实抓法封存审计。
- tabletop_native_teacher_worker_lease._verify：真实持锁CPU子进程复现“Tripod配方+Pinch配置”曾可通过；现在App创建前拒绝style/profile混用。保留原PID、锁、deadline、源SHA和授权身份检查。

## 验证及保留失败
新增10项行为案例，分别保护配方来源/语义/目标、输入来源/中间物理行/账本，以及真实持锁进程的抓法一致性。先失败后修复，不使用源码字符串断言或新增测试专用生产接口。CPU账本明确标为cpu_fixture，不能产生原生准入。
最终三个不重叠文件组：161通过/0失败（278.02s），106/0（79.26s），102/0（58.32s），共369通过/0失败。
命令均为 /mnt/mydisk/xk/envs/dh116/bin/python -B -m pytest -q -p no:cacheprovider 后跟readiness.cpu.groups中对应tests。
初次审计回归93通过/3失败：两个旧CPU mock未接收grasp，另一个旧resume fixture缺style；仅修正夹具，原封存/清理断言保留。
另一次lease回归选错不存在的文件名，pytest exit4、未执行测试，日志保留；更正命令后102通过。
原始red、首次失败和最终日志保留Git外，readiness保存SHA；不上传实际账本、模型、用户文件。
本包按isaaclab-auditing-tests和verification-before-completion审核测试价值与证据；没有套用Isaac Lab develop的新训练CLI或升级运行环境。
Antigravity当前quota阻断，无本包新外部审查意见，未伪称经其确认。

## 准入边界和下一包
本包Isaac实例0、物理/求解器/控制步0、原生PPO更新0；369项是CPU测试，未授予接触、抓取或正式Teacher资格。
独立Tripod Reset历史真实回执可只读复用，但本包尚未冻结并执行新的Tripod PPO配方，不能把CPU工厂接线写成原生训练成功。
下一步：以当前发布SHA冻结Tripod短smoke8真实worker配方和完整源码闭包，检查实际单实例锁/GPU/预算，执行新身份的一次有限原生更新；若真实回执合格再绑定恢复诊断。失败仅阻断自身后继。
Pinch、Tripod、Power仍分别验收。保持原双RGB、24D/371D、528保护文件和3份历史Checkpoint，R2/Force Proxy关闭；不改资产、Mimic、关节范围、30mm/5s/0.5mm标准。


Registry回归60通过/0失败（19.33s）。528保护文件、3份历史Checkpoint及两份固定双RGB配置已重新核对原SHA。
