# 三指 fresh17 Teacher Worker 组合

基线 521e4f26496ccf3046b8b79df6b42e58a57b0545，Ubuntu开发/Windows同commit发布。

## 实际执行代码

tabletop_native_teacher_session.build_fresh_tripod_owner 复用已做过真实Reset的
build_fresh_supported_owned。它重新核验独立manifest，绑定原task config、SDF来源、
预期Reset seed和同一个预算/Journal；拒绝调用者给定的旧mesh plan。
返回同一fresh原Episode的NativeEpisodeOwner，不创建第二个App、不先做11槽Reset、
不推进额外物理或求解器步。构建后再次比对seed和task SHA。

tabletop_native_teacher_actual_provider._acquire 按已验证抓法明确选择
build_fresh_pinch_owner或build_fresh_tripod_owner；没有Power默认回退。
_prepare_source_pack 使用模型抓法选择精确SourceAdapter类型，继续核对同一rope实例、
实际参数getter、CUDA设备、两手、SDK2/rod0和Episode身份，拒绝交叉抓法或错owner。

tabletop_native_teacher_actual_factories.build_actual_factories 允许上述两个独立smoke来源，
保留真实lease/parent deadline、StaticContract、live Context与模型身份检查。
继续使用原NativeContextTeacher、DeformablePPOTrainer、KL Guard与Checkpoint恢复，
没有复制PPO算法或替换现场模型。

## 测试与失败证据

- 新增fresh factory绑定测试先因实现缺失失败，随后通过。
- 新增3项组合测试分别复现：工厂拒绝三指、Provider错误走二指factory、
  源打包强制NativePinchSources。修复后3项均通过。
- 完整选定回归初次52通过/4失败：旧清理夹具的preflight为空，真实抓法派发读取style失败。
  仅为该夹具补充明确pinch字段；原异常保留、日志关闭、RNG恢复等全部断言保留。
- 最终56通过、0失败、81.98秒：新工厂/组合、原Session、实际Provider、
  工厂与更新恢复案例。该集合与上一包102项有重叠，不相加宣称总案例数。
- 新正例实际构造CPU策略与原优化器；旧恢复回归也在CPU夹具执行。
  GPU/PhysX/native getter仅由明确测试边界替代，均不授予原生资格。
- 原始失败和最终日志保留Git外；没有上传原始实验数据、模型或临时日志。

## 还差什么

本包没有启动Isaac：新增原生0、控制0、求解器0、原生PPO0、抓取资格false。
虽然Provider和factory已有实际调用关系，冻结worker recipe/semantic、
worker_training_target_contract和输出账本/输入审计仍存在Pinch-only门槛。
因此当前三指worker尚不能通过完整启动准入，未启动新PPO，也未声称接入完成。

下一包将逐项接通三指冻结配方、来源闭包、checkpoint目标与按抓法的完整账本审计，
并验证三指不能接纳二指source/context。全部通过后才冻结新的短原生更新/恢复配方。
smoke更新仍不等于对向受力、固定腕保持、离架Lift或5秒Hold成功。
Power保持全握/掌面独立来源要求，旧模型、双RGB、物理、24D/371D与成功标准不改。
Antigravity仍无新可用审查回复，未修改付费/安全设置。

Registry：60通过/0失败、18.01秒。528保护文件与3份历史Checkpoint保持原SHA。
