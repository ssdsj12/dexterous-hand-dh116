# DH116 分包实施计划及 PPO 文档合并记录

日期：2026-11-15 UTC。

## 已完成的本包内容

- 用户已批准现有执行链分包修复设计，并审阅通过书面规范。
- 随后完整读取用户PPO文档：841段正文、20张公式图、页眉页脚；SHA256为 `a6a04d7dab20048d22b152762eb07018c3269cd63bfad77ed2b11799d7ed97d9`。原件未上传Git。
- 用户明确批准采用新版本Teacher：保留371账本和历史模型，独立66项公共字段＋真实rod＋task_context；smoke先更新/恢复，full再验证完整抓取。
- 总计划分为生命周期/相机/封存、Student数学与准入、Student来源绑定、Teacher来源与运行四个子计划。各项列文件、失败回归、最小补丁、验收和提交范围。
- 原设计保留版本记录，Teacher输入与奖励条款由新修订明确覆盖。

## 验证范围

基线代码审查为1913个tracked Python文件AST通过、262项主链CPU通过。设计发布前53项调度发布CPU通过。这些是已经取得的基线结果，不是计划中新增测试通过数。

本包只提交Markdown及登记/发布文档。实施计划中的Python代码块仅进行语法自审；尚未写入生产模块或测试文件、未执行这些新测试、未创建模型、未运行原生实验、未更新真实PPO。

计划自审覆盖：
- 每项引用的原API与当前源码核对；
- R03保留read_expert的tuple返回；
- R01只复用一次estimator；
- R02分类/回归/旋转共享数学与新损失版本；
- R05同calibration换matcher拒收；
- R06/R08保留冻结旧代码、失败receipt和预算，两个动态属性错误均覆盖；
- R04不以offline BC替代真正online DAgger。sealed路径先修，live producer witness单独明确；不能把close之后的证明作为每轮optimizer的循环前置；
- 新Teacher66+rod+50context独立6键schema；mask encoder内部缺测表示不改raw NaN/validity；
- 两profile的候选权重、gamma、时长积分和事件只声明设计来源，未标原生验证；
- actual factory采用static→本次owner Reset→live核验→model/trainer/reward顺序；
- update计数使用原trainer.counters，0接受更新不扩预算；
- 32control需要256测量SDK步，setup/Reset另计，新预算未借用旧0-update配方。

## 下一执行波次

优先完成R08保留账本重审、R03源绑定、R05准入及R07投影的独立CPU修复；R01/R02数学回归可以并行。Teacher profile/context/reward/session随后组合，真实源缺项只阻断对应原生后继。

每包运行实际失败/通过回归，检查528保护文件及两个历史Checkpoint，更新报告与readiness，Ubuntu提交后经Windows导入同commit发布，核对远端SHA。PR#2保持Draft。当前所有真实训练和任务成功资格保持原状态。
