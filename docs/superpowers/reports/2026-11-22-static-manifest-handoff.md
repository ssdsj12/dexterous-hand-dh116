# Teacher静态来源账本的同调用传递
基线fd39382cb0d09eeb58d0107322322295ef903ad8。仅修复同一次静态合同内重复重放完整历史来源，非全局缓存。

## 修改和边界
tabletop_native_teacher_preflight._static_preflight_with_manifest保留原完整检查，返回原JSON报告与本次loader创建的不可变manifest。
公开static_preflight的签名、字段和dict返回不变；verify_static_contract只在static_ready后接收该对象，删除紧邻第二次load_reset_manifest。
易变budget的第二次检查保留；独立Builder、Source、live消费边界的完整manifest.verify均保留。
不接受调用者注入已通过标志，不缓存下次调用，不声称文件系统原子事务。原371D、24D、物理、资产和固定双RGB不变。

## 失败与CPU验证
新增真实来源重放用例先在旧Session失败：两个同SHA对象不是同一对象。只修改test-owned临时manifest，下一次调用仍严格拒绝变更。
预算I/O仅CPU synthetic limits，未授予lease；没有新日志采集、模型、Teacher查询或SDK调用。
相关preflight/session/Tripod admission 95通过（93.50s）；当前配方/worker/launch/Registry135通过（34.27s），两组不重叠共230通过0失败。
v9冻结回归先因缺package失败，随后补真实冻结配置通过；所有red/green日志及SHA留在独立证据目录，readiness列出命令。

## 性能实测
真实完整静态合同三轮交替对照：旧Session中位24.179391s，新14.751464s，节省9.427927s，1.6391倍；配置digest、reset_seed、manifest、PPO和budget等语义一致。
基线Session来自fd39382c，当前公共preflight仍执行一次完整来源校验；该实验隔离同调用重复加载差异。
真实来源manifest SHA ed086c283c177e52a5d9358c2f4887ccaea3d5c568e4770b19dc517d5f1f6d5c。
控制数据复制另行剖析19真实包：约53ms总计，候选仅省7ms，故不改控制编解码；没有为了微小收益引入新控制行为。
benchmark首次包装缺subprocess导入导致NameError，未开始计时；修正运行环境后完整六次均完成，失败包装日志保留。

## 审查、冻结与限制
独立只读审查无重大问题；Antigravity额度不可用，没有冒称新回复。
v9绑定565文件新闭包，指纹8b369481554184fd9ee1e39f73e5fa41838eeb015bd4be413e8085133a1217cb。
v6/v7/v8均未执行，不改写为失败；v5真实18SDK/2动作/0更新和退出SIG11继续保留。
新配方仍1环境、8控制、66SDK、64Rope调用/768内部子步、1更新、180s内层/240s外层；新原生运行须检查真实资源与身份。
CPU节省不代表整个原生流程必然在预算内完成；自然退出、实际更新和抓取资格尚未获得，本包0原生/0PPO。
528保护文件、3旧Checkpoint、双RGB配置及封存生命周期/Reset来源保持SHA。
下一项在同预算内验证实际clock、异常缓存释放、PPO更新与自然退出；Teacher完整抓取、正式BC/DAgger继续blocked。
