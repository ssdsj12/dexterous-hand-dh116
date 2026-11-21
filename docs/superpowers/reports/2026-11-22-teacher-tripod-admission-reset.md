# 三指 Teacher 静态准入与原 Reset 执行链

基线 0dcf5a11deba526082697f286820f9eac53ba26f；Ubuntu开发/Windows发布。

## 根因与实际修复

三指真实17槽Reset与独立Adapter已有，但原入口仍只认可二指manifest、
11槽读回与二指Reset。不是通过降低几何或自然退出标准解决，而是选择已有对应实现。

- tabletop_native_teacher_preflight.load_reset_manifest：三指明确转到
  load_tripod_reset_manifest，重新读取原始来源，保留原二指加载器。
- static_preflight：只有独立三指manifest有效，才构造其六指腹
  MeasuredTripodResetApproach并检查独立Adapter。Power仍拒绝；不转用二指证明。
- tabletop_native_teacher_session.verify_static_contract：允许已具备独立来源的
  Pinch/Tripod smoke，保留模型抓法、reward/PPO数值、预算/真实kernel锁核对。
  full contact/slip/hold profile仍无条件拒绝缺失的完整Adapter。
- NativeEpisodeOwner.reset：三指使用原DualHandPhysicalEpisode.reset的
  supported_mesh_plan与独立record_reset_readback入口，写入原NativeJournal。
  数值审计、原ClearanceGuardedExchange、真实Reset/动作history和资源租约保留；
  绝不先做11槽Reset再追加，也不增加隐藏物理步。
- validate_live_mesh：按已验证17槽plan版本选择既有直接native读回；
  相同owner/clock/current/target/solver必须匹配。旧11槽行为仍使用原validator。
- live_preflight：只接受抓法匹配的不可变manifest与独立距离reader。
  三指另外要求真实PowerSharedRodRuntime的17槽身份和原生源scope；
  CPU fixture不能借此取得原生资格。完整live入口仍需实际Collector运行确认。

## 验证

3项新测试先失败、修复后通过。第一次Owner测试夹具误补丁局部import名，
随后明确隔离GPU数值getter（由既有数值审计测试负责），才在旧11槽路由处失败；
失败日志全部保留，未改变生产校验以迎合夹具。

最终选定102项CPU回归通过、0失败、133.81秒：
三指准入/Owner桥、原preflight、原Session、三指Adapter、
原supported Episode及17槽Reset。三指native manifest采用历史真实保存数据；
临时预算文件和kernel锁仅属于CPU测试目录，不是实际原生运行授权回执。

新增测试同时验证静态通过不产生live/Teacher资格；缺少独立manifest仍拒绝；
第17槽current/target、owner或clock漂移会拒绝，且无SDK/求解器推进；
原Episode只Reset一次，17槽读回有独立持久化记录。

## 尚未完成

本包新增原生0、控制0、求解器0、PPO0；三指实际worker尚未接通。
下一包将原fresh17 builder交给NativeEpisodeOwner，再接实际provider、
worker语义/目标/ledger审核和小预算更新恢复。没有把CPU通过写成原生抓取成功。
Pinch/Tripod/Power最终仍须分别满足30mm/5s/0.5mm、无外部支撑与真实力来源。
双RGB固定配置、24D/371D、原资产、关节保护和R2关闭状态均未改。

Antigravity本轮重新查看仍显示额度耗尽，没有新回复；未修改账户或付费设置。

Registry：60通过/0失败，15.29秒。528保护文件与3份原Checkpoint保持原SHA。
