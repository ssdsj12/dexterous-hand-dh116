# Tripod smoke8 v4

冻结记录：统一来源/准入health typed入口；本轮已执行，结果见下节。保持原预算，见[路由报告](2026-11-22-teacher-tripod-live-health-routing.md)。

## 本轮真实执行结果（不覆盖原冻结记录）
测试HEAD e7bbec2960046c970c57b9ddafaa6a4d9591fb81，attempt-q0arlsf6。
真实2个SDK初始化步，控制/solver/PPO均0；171.223436s，外层exit1，signal=null，无强制结束。
回执SHA f8ae85d4f9f7f3b8d5652a3e786b9cd444822eafde945d6a23409ee2dfe06633。
参数native_getter、health0及独立live_preflight已越过；随后Collector只接受exact NativePinchSources，拒绝真实NativeTripodSources。
app.close返回，fault-path资源释放仍false，无shutdown_complete；不可计生命周期通过或抓取成功。
原始账本SHA580b0a3aa2cac095f5be4797102728fd4541457095a9c49946672c9cc0aca57b；失败指纹已消耗，不重跑。
