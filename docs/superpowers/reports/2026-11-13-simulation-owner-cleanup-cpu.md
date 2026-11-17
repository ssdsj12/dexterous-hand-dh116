# SimulationContext ready subscription 清理与残留 owner 诊断

日期：2026-11-13。独立基线 `6057a951681567d01d2bac5e49deab0d237256ff`，工作树 `simulation-owner-cleanup-6057a95`。

结论：已找到安装版 `clear_instance()` 遗漏的一个明确 subscription，并用原安装代码 AST 在 CPU 复现其持有关系，提供独立可选修正。**不能据此宣称 v3 的 SimulationContext 已实际释放，或找到了 constructor frame / 原生线程的根因。** 本包不修改 v3、SDK、配方、physics 或任何旧源码；不重跑失败配方，不启动 native/GPU/训练，不 push。

## 已有原生证据的只读对照

精确原件与 SHA 保存在私有 `archive-owner-comparison.json`：

| 记录 | frame-v2 正常对照 | 左 Pinch v3 |
|---|---:|---:|
| 外层 receipt exit | 0 | 1 |
| inferior signal | None | None |
| finalize stopped threads | 137 | 136 |
| after close simulation alive | true | true |
| after close simulation 直接 referrers | 5 | 2 |
| required hand/solver/view owner 释放范围 | 原 required owners false | 其余 9 owners false |
| 历史 qualification scope | physx_lifetime_frame_detachment_v2_only | single_left_free_rod_lifecycle_readback_v3_only，未通过 |

两次运行都在 close 后保留：

1. `SimulationContext._on_post_physics_ready` 的 bound method。
2. 安装 IsaacLab `simulation_context.py:290` 的 `__init__` frame。

frame-v2 另外有三个旧 acquire/lambda frame。v3 的 constructor frame 在 `resources_acquired` 时已出现，在 `before_stop / after_stop / before_app_close / after_app_close` 保持同一 object_id。v3 在 STOP 后，singleton、普通三个 timer/stage callback 和 lifecycle closure 已不在直接引用列表；仅剩上述两个 owner。v3 文件实际有 238 个 physics rows，最后 SDK 240；本审查读取数量和末值，不替代父任务已做的全账本/force 精度审计。

v3 primary exception 来自末尾 `validate_pinch_ledger` 的 `candidate hand/table/rod force accounting differs`。旧正常 frame-v2 也有同类 sim 残留，且线程数还多一个，所以不能把 sim alive 归因于该 precision exception，不能由 136 threads 推断泄漏或 shutdown 因果。原 normal qualification 验证的是列出的 hand/tensor/solver owners，**没有验证 SimulationContext 完整死亡**；本包不回写历史资格。

frame-v2 的各引用文件都与其原 receipt 内 SHA 核对。失败 v3 receipt 未包含这些 artifact SHA，因此本次仅明确记录当前原目录字节 SHA，不冒充已有 receipt 对这些文件作了历史绑定。

## 安装源码证明

| 源码 | SHA256 |
|---|---|
| Isaac Sim core SimulationContext | `0be3b0eaa44e6110cec520be9ac2c08cea17ba97db14b1b0aa60d87824deceb3` |
| IsaacLab SimulationContext | `d95f2eaa138b4fff845ac72439caa2eca5deab4f0d5609e06f163d20eef4df93` |
| carb.events _events.pyi | `5b65d0b62734824f532de87aca92829a8f5d025b2f02d33378335e62afcc501f` |

绝对源路径由新 `simulation_source_contract()` 返回并写入审查 JSON，不导入 SDK。

- core `__init__` 158–161 行将 message-bus 的 `PHYSICS_READY` subscription 存入 `_on_post_physics_ready_callback`，callback 是强 bound method。
- core `clear_instance` 208–215 行清用户 callbacks、stage/physics/event timers 和 singleton，未清该 ready subscription。
- core `clear_all_callbacks` 1273–1278 行只清五个用户 callback dict，再执行原有 `gc.collect`。它也未清 ready subscription。
- IsaacLab `clear_instance` 673–680 行仅额外 unsubscribe app-control-on-stop，再调用 core clear。
- 安装 `carb.events.ISubscription.unsubscribe()` 在 pyi 232–239 行存在。`create_subscription_to_pop_by_type` 文档还说未显式指定 name 时，从调用 traceback 生成名字。这个说明只是 constructor frame 关联的线索；其 C++ holder 的实际 Python-frame 生命周期并未由现有记录证明。

现有记录只有 sim 的直接 referrers，不能确定谁保留 constructor frame。其上游是 event-name trace、其他 profiler/cache、异常链或别处，仍为 NA；没有擅自清掉该 frame。

## 独立最小实现

新增 `tabletop_simulation_owner_cleanup.py`：

`clear_known_simulation_instance(sim, contract=...)`：

1. 在 mutation 前重验三个源文件的精确 SHA、contract 以及实际 callback / clear_instance 方法文件身份。
2. 仅对已知 `_on_post_physics_ready_callback` 调用安装 API `unsubscribe()`，返回后清该字段。
3. 继续调用原 `type(sim).clear_instance()`，不替代 SDK 原清理。unsubscribe 抛错时不假清字段，仍尝试原 clear 并返回失败。
4. 保存的结果只有标量，表示 API 是否返回。没有把该结果当作 sim/native 已释放。
5. 异常只保存文本，并断开本 helper 调用链中已完成的异常 traceback；finally 清本 helper 的 sim/handle/bound-method 别名。**不会搜索并清除任意 constructor frame。**

没有新增 `gc.collect`、STOP、app.update、SDK/solver step。原 clear 内部既有 gc 次数保留。没有直接调用任何 native destructor 或修改 SDK 内部缓存。

未来独立批准的 owner 可以将 `functools.partial(clear_known_simulation_instance, contract=contract)` 作为 clear_sim callback；partial 只保存源合同，不保存 sim，也避免新建含 sim 参数的包装 lambda。当前冻结 v3 没有接入此 helper，不产生重跑授权。

`probe_simulation_owner_edges(weakref, max_referrers=16)`：

- 只读现有 Python referrers，重点展开 constructor frame / ready bound method 的上游一层，输出全为 bounded 标量字典。
- 不读取 `f_locals`，不清 frame，不 collect，不遍历 native threads，也不保存 referent。
- 过滤这次诊断创建的引用容器；输出是否截断和原 Python referrer 数。
- `gc.get_referrers` 不能看到所有 C++ holder，故 `root_cause_proven/native_release_verified=False`、`native_thread_attribution=None` 始终保留。将来批准运行的 owner 可在已冻结边界采样；本次只运行 CPU 复现。

## 可执行 CPU 复现

测试从**实际安装 source AST** 提取 ready 注册语句、`clear_instance`、`clear_all_callbacks` 和 ready callback；只用 event bus / subscription / backend 的 CPU doubles 承载，没有 import/实例化真实 SimulationContext。

- 原 clear 后，原 singleton 已为 None，但 event subscription 的 bound method 仍保持 sim 活着。
- 新 helper 依次 unsubscribe→原 clear，CPU sim 在 caller 释放后死亡。事件序列验证原 clear 内部一次 gc，helper 没有追加 gc。
- 另一个 fixture 独立保存 constructor frame。新 helper 删除已知 callback 后 sim 仍活着，两层 probe 能指明该 CPU frame 的 `creation_frame` 上游字典；释放该显式 CPU holder 后 sim 才死亡。这是未知 upstream 情形的复现，**不是声明真实 constructor frame 来自这个 fixture 模型**。
- unsubscribe 故障仍调用原 clear、不假清 subscription，保留异常对象也不会增加 helper 的 native alias。
- 不支持的类型/source、改 SHA、重复 absent、probe bounds 均保持明确状态。
- **直接调用冻结 v3 `run_lifecycle`** 的两个 CPU case：在 acquire callback 注入同类 validator precision 异常，用新 clear callback（counter 240 为显式 CPU fixture，不执行 238 次测量）；独立 constructor holder 不存在时 close 前 sim 已死亡，存在时仍活着且诊断不宣称根因。没有修改原 v3 方法。

最终相关 **159 passed in 58.03s**，13 新增 +146 既有，8 文件，exit 0。初始新增测试因模块缺失 collection error 后实现；最终命令：

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q`

- `tests/test_simulation_owner_cleanup.py`
- `tests/test_dh116_physx_tensor_lifetime.py`
- `tests/test_dh116_physx_lifetime_frame_v2.py`
- `tests/test_dh116_physx_lifetime_probe.py`
- `tests/test_dh116_physx_lifetime_validation.py`
- `tests/test_dh116_pinch_v3.py`
- `tests/test_dh116_pinch_v2.py`
- `tests/test_deformable_grasp_runtime.py`

camera peer 只读审查新 module，source/mutation/异常/owner-probe 边界未见新增阻断；没有重跑测试或 native。该短审不替代实际 SDK unsubscribe 后的原生 readback。

## 证据与剩余项

`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/simulation-owner-cleanup-cpu-bir0terq`

- `archive-owner-comparison.json`：两次原 receipt、全部阶段直接引用、源码合同、当前原件 SHA；线程因果仍 NA。
- `cpu-owner-reproduction.json`：两种 installed-AST/CPU event-holder 复现，是否存活及二层只读 owner edges。
- `final-cpu.json/log`：精确 argv/env、exit、耗时、原始测试结果。

本提交仅新增 module、tests、本报告三个文件。已经解决的是可执行 CPU 模型里的已知 subscription 持有路径，并准备了受 source 约束的独立清理调用；真实 constructor 上游、真实新清理后的 liveness、任何 native destructor/线程因果仍未验证。需要未来独立批准的版本在同边界补读两层 owner，不重复当前失败配方。

## SHA256

- `src/go2_pvcnn/control/dh116/tabletop_simulation_owner_cleanup.py`: `e93c53968a64f76301bddb9876ce12ce6b7692dc95b4ae6068fa115c7f5736de`
- `tests/test_simulation_owner_cleanup.py`: `bb3eb026fe15ff981ed8d7aa40e4f5c842150ca810e76995f2878a4cb7f791f4`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/simulation-owner-cleanup-cpu-bir0terq/final-cpu.log`: `91e221f7fe17907482efa672ae3270eb0e13c8a7bc3e206869b6aabbca6e83ee`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/simulation-owner-cleanup-cpu-bir0terq/final-cpu.json`: `45f37bac11db247edef7d6c18f07163822075338293e79f767d65cec61ad48dd`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/simulation-owner-cleanup-cpu-bir0terq/archive-owner-comparison.json`: `b3f2d67f380bb159383e665fe1899269d1630e6740312a596469d289a2c2e876`
- `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/simulation-owner-cleanup-cpu-bir0terq/cpu-owner-reproduction.json`: `0bf222a479b37c0659a08f66e3870f77c5f4f6f5f39f832cc6484249af7f2d89`
