# 固定锚 SlipLifecycle v2：W4 Task1 CPU 独立包

日期：2026-11-15。状态：首轮候选 2d65f419f2e0298b436b07b4f434f8939c52bea4 因公开 receipt 别名 P1 被 QUALITY 否决；本文新增修复候选的 RED/GREEN 与 CPU 验证，待新候选精确 SHA 的 SPEC/QUALITY 复审及主线集成。没有 native/full 来源资格。

## 范围与基线

- 当前修复 worktree：`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/slip-lifecycle-v2-receipt-fix-20261115`；旧 `slip-lifecycle-v2-20261115` worktree 与首轮候选冻结保留，未改动。
- 当前分支：`codex/slip-lifecycle-v2-receipt-fix-20261115`；修复起点：`2d65f419f2e0298b436b07b4f434f8939c52bea4`。首轮起点为 `8193f9b9a00465f6e01df1de17d8b1fc1b425b76`。
- 本独立包仅涉及本报告、`src/go2_pvcnn/control/dh116/tabletop_native_reward_sources_v2.py` 与 `tests/test_native_reward_sources_v2.py` 三个 ownpaths；本次修复只改这三个路径，不修改旧源码、registry、readiness、README 或 CHANGELOG。
- 只读复核旧 `FlexibleGraspEpisode._slip` 及 step 的锚提交/输出路径，和旧 reward `_measurement`、`_read_anchors`。旧脱离保锚语义由直接调用的独立证据测试确认。

## QUALITY P1 根因与所有权修复

首轮实现把 owner 的 `_current_receipt` 直接暴露给 observe/reset/current_receipt 的调用方，后续又把其 references 当作权威历史。`frozen=True` 只能阻止常规赋值，不能阻止 `object.__setattr__`。QUALITY 未触及任何 owner 私有字段，就实际复现：替换 references 为 new-nearest/.99、修改嵌套 anchor.reference、清空 references 后重新锚定，以及向 reset 返回值注入 tick100 引用后让 tick11 接受未来锚。

原 QUALITY 证据：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/slip-lifecycle-v2-quality/reproduce-output-alias.log`，SHA-256 `c71c497b0a16cf5e26d2d30daaa75dc05600ef961518fb4243172acd415bde4f`。旧 worktree、候选与私有失败证据完整保留。

修复保持 owner-owned 完整 receipt 作为私有权威历史，新增 `_receipt_snapshot` 在每次公开出口构造独立 receipt，并逐层新建每个 FixedAnchorV2、Measurement、SourceStamp。observe、reset、clear 和每次 current_receipt 都走此边界，跨返回值及输入/输出不共享可被 object.__setattr__ 改动的 dataclass 叶子；可安全共享的部分只有包含 primitive 叶子的不可变 tuple。调用方改变 receipt 字段或嵌套 dataclass，不能改变 owner 当前状态、锚历史、几何记录或下一次合法观察。

原测试中“当前 receipt 与返回值同一对象”的断言改为值相等；新增独立对象身份检查及公开篡改测试。修复没有扩大到 Task2 adapter、native 来源资格或 solver_time 的额外规范。

## 行为与边界

`FixedAnchorV2` 和 `SlipLifecycleV2` 是 frozen dataclass。owner 首次合法 observe 可 lazy 绑定 episode，显式 reset/clear 仅接受生命周期内未使用的新 UID；清除锚和当前几何记录但保留 UID 历史。reset 后及相邻观察的 tick 必须为 exact 非负有限 int 且严格 +1；bool、未知 contacting、跨 episode 无 reset、重复/跳过 tick 都拒绝。失败不消耗 UID、不推进 clock、不覆盖现有锚或 receipt。

第一次接触必须含左右 × material/tangential 四组完整引用及四组有限非负 drift/uncertainty。UID、reference 和 reference_tick 在首次绑定时冻结，reference_tick 必须位于当前 reset 与当前 tick 之间；后续引用要求精确结构相等（含顺序），不得重新选择最近点。结果拥有独立的 tuple 快照。

- 从未锚定且无接触：`never_anchored`，无伪引用、无接触 pairs，NA 原因 `no_bilateral_contact_since_reset`。
- 已锚定而脱离：`detached`，原引用保留，`contact_pairs=None`，NA 原因 `fixed_anchor_detached`。
- 重接触仍使用原引用和实际传入漂移；测试中的 .0007 漂移不会清零。

几何漂移为独立字段，只接受真实旧 `Measurement`/`SourceStamp`，要求 valid/applicable 为 exact True、有限非负 m 值、合法 CPU/unqualified world clock、当前 episode/tick，以及同一几何 bundle 的完整 clock_identity 相等。任意 object、假同钟对象、NA/非数值、单位错误和时钟差异均以旧 `SourceFailure` 类型拒绝。脱离时保留合法几何测量，几何不能替代缺失的 contact_pairs，也不会将未知或 NA 补为 0。

公开 dataclass 和 caller bool 均不是原生来源证明；receipt 固定 `source_scope=cpu_math_fixture`、`native_ready=False`。本包没有接入 native provider、Task2 source adapter、App、GPU、Isaac 或 full Teacher admission。Task2/Task3/Task4 仍未完成；不得把本 CPU helper 报为 full 来源已接通。

## 实际验证（当前修复与首轮历史分开记录）

全部 pytest 经 Ubuntu CPU wrapper，环境前缀固定：

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider
```

当前修复私有证据目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/slip-lifecycle-v2-receipt-fix-20261115`。

| 当前修复证据 | 实际结果 |
|---|---|
| ownership-red | 仅先改测试、实现仍为首轮候选；新测试文件 **22 failed / 99 passed**，exit 1。四类原反例及 observe/current_receipt 字段、anchor、Measurement、SourceStamp 叶子的共享均暴露失败。新增另1项跨tick返回值检查在旧实现已有保护。 |
| ownership-green | 新测试文件 **121 passed**，exit 0；含原98项与新增23项。 |
| ownership-regression | 与首轮504项相同的5文件集合，加新增23项，合计 **527 passed**，exit 0；其中旧相关回归406项。 |
| ownership-protection | 在相关回归完成后执行 Repository.verify_protection；**528 个受保护文件 + 2 个历史 Checkpoint PASS**，exit 0。 |

pytest 命令后缀：新文件为 `tests/test_native_reward_sources_v2.py`；完整回归为 `tests/test_native_reward_sources_v2.py tests/test_native_teacher_reward.py tests/test_native_phase_contract_v2.py tests/test_native_teacher_context_v2.py tests/test_dh116_r0_reward_domain.py`。原始日志记录完整环境、命令和退出码。

| 当前修复日志 | SHA-256 |
|---|---|
| `ownership-red.log` | `db53b36bbcc6511a22c9ee9c738a78b8eebc4164a97333c3f346acaa1bc96f90` |
| `ownership-green.log` | `2e88200fbaac5873effea0a2a87db7ffdee8480cdae973bc4cd97138ac259ac3` |
| `ownership-regression.log` | `f824514ae56fec05d97e47ff2088e25b3416848dad32987a5bd469a1e54d4dfc` |
| `ownership-protection.log` | `cf155b6fe766d89a9bb99a8b94e115207e22e40fbf16cf3fe5c2ab225c79b3a6` |

以下为首轮历史结果；原98/504通过并未覆盖公开输出别名漏洞，不能作为该P1已修复的证据。

| 证据 | 命令后缀 / 结果 |
|---|---|
| old-evidence | `tests/test_native_reward_sources_v2.py::test_old_slip_retains_anchor_after_detachment`；1 passed，exit 0 |
| red | `tests/test_native_reward_sources_v2.py::test_fixed_anchor_contact_loss_keeps_reference_and_explicit_na`；1 failed，exit 1；原因是新模块尚不存在的预期 ImportError |
| red-all | `tests/test_native_reward_sources_v2.py`；97 failed / 1 passed，exit 1；缺少新模块 |
| green | 同一新测试文件；98 passed，exit 0 |
| green-final | self-review 将资格 flags 的 TypeError 检查直接指向 owner 后，再跑新测试文件；98 passed，exit 0 |
| regression | `tests/test_native_reward_sources_v2.py tests/test_native_teacher_reward.py tests/test_native_phase_contract_v2.py tests/test_native_teacher_context_v2.py tests/test_dh116_r0_reward_domain.py`；504 passed，exit 0 |
| protection | `Repository(worktree).verify_protection()`；528 个源码/脚本/配置文件和 2 个历史 Checkpoint 校验通过 |

上述首轮 regression 包含新文件 98 项和旧相关回归 406 项，不重复合计。当前修复完整集合为527项，CPU无未解决失败，仍待独立精确SHA复审。native 测试执行数为 **0**；实际物理抓取验证数为 **0**。

私有证据目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/slip-lifecycle-v2-task1-20261115`。每个 pytest 原始日志包含真实完整命令、输出和退出码；日志未加入 Git。

| 日志 | SHA-256 |
|---|---|
| `old-evidence.log` | `3bed9ba319e4bc9699b15e52480f874f860e3a9e6254bde8b44292572ede08d9` |
| `red.log` | `dea81b1d65d0d457c145f421567545b3c2c8ce0432e0f0fac9194cf699025490` |
| `red-all.log` | `b634988c45fdf50e9423fd10e4624b152b0b19e41d96ac7c2e4c8be45adcc64a` |
| `green.log` | `0f6ceb17cb09b08fd472739097db31087977451f3e2308f3036527757878ce78` |
| `green-final.log` | `05eb16b959d1064e2e642ca7de87ba365bccb8c458b17ed5e12196747e17b8c4` |
| `regression.log` | `83bbfbdd02ee6d1f3b36886bf0791232476efe3fd5e41f3ff4d4212644ca2473` |
| `protection.log` | `2a7ef1b5ad4b9f2fcc9c9c8f738b1c83de751ab0e191b7f5bb54dd0764dbc12e` |

## Self-review 与交付

首轮 self-review 遗漏公开 receipt 与 owner 内部历史共享的问题。本次以公开 API 复现根因，新增 object.__setattr__ 对 receipt 字段和各层 dataclass 叶子的篡改、未来引用注入、历史擦除，以及错误后同 tick 合法恢复检查；检查 input/output 与各次返回值的对象独立性。四组引用、clock、reset、几何边界和旧406项相关回归继续覆盖。新模块只复用旧 Measurement/SourceStamp/SourceFailure，没有调用旧 owner 的有副作用验证，也不修改旧版本。

仅按三个明确路径暂存，检查 staged diff 和 whitespace 后在 Ubuntu 本地提交。未 push，未写发布队列；由主代理在 SPEC/QUALITY 审查后决定集成。
