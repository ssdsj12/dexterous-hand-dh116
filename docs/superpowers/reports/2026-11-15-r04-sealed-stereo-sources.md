# R04.1/R04.2 sealed stereo source binding（仅封存离线路径）

日期：2026-11-15。开发基线：3011160cc9df31f9430b167dfd65047845e3e4f5。
分支：codex/repair-r04-source-20261115。本包仅实现已批准的 R04.1、R04.2；
**整个 R04 尚未完成，R04-live producer witness 仍未设计/实现。**

## 改动及资格边界

- 新增 verify_stereo_dataset_sources：formal 数据必须有非空有序 source_bindings；
  每个 binding 恰含绝对 regular proof 路径、proof SHA256、按升序选择的不重复有效 row indices。
  重新读取 proof 原始 bytes 并验 SHA，执行现有 _native_run，再用真实 _process_output
  重新验证 raw、manifest、records 均为同一份实际 external completion 的输出。
- dh116-stereo-source-manifest-v1 恰含 version、Teacher checkpoint SHA、native raw binding、
  records binding。raw 必须精确匹配 proof；manifest 与现有 candidate records packet 的
  Teacher checkpoint SHA 均须匹配 dataset source SHA。复用 rows_from_records，以 tensor
  dtype、shape、原始 bytes（包括 NaN payload）和带类型的嵌套结构逐项比较真正训练 rows。
  只重算 payload/manifest/proof 的新哈希，不能越过原 completion 的产物清单。
- successful BC 仅复用现有 mode、Teacher qualification、candidate flag 的分类变换；
  原有 _success_pool 的 Teacher 原生成功 proof、逐 episode 身份和完整首轮成功条件仍必需。
  DAgger visited failure states 仍非 successful demonstrations；来源校验不授予任何 native task
  资格，也不改变 force proxy、ReferenceGT 或 Actor 输入边界。
- 新增 validate_sealed_stereo_dataset，formal 数据先验来源，再执行现有 dataset 验证。
  磁盘 load_dataset 接入该入口；build_dataset 深拷贝新 source_bindings 参数。
  admit_candidates 传递 binding，并在保存前显式执行 sealed 验证。CLI 仅 admit 模式接受
  --source-bindings JSON list；bool、qualification summary、额外字段和畸形引用被拒绝。
- schema_only=True 是明确的 CPU fixture 豁免；loader 仍需 allow_schema_only=True。
  共享 validate_dataset、batch_dataset、BC step、DAgger append 未加入 sealed-only guard；
  run_online_dagger 的 collect→append→optimizer 顺序保持不变。R02 loss-v2 及其 metadata 未修改。

## TDD 与实际验证

所有执行均在 Ubuntu 专用 worktree、canonical DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python、
DH116_NUM_THREADS=2、CUDA_VISIBLE_DEVICES='' 下，经 bash scripts/python.sh -B。
没有运行 Isaac/native/GPU、采新数据、真实训练或 push。CPU 测试中的 optimizer 执行仅为既有
schema-only fixture 回归。

私有证据目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r04-sealed-20261115/。
每次保留 .command、.exit 和 .log；原始日志、临时 records 及 fixture 不进 Git。

| 检查 | 实际结果 | exit | log SHA256 |
| --- | --- | --- | --- |
| RED：新 source-binding + CLI 测试，尚无生产实现 | 63 failed, 16 passed | 1 | b74c0d0e84a8335a324ed70f3ecc84b1e4e89ccc1c962a62fbd9645f7069742e |
| GREEN：同一 targeted 命令 | 79 passed | 0 | a63c76b8d32f188f377664b3b93bf413691cfea517a95b0a091036b7f7528bde |
| GREEN：source binding、collection、DAgger、CLI、training lineage、BC mode | 108 passed | 0 | 01d3aa922a92b3e826dbbb0e3f853be97b46471836a524bc84ddfdfef1f974fe |
| Repository(...).verify_protection() | 528 protected sources + 2 historical Checkpoints | 0 | 1c310b8519d13caddeea06feea6f5949f2e8afdb737d065261169b1d30aa67b5 |

RED/targeted GREEN：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_source_binding.py tests/test_dh116_stereo_training_cli.py
~~~

完整本包回归：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_source_binding.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_dagger.py tests/test_dh116_stereo_training_cli.py tests/test_dh116_stereo_training_lineage.py tests/test_dh116_stereo_bc_mode.py
~~~

保护检查只调用以下 API，不在功能分支调用 verify_local_target：

~~~python
from pathlib import Path
from scripts import run_dh116_work_packages as s
s.Repository(Path.cwd(), evidence_dir=Path(
    '/home/xk/.local/share/dh116-autonomous-3c80166/evidence')).verify_protection()
~~~

测试覆盖 missing binding / missing proof / proof 与各产物 SHA 改变 / 异 run completion /
manifest raw 错配 / records checkpoint 错配 / 伪造自签替换文件，以及固定 proof 下 RGB 单像素、
proprio、371D label、validity mask、actions、dtype、identity、行序、label tree 的精确篡改负向。
CPU upper fixture 只 mock _native_run 的下层 native 资格；真正 _process_output、文件 SHA、
completion 产物路径/哈希、manifest、records、rows join 全部真实执行，未 mock 新 verifier。
其正向结果仅证明上层绑定，**不是 native admission**。另有不 mock 下层的 CPU proof 拒收、
成功来源也不能绕过 Teacher success gate、sealed loader、admit CLI 与原在线 schema fixture 回归。

## 明确保留的限制与下一步

1. 当前生产者没有生成此 sealed source manifest；目前没有带有该 manifest 且通过真实 native
   source gate 的合格正向 proof。因此真实 sealed positive 验收受真实来源缺失阻断，不能从
   CPU fixture 宣称成功。本包不伪造 producer、不重写旧 proof/completion，也不启动新 native。
2. 旧 formal 文件缺少 binding/manifest 时现在拒绝加载；保留原始文件，不进行自动迁移。
   这属于明确的 fail-closed 兼容变化。
3. 直接 mutable in-memory BC/DAgger 输入的来源缺口未由本包关闭。R04-live 需要真实
   collector/session 的 opaque producer witness、fsync 原始记录与每轮 optimizer 前验证；
   不能把“最终 run 自然退出”提前成在线每轮训练的循环前置条件。
4. R04-mask 的 masked/NaN 标签兼容是另一个未来任务。本包仅精确比较 tensor bytes，
   未放宽已有全 finite label/Critic guard，未补零或改写未知标签。
5. 本包不修改公共 README、CHANGELOG、registry/readiness；由 root 后续审查与统一发布。
   仅本地提交，Windows 发布及远端核验由 root 负责。
