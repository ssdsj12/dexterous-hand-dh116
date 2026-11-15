# R06：独立 opt-in 故障安全生命周期

日期：2026-11-15。实施分支：codex/repair-r06-lifecycle-20261115。基线：9d3c3af0f0203a95c663641f00c8a88732a30fd7。

## 改动和边界

新增 scripts/dh116_fault_safe_lifecycle_v3.py；调用方须显式使用 repaired_lifecycle_scope() 才会临时替换旧模块的 run_lifecycle。嵌套 context 和异常退出都会恢复调用前绑定。LifetimeTrace、OWNED_NAMES、snapshot_weakrefs、run_owned_call、clock、VERSION 均在调用时从旧模块读取，并复用既有 bindings。

acquire、freeze、观察、clear、drop_alias、STOP、manager/ownership readback、可选 post-STOP update、close 及 marker 写入分别记录 attempted/returned/error。clear 即使抛异常，也会丢弃本地 sim 强引用、尝试 STOP、恰好尝试一次 app.close。异常和收尾异常仅保存字符串；捕获后清理完成的 traceback 帧，不将原异常延续到 STOP/close。trace 初始化、watch、checkpoint、event 或 before-close marker 写入失败也不会跳过 close。

按追加安全约束，update 前必须已验证 clear/STOP、冻结状态、manager release 和 weakref ownership release；clear/STOP 失败或释放未知时禁止 update。update 后在 close 前再检查 manager、ownership、freeze。未满足正常收尾门槛时只写 fault_before_app_close.json / fault_app_close_returned.json，不写正常 before_app_close.json / app_close_returned.json。最终 cleanup_stages.json 和 result.json 各自独立尝试写入；原始、收尾和写入错误均聚合为 RuntimeError。若某文件不可写，另一文件仍会尝试；没有将写入尝试声称为持久化成功。acquire 本身失败但收尾全部验证成功可以记录正常 close，result.failure 仍非空且重新抛出。

旧 scripts/dh116_pinch_acquisition_v2.py、所有消费过的 native entry/recipe、历史失败 receipt、根 README/CHANGELOG/Registry/readiness 保持原样。未复制或修改物理实现。

## 实际 CPU 验证

所有测试使用批准的 scripts/python.sh wrapper，DH116_NUM_THREADS=2、CUDA_VISIBLE_DEVICES=''；未安装依赖、未修改环境。

| 执行 | 实际结果 | exit |
| --- | --- | --- |
| 首次 RED，新模块尚不存在 | 40 failed，均为 fault-safe lifecycle missing | 1 |
| GREEN，实际新模块 + CPU callback fixture | 40 passed in 1.25s | 0 |
| 旧 Pinch / entry / scoped caller / lifetime / worker 回归 | 184 passed in 381.14s | 0 |

首轮候选 53aac71874fce9230148c9b91632888138dbf1c6 的两个 GREEN 执行合计 224 passed、0 failed；下述 SPEC 复查随后发现首轮未覆盖最终输出写入失败，故这些数字不是最终版本的全新重跑结果。新增 40 个用例覆盖健康序列、acquire/clear/STOP/close 故障、freeze 三边界异常/true/unknown、manager false/unknown/异常、ownership 未知/仍被持有、trace 和观察故障、原始与多个 cleanup 错误同时保留、Sim/Runtime weakref 在 STOP/close 前释放、保留 callback frame 无 sim 别名、准确阶段账本、close marker 分类、opt-in 正常/异常/嵌套恢复。使用真实既有 run_owned_call；没有用替身替代被测 run_lifecycle。

RED/GREEN 命令：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_fault_safe_lifecycle_v3.py --tb=short
~~~

旧路径回归完整命令：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_dh116_pinch_v2.py tests/test_dh116_pinch_v3.py tests/test_dh116_dual_pinch_entry.py tests/test_dh116_bilateral_owned_owner.py tests/test_tripod_native_scope.py tests/test_dh116_physx_lifetime_frame_v2.py tests/test_dh116_physx_lifetime_probe.py tests/test_dh116_physx_tensor_lifetime.py tests/test_dh116_physx_lifetime_validation.py tests/test_dh116_physx_lifetime_scheduler.py tests/test_dh116_physx_lifetime_debugger_stopped.py tests/test_dh116_dual_pinch_worker_v2.py tests/test_dh116_bilateral_owned_worker.py tests/test_dh116_pinch_v4_worker.py --tb=short
~~~

## 私有执行证据

证据目录：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r06-fault-safe-lifecycle-20261115。原始日志和 command/exit/SHA JSON 不进入 Git。

| 日志 | SHA-256 |
| --- | --- |
| red.log | 85cd64d8a0f3814af29fc9d84343dfadaf3059acbacbd8d5825dfbbd1fb0eb01 |
| green-first.log | fa3e2879fd84363fe407ce9211aae1041913e822ec5188ba3aef4f5a85abf3a0 |
| regression.log | 4da4429b80e760b63b93fe84bba776a7991e19cf8656d615f575855e70c8f410 |
| protection-before.log | 7c8579322d450fca03b656975cb9413722c8de0a4b83e80301afe02d47e4e7ca |
| protection-after.log | 7c8579322d450fca03b656975cb9413722c8de0a4b83e80301afe02d47e4e7ca |

实施前后均调用 scripts.run_dh116_work_packages.Repository(Path.cwd(), evidence_dir=Path('/home/xk/.local/share/dh116-autonomous-3c80166/evidence')).verify_protection()，实际验证 528 个受保护源码文件及 2 个历史 Checkpoint 均完整；没有在本分支调用 verify_local_target。diff --check 和精确三文件暂存清单在提交前检查。

## 验证限制

这些结果仅证明 CPU 故障收尾控制流、别名释放约束和兼容路径回归。没有启动 Isaac/GPU/native 实验、训练或推送，没有新自然退出证据。result 始终保留 native_qualification=false、teacher_admission=false、grasp_qualified=false、process_exit_verified=false；app.close 返回或 CPU timeline 不能授予 native/Teacher/Student 资格。旧原生失败结果仍为失败，未来任何新 native recipe 采用与原生自然退出验证须另行满足 D 阶段授权。


## SPEC 复查修复：最终输出写入故障

首轮候选的最后两个 write_json 调用没有使用故障捕获边界：cleanup_stages.json 首写失败会阻断 result.json 写入，并使 OSError 覆盖 acquire 等已有错误。第二次写入失败也会直接抛出 OSError。新增用例在修改实现前重现了这两个问题及“两文件都不可写”的情况。

修复将两个最终输出分别放入独立 attempt。cleanup_stages.json 是最终输出之前的 cleanup 阶段快照，包含当时已有的 failure 字符串，不在文件中预先宣称文件自身或后续 result 写入成功。首写失败后更新 result.failure，仍尝试 result.json；次写失败保留已写账本内容，不重复写入或覆盖该文件。所有捕获的写入错误与原 acquire/cleanup 错误共同进入最终 RuntimeError；若两文件均不可写，最终异常仍保留全部已知原因。输出故障不重新创建资源、不重复 STOP/close、也不授予通过或持久化资格。

| 本次执行 | 实际结果 | exit |
| --- | --- | --- |
| 新增最终输出故障 RED | 3 failed、40 deselected，均为预期 OSError 覆盖原错误 | 1 |
| 新模块完整 GREEN | 43 passed in 1.23s | 0 |
| 关键旧 lifecycle / bindings 回归 | 13 passed in 3.92s | 0 |
| 保护检查 | 528 源文件 + 2 历史 Checkpoint 完整 | 0 |

本次新鲜执行为 43 + 13 = 56 passed、0 failed。旧路径源码未变，因此没有重复首轮 184 用例 / 381.14 秒的完整回归；其结果仅按上述首轮候选及原日志引用。新增 3 用例断言首写失败仍尝试次写、次写失败不覆盖账本、两文件均失败时最终异常含原始/cleanup/两个写入错误，并确认 weakref 释放和 STOP/close 各一次。

本次 RED 命令：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_fault_safe_lifecycle_v3.py -k final_output --tb=short
~~~

本次完整新模块 GREEN 命令：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_fault_safe_lifecycle_v3.py --tb=short
~~~

本次关键旧路径回归命令：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_tripod_native_scope.py tests/test_dh116_pinch_v2.py::test_actual_lifecycle_releases_owned_contacts_before_stop_and_close tests/test_dh116_pinch_v2.py::test_wrapper_restores_all_bindings_on_exception tests/test_dh116_pinch_v3.py::test_empty_owners_preserves_primary_failure_and_actual_app_close tests/test_dh116_pinch_v3.py::test_v3_wrapper_restores_bindings_on_failure tests/test_dh116_physx_lifetime_probe.py::test_real_scope_orders_sync_drop_stop_update_close_and_records_two_boundaries tests/test_dh116_physx_lifetime_probe.py::test_acquisition_exception_preserves_failure_and_closes_without_owned_refs tests/test_dh116_physx_lifetime_probe.py::test_close_error_has_no_success_marker --tb=short
~~~

新增证据目录：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r06-fault-safe-lifecycle-20261115/spec-fix-final-output。首轮证据未覆盖。

| 新日志 | SHA-256 |
| --- | --- |
| red.log | 184f628d3d73419ac7d19ee3b81641fde74bb32c9524f6919fa900570910e2d0 |
| green.log | bebc51658e07e6e2b539412b0036c8865568b1f9ada3766c8b06d0bd4e414865 |
| key-regression.log | b1c07a7d3cc4760d4be03bd4ec38daabc4326961fbc8ae569b49a08be4dc5650 |
| protection.log | 7c8579322d450fca03b656975cb9413722c8de0a4b83e80301afe02d47e4e7ca |
