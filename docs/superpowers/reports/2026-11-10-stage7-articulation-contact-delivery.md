# Stage 7：完整关节读回与接触交付原生检查

日期：2026-11-10。状态：**只读观测实现、CPU 回归和限定原生采集；不是新 MPC、抓取成功或 Student 提升报告。**

## 结果与适用范围

修复实际 SDK enum wrapper 的等值识别后，独立原生 Model16 采集正常退出，128 个物理帧、1024 条环境状态的完整归档校验通过。左右手合计 44 个当前读回字段各有 1024 个有效样本；隐式 PD 估算、实际 motor torque 和未来候选动力学的左右手共 6 个字段保持 NA。

接触交付的两项独立实验均未获得全局 full-report 数据：post-step observer 触发 128 次，但其三个向量和原同步 poll 均为空；full-report subscription 没有收到 callback，原同步 poll 也为空。**空报告不能证明没有物理接触、没有桌面或支架反力。** 现有 Tensor raw 数据与全局报告分别保留，未用配置路径补造 actor/collider/partner ID。

所有成功运行均使用原 Teacher/MPC、8 个环境和原始动作来源。每种抓法的 4 个环境是标称来源副本，且本报告的所有回合均被采集预算截断、terminal 为 0；不能报告成 0/4 完整失败率，更不是各 100 个不同初态。Teacher/MPC 或 Student 原生成功数均未因本报告增加。

## 实现

- `tabletop_bimanual_mpc_articulation_measurements.py`：复用现有两只手的 `root_physx_view`，逐 getter owned-copy、实际元数据与路径映射、全 DOF/links/单位及前后时钟与 source 身份。当前 M/C/G 只标记为 current-state raw，不是任意未来 q 的动力学 evaluator。
- `tabletop_bimanual_mpc_model_capture.py`：默认关闭的 pre-reset 480Hz 观察者，记录同一实际 issued24、完整时钟、原始返回值及 NA 原因；getter、prior observer、序列化或写入失败仍保留首 terminal 和已取得的 invalid partial。
- `tabletop_bimanual_mpc_contact_delivery_probe.py`：两种互斥、默认关闭的 SDK observer。callback 入口先复制借用 buffer，再等待记账锁；独立 ledger 保留 orphan/late receipts，且不将到达窗口当成物理 frame/env 归属。
- 原 Teacher collector 显式安装 Sequence→Aux→Model→Core，按相反顺序清理；Aux 重复 finish 会重试恢复自己拥有的 hook，各归档均不重复写入。CLI 在 Kit 启动前限制独立、短预算的原 Teacher 测量路径。

原 `tabletop_command_kernels.py`、`tabletop_free_wrist_grasp.py`、`tabletop_tensor_world.py` 和 `tabletop_vision_command_shadow.py` 字节未改。M0、原 PPO v2/Tanh、3844D Actor、371D Critic、24D 顺序/尺度、原物理/材料/执行器及 30mm/5s/0.5mm 标准保持。新仿真特权读回不进入 Student。

## 实际验证与失败

| 独立目录 | 实际结果 | 资格边界 |
|---|---|---|
| source07 CPU | 561 passed、1 skipped、147 subtests；1 failure | 旧 pytest 在替换哨兵时读取 `__class__`；未启动 Native，保留失败目录 |
| source08 CPU | 562 passed、1 skipped、147 subtests | 仅测试哨兵改成 raising callable，运行源码未因该问题改变 |
| source08/post_step16_v01 | exit0；16 controls、128 frames、1024 states；完整后审通过 | 128 callbacks 与 128 polls 的 headers/points/friction anchors 全为空；8 回合截断 |
| source08/full_callback16_v01 | exit0；相同帧/状态计数；完整后审通过 | callback 0 次，128 polls 三向量为空；8 回合截断 |
| source08/model16_v01 | exit0、128 frames、1024 states，无 provider/write error；**字段绑定后审失败** | 1024 个 `model_binding` 失败，q/M 等字段诚实 NA；原始矩阵、失败与 source 全部保留 |
| source09 CPU | 574 passed、1 skipped、147 subtests | 包含独立目标函数算术候选；后者不属于本观测发布单元 |
| source09/model1_v01 | exit0；1 control、8 frames、64 states；完整后审通过 | 实际 enum 协议、布局及当前字段读回；没有任务成功或长时间稳定证明 |
| source09/model16_v01 | exit0；16 controls、128 frames、1024 states；完整后审通过 | 44 个读回字段各 1024 valid，6 个设计缺口各 1024 NA；8 回合截断 |
| source09/cpu_public_observer_v01 | **526 passed、1 skipped、147 subtests，12.35s，exit0** | 本发布单元的 17 文件回归，排除另一个目标函数候选和后续 F0 实现 |

CUDA skip 是显式 CPU-only 验证环境无 CUDA；没有以 skip 代替 Native。

source08 Model16 失败原因已在新原生采集中复现和确认：`DofType.Translation` wrapper 与其声明成员 `is False`，但 `==` 返回真实 bool True；双方 `.value` 和 `int()` 均为 1。旧 identity 检查拒绝了合法成员。修复仅在同类型、声明成员映射、严格等值与双方整数完全一致时接受；bool、fraction、未知成员、伪造 repr/name、仅提供 `__int__` 或不一致转换仍拒绝。没有从历史 repr 反推整数、重写旧 NA 或降低检查标准。

实际左右手各为 **17 DOF、23 links、22 joints**；q 为 `[8,17]`、当前 generalized M 为 `[8,17,17]`。实际命名映射保留主动、被动和六轴 wrist，不将 finger-only 11D 当完整 mounted articulation。上述是该原生场景的读回，不以配置推论代替实际维数。

## 命令、冻结与原始证据

服务器根目录：

`/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/`

`stage7_mpc_observer08`、`stage7_mpc_observer09` 均为独立派生目录。冻结基于前一已绑定资产路径的 source，并明确覆盖文件；不是完整 Git checkout。原始 USD 内容及四个原控制器 SHA 保持。

| 证据 | SHA256 |
|---|---|
| source09 `source_manifest.json` | `415de78038b5c78f5a8aacead358847bb8c2ef213c8b8f932777d3d0c3393112` |
| source09 完整冻结 CPU `validation.json` | `3d09fe0f51d0cf864a6ad19f21f2b88700ac71957ea5ced90e4a83e8b2cd9b1e` |
| source09 公开观测范围 CPU `validation.json` | `6dd7f36fa940f88436927fe500e17252b4e1c5ff86fc006893e6f6721e1f74b8` |
| 修复后的 articulation provider | `e95033a7bf5efc9c037aa16655693f47b3cba6400a48cf87a3a9a122534443a7` |
| source09 Model16 raw | `895bd853ee9d58bbab1d91c113b594f2c2e997074c6717a3db4b8de99b8ae021` |
| source08 Model16 失败 raw | `b4b683fd0abd3701f0e56d14cf4161a124ef936ae11bc7fe638e0d68310817da` |

各 Native 子目录外的 `command.json` 保存实际 argv/cwd 和显式环境配置，`exit.json/outcome.json` 保存真实退出、source/input 前后核对及文件身份。运行前、持非阻塞全局锁后再次检查唯一实例；各任务逐个显式 prepare/execute，不排队、重试或覆盖目录。启动辅助工具和原始日志/模型/归档保持私有，不上传 Git。

该发布单元的实际 CPU 命令，在独立 `/tmp/dh116_stage7_mpc_observer09_cpu` 执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
bash scripts/python.sh -m pytest -q -p pytest_mock -p no:cacheprovider \
  tests/test_bimanual_mpc_state.py tests/test_bimanual_mpc_measurements.py \
  tests/test_bimanual_mpc_capture.py tests/test_tripod_native_evidence.py \
  tests/test_tripod_raw_contact_capture.py tests/test_native_sensor_batch.py \
  tests/test_vision_teacher_collection.py tests/test_bimanual_mpc_scene_measurements.py \
  tests/test_bimanual_mpc_contact_provenance.py tests/test_bimanual_mpc_aux_capture.py \
  tests/test_bimanual_mpc_aux_integration.py tests/test_bimanual_mpc_articulation_measurements.py \
  tests/test_bimanual_mpc_contact_delivery_probe.py tests/test_bimanual_mpc_model_capture.py \
  tests/test_bimanual_mpc_model_integration.py tests/test_bimanual_mpc_experimental_options.py \
  tests/test_bimanual_mpc_delivery_integration.py
```

关闭后的独立后审先核对真实 process exit、close markers、source/压缩文件 SHA，再完整读取 EOF/CRC，检查 DTO/raw、相同 issued24、实际 480Hz clocks、env/episode/reset/no-op 和 terminal。两个观察 arm 与修复后的 Model1/16 均通过。只读分析不会给 callback 分配 authoritative frame 或 env，也不会授予力、动力学或任务资格。

采集 wall time：post-step 160.86s、full callback 158.88s、修复后 Model16 579.78s，均只覆盖 16 controls。**这是低速诊断模式，不是 60Hz 实时 MPC 性能证据。** 44 个有效字段是当前 SDK 数据；M/C/G 的补偿符号、重力 flags、armature 归属、接触/驱动预测误差与未来候选 evaluator 尚未资格化。

## 未完成项

全局 report 空数据的根因尚未确认；未切换 threshold、sleep、GPU/clone、custom geometry 或物理设置来取得数据。实际 Tensor resolved sensor/filter 元数据需要新的来源及时钟括号；旧无时钟 manifest 不重新打时间戳，旧 numeric partner ID 仍 NA。

H8/64-substep 动力学与完整 FK/碰撞验证、16.67ms 准入、逐指恢复、LOAD_TRANSFER、完整共同暂停及各抓法 100 个不同配对初态仍待后续实现和原生验收。原二指 Teacher 4/4、三指 0/4 的完整回归见[此前报告](2026-11-10-stage7-mpc-contact-source-audit.md)，不作为本次新结果。无新 Teacher 晋级、新专家池、B3 或 Student PPO；PR 保持 Draft，不合并。
