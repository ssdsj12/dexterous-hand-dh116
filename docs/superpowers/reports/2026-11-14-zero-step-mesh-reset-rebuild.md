# 原双手自由杆 mesh Reset：独立零步重建适配器

基线 06b11c3c95ff1c2937eb41e861b6fbffed944e32。本包只新增 tabletop_mesh_reset_order.py、test_mesh_reset_order.py 和本报告。没有修改旧 runtime、backend、控制器、原始物理参数、历史证据或 consumed recipe；没有启动 Isaac/CUDA 物理、编译、训练或推送。

## 根因与选择

锁定 backend 的 set_mesh_poses(teleport=True) 只写 d_tgt/d_tele；真正提交发生在下一次 begin_step 的 stageMotionK。原 reset 后首个正常 step 又写 teleport=False，覆盖该 flag。因此第一帧从构造时的 mesh current 插值，不能把原 234 步完整执行／自然退出等同于 mesh Reset 已同步。后续 episode 同样存在此问题，仅移动首次构造顺序不能修复后续 reset。

经父代理审查选定独立 opt-in 方案：完整执行原 DualHandResetTransaction，包括实际 q/qd、mount/root、targets/gains、双 composer 全零写入及 kinematic refresh；随后先同步并记录现有 rod 状态，执行 clear_meshes，按原顺序重新 add_mesh 同样的 11 个 SDF。每次 add_mesh 内部实际执行 slot commit 和 solver-stream synchronize，不执行 rod/SDK step。左右各五个 original pad，table 保持 slot 5；左右分别 0–4、6–10。

首次已有的 11 次构造 add 没有删除：它们在测量 stepping 前被 reset 后的 11 次重建替换。这样无需复制／拆分原 hand reset，且同一 rod 的后续 episode 使用相同修复。代价是每次 reset 重新分配并上传同样 11 个 grid。几何是否安全仍由既有几何门负责，适配器不假设初态无碰撞。

## 真实 API 与同步

源码根：
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/parameter-getters-build-e96wtu/source

| 源码 | SHA256 | 行与作用 |
|---|---|---|
| csrc/sdf.cu | e5b3b1b00881aa1aff1697214f36b4e12310f8d62655455be7ce6b6df0369c12 | 37–62 target/teleport；140–160 add commit+sync；195–202 clear；267–309 begin/end |
| csrc/sdf.h | e9bf17d2980f58b6ebff76b42dad8322eff175c30c0c87959fd6c122faaec4fb | 94–116 private commit、pair/current/target buffers |
| csrc/bindings.cpp | 1be4aad4bfc6de1c4a2c44e891d57ce7e163b78c24a8bf99619d1cb1fe0ce836 | 1023–1063 target-only setters/clear；1151–1214 native parameter snapshot |
| csrc/batch.cu | 899e95da5aab86a40442be8effbcadbd6d2923a27b9554475d38f6fef631d868 | 401–415 reset_envs 不提交 mesh；422–426 gather 仅杆状态 |
| 既有 .so | 2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec | 不构建新库 |

生产验证复用 NativeParameterEndpoints.source_basis()：72 文件源码清单、实际 .so、原零步参数 getter 来源。clear_meshes 自身没有显式 cudaStreamSynchronize，且其 cudaFree 返回值未单独检查。新事务先调用真实 parameter_snapshot()，该 getter 明确同步 solverStream 后 blocking D2H，再清 grid；每个 add 又同步同一 stream。不依赖未证明的默认流／隐式 cudaFree 同步，不调用 step_free(0)、gather 或额外 reset。

clear_meshes 使 pair 捕获无效，释放 grid、清 grid/box 容器和 pending/moving，不释放 d_pairs 或改变 pairCapacity。add_mesh 保留其他 pending 状态，只提交当前新 slot。全 11 槽完成后对应 teleport flags 均经原 commit 清除。下一原 begin_step 清 count/force accumulators 并重做 broad-phase。旧 grid/collider/view 地址不能再用，逻辑身份按原槽位重建。

公开接口无 mesh-current、pair-capacity 或 force scratch reset 的独立 getter。receipt 明确 native_current_pose_readback=None、pair_capacity_readback=None；不把 scratch force 零值当 reset contact。原 reset 已清双手 composer；后继必须使用原新 Episode/exchange 的空 reaction 初态，旧 owner 按既有生命周期结束。

## 来源与事务边界

入口：
prepare_mesh_reset_plan(project, prepared, rod, sdf_module=实际已加载sdf模块)
RebuiltMeshReset(seed, episode_uid=..., mesh_plan=plan).run(sim, rod, bindings, owner_resets=...)

后继 scope 可用该类作为独立 reset factory。本包尚未接入默认 builder/native launcher。plan/receipt 仅保留 owned 数组和标量。

生产验证不只检查自签摘要：

- 原 task JSON 硬锁 SHA 8b4f7f8d3fb96c12b9794a7c5daaa2b65f9231699348d5635044bd3f5ac75d63，重读 table center/size/top。
- 固定 SDF result.json SHA db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab 重建十个 pad 名称、side/body、NPZ path/SHA、capture SHA、frame/unit。
- 逐 NPZ 重读 values/origin/spacing；摩擦重新按原 cfg/hand material 的原 max 组合计算。
- 桌面用原 NumPy-only ropesim/sdf.py、原 box_sdf、原边界/spacing/margin 重算完整数组。
- 直接改 entry 再重封，或同时改 embedded scene 再重封两层摘要，均不能改变上述来源。

重建前后持久化 owned 参数、p/v、全部 q（含 padding）及实际 clocks，逐项一致；slot/shape/source/solver-error/完整 episode identity 不符均拒绝。mutation 区域必须保持 _needs_reset is True 且 raw clock_valid is False；receipt 分别保留 before、mutation-after、最终有效 clock。异常后事务不可重用，rod clock 失效，禁止用部分 mesh 继续 step；只有新的完整 reset 成功才可能恢复，不声称回滚。finally 清 native/callback 别名。

## 有界 SDK clock

已安装 SimulationContext._physics_timer_callback_fn 累加 callback step_size（simulation_context.py 1360–1362，SHA 0be3b0eaa44e6110cec520be9ac2c08cea17ba97db14b1b0aa60d87824deceb3）；get_physics_dt 最终从 authored Hz 返回 1.0/physics_hz（SimulationManager 197–217，SHA 97d1ef88202f370eac00281b193581ba02b5c95ed64131f153c006e3ab104357）。

旧真实 SDK234 timestamp=0.4875000254251063，恰等于 234*float32(1/480)，原 reset 的 1e-9 容差会拒绝。新适配器仅在原 reset 调用的局部 scope 绑定 actual-clock 检查，finally 恢复旧函数；拒绝嵌套／未知外部 binding。

本包限定 0<=SDK step<=240。误差界：
n*abs(float32(actual_dt)-actual_dt)+8*eps64*max(1,n*actual_dt)
且必须小于 min(2e-7,actual_dt/2)。actual timestamp 原样保留，reset/rebuild 前后逐值不变，不投影为 n/480。SDK234 真实标量 fixture 通过；241 步与错误时间拒绝。不声称任意长度 PPO 或 5 秒以上 episode 的 reset 时钟合同已完整可用。

## CPU 验证

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_mesh_reset_order.py tests/test_dual_hand_reset.py tests/test_dual_hand_rod_runtime.py tests/test_dh116_dual_hand_episode.py
~~~

最终 **78 passed / 10.78 s**：25 新测试、53 原 reset/runtime/episode 回归。原 Python reset/runtime 实际执行，CUDA部分使用明确 CPU native-API state-machine doubles，不声称执行了真实 CUDA commit。实际执行了原 NPZ、原 CPU SDF sampler、72 源文件校验。

覆盖旧 deferred teleport 覆盖；实际 hand reset/kinematics 后 11 次 readd；第二 episode 在已有 solver step 后 reset；原 slot/table/material/capacity 语义；zero-force composer；部分 clear/add、slot、clock/episode/tick、参数/节点故障；typed last_error；多层重封；异常 traceback 引用；SDK234 时间和有界拒绝；clock binding 正常/异常恢复。

私有日志：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/mesh-reset-order-cpu-20261114-v1wvr4ha

- red.log：10 failed/1 passed（新 adapter 尚不存在）；SHA 8f6a90acbb031fc584b992c810a8d8142cac1529414605770ccac70ff7e77bbf。
- final-regression-v2.log：78 passed；SHA abac09525673216cad7db39b9ffe2430b12e59cdfe8df0d10781473a5057e78b。
- final-protection.log：528 受保护源文件＋2 历史 Checkpoint 通过；SHA 47b8ee1e5c42ba12b0cf04c953c0531f651b79d7a7a3084e86b21d4319b221b0。

clock identity/error-code/source 重封/mutation-valid 负例也分别先红再修；这些中间输出仅在本轮工具记录，不伪称有独立日志。

代码 SHA：94394cd6a10c528422bda28f7ee8813587ba1ca970dff54d9eabcc41ab19b50a。
测试 SHA：98c10211ecb436d98da12702d2e524a2867149e4fc79e7d8976c583cb160d610。

## 未取得的资格

无原生 current/commit 实测、无即时 pair-capacity readback、无重建后 contact-free／抓取／连续材料滑移／Teacher 资格。无新 recipe。后继集成须持久化 plan/reset receipt、冻结新源并审查生命周期；旧 projected Pinch 成功回执不能作为修复后证据。

独立只读审查：camera_v5 对最终代码 SHA 94394cd6a10c528422bda28f7ee8813587ba1ca970dff54d9eabcc41ab19b50a 的来源重建与 mutation clock 差异复核，确认已识别的两层重封和 raw-validity 隐藏路径关闭，未发现剩余已识别阻断。该审查未编辑、未重跑 78 项测试或原生，不扩大为全 SDK 行为证明。
