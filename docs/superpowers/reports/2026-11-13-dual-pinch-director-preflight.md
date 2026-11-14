# 真实材料方向端点读回：独立18步原生配方准备

基线 577a51dc15ea162dd5208787cb0edb4c6abe49f9。新增八个配置/执行/校验/测试文件，复用已发布的 MaterialDirectorReadback 和已通过18步 closure 原路径；不修改旧配方、原物理、材料、关节/腕部限制、24D合同或历史数据。本报告为执行前代码和CPU证据，原生0步、PPO0更新；不是材料滑移或Teacher资格。

## 实际修改

- prepare_dh116_dual_pinch_director_interval.py：在原 build_reset_owned 完整实际reset后，以原rod owner读取一个reset端点；在原 ClosureExchange 每次完整SDK+12 solver子步后附加40真实材料段的XYZW四元数及41节点。单独保存第41个padding，不当作额外材料段。
- 原 builder、episode、实际动作提交、几何 guard、clock journal 和正常资源释放均复用，作用域绑定结束后恢复原模块。原物理终止原因优先于新读回失败。读回失败保留同一个已完成物理行及明确NA/error，下一步不继续。reset读回失败不允许action。
- verify_dh116_dual_pinch_director_interval.py：只接受原closure真实调度回执SHA dd64cce38312cb3fe044e210c8560f44fae33b218e5b8e2ba97435a3969c9bc0，逐字节核对其独立postexit、physical/result/clock，禁止编造依赖输出路径。核对锁定CUDA源、扩展和真实source_basis。
- run_dh116_dual_pinch_director_work_packages.py：独立worker重入标记、GDB、单实例锁、真实依赖receipt、一次指纹门和postexit校验；普通脚本直接执行不得绕过这些门。
- validate_dh116_dual_pinch_director_interval.py：完整复用原外层exit、真实clock、prepared geometry、资源清理和GDB门；独立验证一个reset+十六个端点，同SDK/solver时钟、41节点、40 rest segments、单位四元数、同owner/material身份。新增reader弱引用必须在STOP后释放；失败前缀保留，不用对象删除代替自然exit。
- CPU tests运行真实Python builder/episode/owner控制流程，但SDK/CUDA是显式测试替身。完整producer账本重放的外层和GDB字段是明确合成fixture，不计入原生验证。旧closure原生账本缺director，必须被新validator拒绝。

MaterialDirectorReadback只使用真实既有RopeBatch.positions/quaternions，默认CUDA流明确synchronize后复制到owned CPU数组。它不step、不gather、不reset、不setter。Rest lengths来自constructor float32 nodes差分；RopeBatch没有rest getter，因此 native_rest_lengths_readback_m保持None。端点四元数不是12子步接触材料历史：continuous_contact_s、tangential_slip、physical_error_bound、手桌无接触资格、Teacher准入全部保持未知或false。

## 冻结预算与身份

1环境、GPU3、单Isaac、0 PPO；总18个SDK物理步=2初始化+8原零动作+8原finger-only动作。180秒上限，每个代码/配置指纹最多一次；禁止重跑已消耗旧closure。实际scope只验证端点读回、时钟和正常生命周期。旧finger-only24D、源profile 67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498 完全保留。

| 身份 | SHA256 |
|---|---|
| command spec | 4aa87632c42e273c3c52a8abb9e20cfa5e9f10ffc1b0623656b0b514c5d89dea |
| scheduler code/config | abfae81f2cc9bfad7cb1698ad88afd5f9aeb8a89f87861fbd436c6defc16cf74 |
| worker source/config | cefdcc8b7c84c7b67bd0e142eba80acd511ce2632540e9abe8b8a7b980f34796 |
| CPU log | cc5671802e778713528810d82acf4c87a5a11bc6282f788b4ae0809a13adb0b7 |
| 被保留的首次接手失败日志 | eb9a7e673042b3bcaa46c180f22460632883e35d2e04969b2260098da8919f88 |

配置只有完成代码审查、已发布同一commit、资源锁可用及本轮standing授权绑定真实指纹后才可执行。旧实验批准不当作重复执行许可。

## 回归和保留的失败

最终隔离worktree回归 **110 passed / 0 failed / 0 skipped，273.64s**：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_pinch_director_interval.py tests/test_dh116_dual_pinch_director_postexit.py tests/test_dh116_dual_pinch_director_worker.py tests/test_dh116_material_director_readback.py
~~~

证据：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/director-final-release-cpu-krCAzd，tests.log / tests.xml。528保护源码和2历史Checkpoint校验通过。

首次接手未整合完的两工作区运行 **122 passed / 3 failed，382.43s**。两例测试只替换sys.modules而未同步package属性，合并执行时读取旧module属性；一例配方在最后fixture更新后未重新冻结，代码指纹失配。失败保存在 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/director-takeover-cpu-O3bBfA。采用子代理已写好的完整postexit修正（含真实bool clock、constructor seed与reader释放门），同步package属性测试替身，并在最终源码上重新计算配方。第二次整合回归108通过/1失败：旧CPU producer保留Sim.physics_sim_view，使两个asset_sim_view在STOP后仍alive，新validator正确拒绝。未修改旧记录或真实释放门；新CPU替身明确模拟SDK清理共享view，并每个测试会话重新生成producer，取消对旧有泄漏fixture的正例依赖，增加共享view存活拒绝回归。该失败日志保存在director-final-cpu-QfH1hU。没有通过删除原断言或降低门槛消除失败。此前worker两个缺失依赖receipt的CPU fixture失败也保留于director-worker-cpu-y8qcpwfa；仅补测试registry依赖替身，实际依赖loader仍严格读取真实原生字节。

## 后继限制

实际CUDA API、同钟17个端点、自然exit必须由新配方确认；本包没有原生结果。即使通过，也不证明连续接触材料锚、真实切向滑移、双手有效承托、R0生产总奖励或任何抓法成功。保持30mm / 5s / 0.5mm，Teacher ready仍0/3；后继优先处理拇指接触、真实R0来源和分抓法可达性。

## 文件指纹

- configs/dh116_execution/deformx_dual_pinch_director_endpoints_v1.json: 4aa87632c42e273c3c52a8abb9e20cfa5e9f10ffc1b0623656b0b514c5d89dea
- scripts/verify_dh116_dual_pinch_director_interval.py: cb5a4f5b38a89be7ad5ac51f2a1819b6dc70956aa4c6c7cb3ac52152091851fd
- scripts/run_dh116_dual_pinch_director_work_packages.py: 868c92564f73b8c603a4093048dd29fb747bc0f9cf9e4e05f55760205d1b3673
- scripts/prepare_dh116_dual_pinch_director_interval.py: c3cf6ccbfb0df006cedac0aa49d6f5f2ee46b16f718f17c4010171b597c87a28
- scripts/validate_dh116_dual_pinch_director_interval.py: f5fbbf045936ec042a989613ed80e606dc89b19c1419ab629517766bc00ac3d1
- tests/test_dh116_dual_pinch_director_interval.py: e588bdee9863a642e6cf6e8b78438ef5ace1020d54e041c749921e83897869c9
- tests/test_dh116_dual_pinch_director_postexit.py: 37daf05d89d9b7e28372cc11a5959000a20af46eda676b69e5415f982bf37f39
- tests/test_dh116_dual_pinch_director_worker.py: 55e48555a5a483c809b45a4e424b5f6a8e5b244e2277b7267cecde61c3c2af87
