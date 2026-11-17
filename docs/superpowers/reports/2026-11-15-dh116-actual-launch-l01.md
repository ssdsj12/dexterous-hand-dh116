# DH116 Actual Native Launch L01

基准 `ce5af9bb650627a252e40c7c397c8efca1d25555`。本包仅实现已批准 Actual Native Provider 计划 L01.1–L01.3：CPU 可验证的真实进程启动、独立 actual companion 身份、parent-issued 租约交接及 worker 入口。L02 provider/scene/artifacts、L03 A→B 恢复未实现；没有执行 App、Isaac、GPU、原生训练或抓取。

## 实现边界

- 保持原 v1/v2 recipe validator、C06 `WorkerLease` 本人 PID/`/proc/locks` 校验及原 PPO 不变。没有 companion 的批准、command/code SHA 和 attempt 目录算法保持原值。actual 单独组合 core identity、companion SHA 和同一完整 source closure SHA，plan/register/run/receipt dispatch 使用同一 attempt fingerprint；旧 core-only 批准不能批准 actual。
- companion 固定 `a_fresh`、物理 GPU3、实际 mask `3`、唯一逻辑 `cuda:0`、原 `gdb_on_segv` 和唯一 provider path/`run_a`。预算 GPU 不匹配、多 mask、logical3、未知键、bool、旧批准和 `b_resume` 均拒绝。原 scheduler 已覆盖 `CUDA_VISIBLE_DEVICES=str(gpu_index)`，本包没有将其描述为空 mask 修复。
- 新 worker 仅接原 C06 config/SHA/additional1/output argv。绑定 SHA 和规范 attempt 路径恢复父调度器的外部 evidence root，随后重读真实 registry/intent/approval/source/HEAD 和父进程身份；路径只定位仓库私有状态，不能代替授权。真实 CLI 通过 canonical module alias 保持 C06 exact `Repository` 类型，不放宽原类型门。
- entry 在锁外做静态校验，然后本人进入原 `native_instance_lock`，同一个上下文覆盖唯一 request、完整 provider 调用及自然清理。request 不自行短暂拿锁后释放，lease.close 不释放 FLOCK。
- parent 必须是实际 scheduler native-worker argv/cwd/interpreter，且本人持本轮 worker.lock 和原 scheduler reservation；parent 不代持生产 singleton lock。AF_UNIX `SOCK_SEQPACKET` 只 accept 一次，kernel `SO_PEERCRED` 及 `/proc` 验当前 Popen/inferior 后代、同 UID/boot、稳定 starttime、exe/argv/source 和本人 FLOCK 的 dev/inode；拒自签发、parent/GDB 代持、重放和过期。
- overlay 只由 server 重读当前已批准 intent 后 exclusive/O_NOFOLLOW/fsync 写入本轮目录，原 C01 authorization/resource schema 保留，owner 是真实 worker。原五键预算与 core/source/semantic、nonce、角色、parent kernel peer 和原始 deadline 绑定；握手消耗预算，不在 grant 时重启计时。
- grant 后连接和 listener 关闭。ActualWorkerLease 公开只读 config/role/resume_origin/device_plan/launch_identity/holder_pid；revalidate 只核原 core lease、同 SHA session 文件和首次 kernel peer 对应的当前 `/proc` 身份，不重连；拒 pickle。
- 实际设备 probe 在 App 前以安装的 nvidia-smi 物理 GPU3 PCI bus ID 对照 CUDA driver 实际 logical0 PCI bus ID 和唯一 visible count。不猜 torch 属性或自动换卡。本包仅做 CPU 输入正反测试，没有调用 GPU probe。
- `_verified_actual_native_receipt` 和 actual 子进程成功路径明确拒绝 `actual_artifacts_not_implemented`，B 明确拒绝 `actual_resume_not_implemented`，不调用旧 A-only v2 artifact auditor 冒充 actual。

## 必须保留的实现澄清

生产 provider 源码尚不存在，必须在完整 source identity、intent 和 grant 前以 `actual_provider_not_implemented` 拒绝，不提交空 provider。CPU 测试仅在临时 git Repository 的固定受审 path 写明确 fixture 源码并绑定 SHA，永不获得原生资格。该早期失败是 L01 缺 provider 非零边界，不要求先签有效 lease。

旧 v2 inputs 仅允许 training config，本包未扩大它。actual core 的 `source_files` 必须包含真实 `configs/dh116_execution/actual-environment-v1.json` 文件，格式为：

```json
{"schema":"dh116-actual-native-environment-v1","experience":{"path":"/absolute/actual/experience.kit","sha256":"<actual SHA256>"},"debugger":{"path":"/usr/bin/gdb","sha256":"<installed GDB SHA256>"}}
```

它的实际 bytes 纳同一 core inventory；experience 还须与 companion 及实际文件 SHA 一致，所以改变 experience 会诚实改变 core。companion 本身不进入 core，避免自 hash 循环。生产代码的本地 AST imports、各级 package initializer、固定 provider 及两份 actual 测试必须在 core 闭包中；worker 还拒未声明的已加载本地生产模块。

Linux AF_UNIX pathname 上限 108 bytes，真实 private attempt 路径过长。因此 metadata/overlay 保留在本轮 attempt/actual-launch 的 0700 目录，socket 使用随机短 `/tmp/dh116-al-*` 0700 目录和 0600 socket；nonce/parent/规范路径受 envelope 绑定，grant 后无重连，原 child monitor 自然结束后清理 socket 目录。没有新 daemon、第二套 scheduler 或 kill/os._exit 伪自然退出。

## CPU 验证与 TDD

所有日志私存 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/actual-launch-l01-20261115`，不提交原始日志。测试使用 CPU fixture/private locks，未触生产 lock。环境：`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES= DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1`。

完整命令：

```bash
bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_native_teacher_actual_launch.py tests/test_native_teacher_actual_launch_process.py tests/test_dh116_work_packages.py tests/test_native_teacher_worker_contract.py tests/test_native_teacher_worker_lease.py tests/test_native_teacher_worker_semantic.py tests/test_native_teacher_worker_artifacts.py tests/test_native_teacher_worker_checkpoint.py
```

最终完整结果 **322 passed / 0 failed / 0 deselected**，312.50s；其中旧组 269 项、新 actual 两组 53 项。`l01-final-green.log` SHA `7ebf5ffd61dd3587e0c24071469c56040bdf90b2e6d02fa44d6c382868f669e5`。

真实 bash CLI subprocess 测试完善后，新 actual 两组再完整运行 **53 passed / 0 failed / 0 deselected**，32.09s；`l01-final-launch-green.log` SHA `43208be74973903ccb5cb1806b1d3b776cca52939d7b3b9bdee0d36b5c79abd0`。单独 real CLI 测试 **1 passed / 27 deselected**，`real-cli-green.log` SHA `95d9e6c280bb1fba929ec294f4497db84e2f039748024ccaba1ba477c27b4646`。最后这次调整仅完善测试的实际 bash CLI 覆盖，生产源码与完整 322 项运行时相同。

已记录的 RED→GREEN：

| 场景 | RED | 后续 GREEN |
|---|---:|---:|
| 首轮 companion/identity 缺失 | 17 failed | 139 passed，含旧相关组 |
| experience 只改 companion | 1 failed / 19 passed | 20 passed |
| provider 缺失须明确失败 | 1 failed | 20 passed |
| 动态 loaded source 漏闭包 | 1 failed | 21 passed |
| scheduler cleanup/device wrapper | 2 failed | 23 passed |
| driver logical PCI 与 physical3 不同 | 2 failed | 24 passed |
| package initializer 漏闭包 | 1 failed | 25 passed |
| 初轮真实 process/peer API 缺失 | 4 failed | 25 passed 最终 process 组 |
| 非 scheduler 自签发 | 1 failed | 25 passed 最终 process 组 |
| 预审真实 entry E/canonical 类型/只读属性 | 4 failed | 52 passed |
| actual 测试源码漏闭包 | 1 failed | 最终 322 passed / 新组 53 passed |

初始 `companion-red.log` SHA `c355ca9cae02822f1b06f9d0c6c8e9f1d7e966aeb3141e287ece3463615b18d6`；`process-red.log` SHA `25f2bbe2303e16d2eb0b30ad31e4019cc1bc4170771eb154218a11f5f95b1114`；`process-self-issuer-red.log` SHA `68e7ef4c78737c569f5a45d960d05f6fb40ddb4b0d581971d13af34a6fccc1b2`；`entry-review-red.log` SHA `33b5545624e93b0eb589c965e51805ef77765f6dc2aeafa2a1b76e651cb54f82`。

真实三进程 CPU direct、原 GDB inferior、parent-held lock 负例及 peer/nonce/inode/expiry/intent/symlink/session 文件/source/锁变化均验证；全部 fixture 标为 native false。另以生产 scheduler/python.sh 同 bytes 的临时 git 仓库实际执行 `bash scripts/python.sh scripts/run_dh116_work_packages.py --plan --evidence-dir <private>`，以及脚本 canonical 类型诊断，未只依赖 import main。

`Repository.verify_protection()` 实读通过：528 protected files + 2 checkpoints；导入 actual launch/process 后无 pxr/Isaac/omni 模块。`protection-final.log` SHA `c67183e6c91c877c2d5c51161f2215bda4c35ff3a955e4ca58879c6e7fa8e3ef`。`git diff --check` 通过。共享 registry/readiness/README/CHANGELOG 交 root 集成，本包未改，Ubuntu 未 push。

## 后继

L01 仅证明可信启动/租约接线软件及 CPU 失败边界。生产 actual 冻结仍等待 L02 provider 源码、真实 scene/source/model 同 snapshot 与 artifacts；L03 actual A receipt→独立 B resume 仍拒绝。实际 A/B 原生预算、自然退出、抓取与 Teacher/Student 准入均未声明成功。主目录保留 manifest 与 BackendProfile source 移植问题由 root 后继处理，本包未绕过 SHA 或覆盖用户文件。
