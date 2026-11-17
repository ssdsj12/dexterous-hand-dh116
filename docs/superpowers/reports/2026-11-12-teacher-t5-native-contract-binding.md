# 原 T5 原生物理合同的 Teacher R0 绑定

当前完整 T5 采集的 `tensor_contract.json` 与旧成功 T5 逐字节一致，SHA256 为 `eda0b7e316a493997192ffec9abd8b5648c826a265d817774dd9117c734ec551`。语义 SHA `cc7d9f525f6ebeb04ff97a2ad0ba9afac02767f897ff3f14558c33c1ae56a0d6`，冷评估物理 SHA `7128e10e341316dd76132eec5ee1d5af6f769691eecbe281d99a26b974c00612`。默认 Teacher128 原生合同语义为 `5bd521d7a2922ea52b1925b26a0546544b7abc399268af19a589507c100c8786`，冷评估物理为 `2bd771fb22f951dd0bdd8f52a0af55c01d24382ad5efe09ab19e22603a936566`。影响身份的差异包括二指主 reset SHA、同钟 batched native sensor readback 和原 T5 `support_deficit` 奖励配置。环境数、rollout、minibatch 及同值抬升基线重复项也有差异，现有合同规范只归一化已知重复批量字段。

新入口复用完整 T5 MPC 采集的 CPU preflight。源是实际历史启动 argv、原始资产与输入 SHA 及实测原生 371D 合同。它只从执行参数去掉旧 MPC/expert source、旁路采集选项及通用 launcher 提供的运行预算字段，保留 T5 的实物任务、配置、物理、控制约束和原生 reward 组成。R0 保持 `teacher_r0_existing` 的原生 reward 一次传递。原运行中的 `support_deficit` 随现有 T5 任务保留。Teacher 模型是新训练所有者，没有历史 MPC 接管。派生任务只写入本次私有输出目录，复用现有 `solid_cylinder` 和全部资产。实际 Teacher 回调会重新比较精确语义与物理 SHA 后才行动或训练，旧默认任务路径照常可用。

```bash
bash scripts/python.sh scripts/derive_dh116_privileged_teacher_t5_task.py \
  --baseline-plan "$reviewed_stage7_plan" --baseline-plan-sha256 "$plan_sha" \
  --t5-launch "$original_t5_launch" --t5-tensor-contract "$observed_t5_contract" \
  --output "$new_private_derivation"
```

BC 及冻结 manifest 准备阶段可用 `reviewed_t5_contract_binding.json`，Teacher 原生 R0 与冻结评估通过 `--native-task-config "$new_private_derivation/native_task.json"` 保持同一身份。预生成合同记录已观察的 T5 张量合同和派生策略身份，原生 R0 运行仍需回调校验。

真实 Ubuntu 只读试验用完整 T5 v01 实测合同和原启动 argv，核对 10 个输入，生成任务 SHA `56cef9f2f811918a070fefd9dc766ca9626410ee02b4a118e49773fba43e4aba`、合同绑定 SHA `7b9e9b91896ceb388c9b874b9aa4e6982959684a8dbe0e3e287155516d9bd84e`。私有目录 `/tmp/dh116-teacher-t5-actual-uZ4pPI`。Ubuntu CPU 回归 85 通过，日志 SHA `c4ff1069449345eec1c1989c8a066797fa32999b14d938dd28c018bf40f397e5`。本包未启动 Isaac，真实 R0 策略控制及抓取成功待原生证据。
