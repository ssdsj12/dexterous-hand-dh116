# WP2 状态估计训练模式与证据准入

基线 `3c80166da75fb5496701c9f3ff673296c5db5615`。新增两套显式配置，复用既有 371D/24D 合同、已验证标签筛选和按物理初态及来源族的连通分组。旧 `formal_training` CLI 与报告状态继续兼容。

| 模式 | 执行路径与准入 |
| --- | --- |
| `schema_test` | 核验张量和归档合同，优化关闭。 |
| `diagnostic_fit` | 只用已验证的 Center XYZ 标签进行训练内诊断，冷启动小样本仍只能进入诊断。 |
| `offline_qualified_training` | 先重新读取绑定 SHA256 的 manifest、候选与物理账本。按物理初态和来源族划分后，只有各分区都有已验证标签的 Head 才参与优化。 |
| `native_qualified_training` | 在离线门槛上，另行重开已完成原生 Stereo proof。校验原始测量、退出证据、proof SHA、同一标定、原深度匹配和完整预处理合同。核验后的数据仍由离线估计器优化。 |

缺少独立原生 proof、只有资格声明、SHA 不匹配或预处理不同均按具体原因阻断。命名模式不接受未由 manifest 读回的临时内存行。成功输出仍不授予 Actor 部署、泛化、BC 成功池或原生抓取资格。没有合格标签的字段保持掩码，Force Proxy 沿用已有真实力学证据门槛。

Ubuntu 使用 `/mnt/mydisk/xk/envs/dh116/bin/python` 和 `PYTHONPATH=src` 运行：

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_state_multicohort.py tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_cold_candidate_admission.py
python scripts/train_dh116_stereo_state.py --dataset-manifest <path> --dataset-manifest-sha256 <sha256> --mode <mode> --device cpu --output <new-output>
```

专项 **49 项通过，0 失败，17.78 秒**。扩大回归另加 `test_dh116_stereo_estimator.py`、`test_dh116_stereo_runtime_task_state.py`、`test_dh116_stereo_runtime_actor_modes.py`、`test_dh116_stereo_training_cli.py`，**105 项通过、1 项失败，24.92 秒**。失败项是 `tests/test_dh116_stereo_estimator.py::test_versioned_configs_bind_canonical_action_and_exact_state_contract`。从基线 Git 直接读出的 LF 合同 SHA 为 `9f145cf8e68b1dfe8e2c7464ec5fc0cb00e81a2707115c2fe148ab49d77bbb62`。将同一基线字节转为 CRLF 后得到既有配置固定值 `a7af9f3f214a646f2f95dce45aa4744f378488de34237431c914bd4137b2ee86`。工作区合同与 Git 基线逐字节相同。本次保留合同和旧 pin，待单独解决 Ubuntu 换行兼容。完整日志留在 `/tmp/dh116-wp2-ubuntu-final-6FvG9m`。

使用原有 manifest SHA `7c1a80bfe77b14a662c00d082d475e0db789b6ee103379c7375b677e1023f514` 在 Ubuntu 读回历史证据。8 行、2 个独立初态、6 行合格 Center XYZ。结果为：

| 运行 | 退出码 | 状态 |
| --- | ---: | --- |
| schema | 0 | schema_validated |
| offline qualified | 2 | blocked_insufficient_independent_groups |
| native qualified | 2 | blocked_missing_native_stereo_proof |
| 旧 formal | 2 | blocked_insufficient_independent_groups |

历史读回未开始优化、未写出模型。日志与报告保存在 `/tmp/dh116-wp2-history-ZAqQne`。测试中的合成 proof 回调仅验证准入控制流程。本轮没有执行 Isaac、采集、真实数据优化或原生资格验证。

下一步先解决 Ubuntu 合同换行兼容，再等待独立初态数据和与训练配置完全一致的原生 Stereo 证明。
