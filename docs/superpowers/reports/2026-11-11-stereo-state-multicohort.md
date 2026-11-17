# 双目状态监督多来源准入与分组验证

本工作包基于远端 `240be7ccef37f2cbce705475692716de6f10ef5f`，扩展现有状态监督入口。正式状态训练仍为 `blocked_insufficient_independent_groups`。

`read_candidate_dataset()` 读取带 SHA256 的 `dh116-stereo-state-dataset-v1` 清单，按原有候选准入逐个核验 RGB 历史、371D 特权状态及逐维有效性、24D Teacher mean 和实际发出的动作、物理初态读回与 480Hz 原始账本。每个样本的相机与特权身份必须一致，包含 frame、reset、physics tick、时间戳、标定和物理初态 SHA。归档的 RGB 处理合同必须与相机标定、分辨率和视差有效性匹配。用于标签的原生参考另行校验同帧身份和 SHA，输入保留归档中由同步左右 RGB 计算的深度及有效性。读回核对归档证据及字段一致性，未重新运行旧匹配器逐像素核对输出。

同一数据集使用同一标定、预处理合同和历史形状。重复文件与跨来源的重复采集 run 会被拒绝。跨 run 复用同一物理初态或同一 source episode family 时，所有相连帧进入同一个分组，改变输入顺序不会改变给定 seed 的 train/validation/test 分配。单源 CLI 参数继续可用。正式模式按 Center XYZ、Wire Geometry、Velocity、Deformation、Contact、Slip、Force Proxy 评估并训练各阶段，只有三个分区均有合格标签的阶段才运行优化。Center XYZ 独立使用原合同的 61–63 维及 `relative_center / 0.05 m`。原生参考、现有 Actor 和成功专家池均沿用各自的准入门槛。

清单中的相对路径从清单目录解析。实际清单保留在证据目录。

```json
{
  "version": "dh116-stereo-state-dataset-v1",
  "sources": [
    {
      "candidate": "cohort-a/stereo_teacher_candidates.pt",
      "candidate_sha256": "<candidate SHA256>",
      "raw": "cohort-a/native/native_physics.jsonl",
      "raw_sha256": "<raw ledger SHA256>"
    }
  ]
}
```

```text
python scripts/train_dh116_stereo_state.py --dataset-manifest <dataset.json> --dataset-manifest-sha256 <manifest SHA256> --mode formal_training --device cpu --output <exclusive-new-output>
```

报告和生成的状态模型绑定清单及候选、账本、Teacher Checkpoint、标定和预处理 SHA。模型只有优化实际启动才会保存。

## 本轮验证

- Windows CPU 回归 **99 passed，21.63s，退出码 0**。覆盖跨文件初态及来源别名、顺序不变、重复 run、时钟/参考/视差/标定/处理合同错误、CLI 正式训练阻断和分阶段标签选择。
- 服务器 CPU 用新入口读取历史原生归档。候选 SHA `4da4f234db2ff77c505d100366bcf30b5c9b2d007f48cfd5b132c1e5a0f8cd70`，原始账本 SHA `da81d600f58864b059cde81131353ce6384be5e84f53bc6446a94bb7b00c6929`。8 行来自 2 个独立物理初态。6 行获得 Center XYZ 标签，Wire Geometry、Velocity、Deformation、Contact、Slip、Force Proxy 均为 0 行。正式模式按预期退出 2，无新模型。报告 SHA `cc7eb71f53a48dad28683f28ba1c2816421644f1a9dedd4592e16d4988d4a92c`，实际执行的监督源码 SHA `989764323f75c435106760d1fd9afc4f95bbaafb31bc80a15e475dc72b253a11`。
- 本轮实际多来源行为由 CPU fixture 验证。现存原生归档只提供一个同步来源，因此没有新的跨来源原生泛化证据。没有启动 Isaac、训练或采集。

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_state_multicohort.py tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_cold_candidate_admission.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_runtime_task_state.py tests/test_dh116_stereo_runtime_actor_modes.py tests/test_dh116_stereo_training_cli.py
```

下一步是按独立批准的采集方案取得更多同步物理初态，并审计三组标签覆盖和独立性。线缆几何、形变、接触、滑移和 Force Proxy 仍需各自的合格物理来源。
