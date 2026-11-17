# 双目 RGB 光轴深度辅助头 CPU 诊断

基线 `3e50850`。WP1 的辅助头以现有 `StereoEncoder` 左右视图 cross-attention token 为输入，两个独立线性输出分别预测左、右相机的光轴 Z。它支持复用 `StereoStateEstimator.encoder`。默认估计器和旧 Checkpoint 的参数结构按原路径加载。

`StereoDepthAuxiliary.forward` 只接收左右 RGB 和视图有效性。输出以米计，限制在指定有效深度范围，并携带 stride-32 像素索引及双视图有效性。参考只经 `native_depth_targets` 进入 `depth_auxiliary_loss`。标签绑定帧、仿真时刻和非零标定 SHA，在对应像素直接取样。缺视图、缺参考、无效像素和超出范围的点均屏蔽。右图参考需要独立解析的 Native Instance ID 和对应的有效性张量。没有合格右源时，右输出没有监督梯度。辅助头的输出是粗网格预测，既没有线缆 ROI 精度资格，也没有几何替代资格。

## Ubuntu 复现

```bash
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python=/mnt/mydisk/xk/envs/dh116/bin/python
out=$(mktemp -d /tmp/dh116-wp1-depthaux-XXXXXX)
"$python" -m pytest -q -p no:cacheprovider \
  tests/test_dh116_stereo_depth_auxiliary.py tests/test_dh116_stereo_estimator.py \
  tests/test_dh116_stereo_wire_segmentation.py tests/test_dh116_stereo_confidence_audit.py
"$python" scripts/diagnose_dh116_stereo_depth_auxiliary.py \
  --proof /tmp/dh116-stereo-20261110T120000Z/native_roll20_block5/640x480/native_stereo_proof_completed.json \
  --proof-sha256 fda6bb67d0d9a141baa2352a7b9b8a4cc4cd7f3e90a7aae8ec411f1269de4172 \
  --output "$out/roll20.json"
```

## 结果

- TDD RED 在模块创建前复现 `ModuleNotFoundError`。Ubuntu CPU 回归 65 项通过，其中 8 项为本工作包新增。覆盖缺失视图、错误时间戳、全零 SHA、禁止参考作为输入、右图独立来源、深度范围和同一共享编码器的左右梯度。
- Roll20 离线复放读取 640×480 已归档原生帧 21，仿真时刻 0.070833337 s，二指和三指各 1 个环境。输出 15×20 网格，每组 132 个左图监督点，其中 2 个落在原生线缆 ID 区域。历史存档无右图 Native ID 和有效性，右图监督点为 0。
- 未训练权重的一次 SmoothL1 损失为 0.137148 m。梯度 L2 为左 RGB 0.135595、右 RGB 0.009849、共享骨干首参数 0.600632、左输出权重 0.995583、右输出权重 0。左图有效监督经双目特征传播到两张 RGB，右输出在无右图标签时未被训练。
- 全场景抽样的二指/三指 MAE 为 141.49/142.70 mm，RMSE 为 174.68/173.45 mm，P95 为 351.56/342.78 mm。这些未训练数值仅用于验证数据和损失路径。原生物理步、优化器更新和 Checkpoint 写入均为 0。
- 绑定的 RGB 存档 SHA 为 `733952a7aa7495b08e98b98d3e90e38266350f79630892f27f6b90a19c793235`，参考 SHA 为 `ab549fd428e0409889f4bf626afb5651bcf24dfd76ef5af2634749be4f2ea8ef`，标定 SHA 为 `02fabb87c548cd982728dac5d6357ca08327258514269b163e880427b2556700`。通过 528 个保护文件和两个历史 Teacher Checkpoint 的 SHA 校验。

旧 Checkpoint、运行配置、相机默认配置、固定 5 mm 参考、Roll20 已验证候选池以及运行时策略均未修改。正式训练需要独立分组的同步左右原生参考及粗网格覆盖审计，线缆深度误差与三维完整率继续使用现有固定标准。
