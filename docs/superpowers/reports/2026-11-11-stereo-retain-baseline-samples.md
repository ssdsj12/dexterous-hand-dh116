# WP1 保留历史原色采样槽

基线 `3c80166da75fb5496701c9f3ff673296c5db5615`。`--retain-baseline-samples` 新增独立的离线对照。完整保留原色轮廓的采样像素、XYZ、有效性、分量、弧长、置信度和轨迹，包括可见但缺深度的 2D 槽。候选沿用原色拓扑和左右 RGB 支持的软补测，去掉重复像素及原色未观察到的分量。候选只占原来空闲的表面槽，不移动任何已有槽，也不填补中心线深度洞。

档案分别保存原始池、候选池和试验布局。候选来源区分未被原版采样的已有稠密深度，以及新的 RGB 软补测深度。5 mm 同像素三维判据在采样后应用，仅作为离线标签。无参考为 NA，不作为真遮挡。该标签未参与 matcher 或采样。现有历史数据缺右原生 ID，所有新增点仍留在候选池，部署晋级数为 0。

## Ubuntu CPU 证据

现有 Python 3.10.12 上 45 项 CPU 测试通过。Front 和 Roll20 各四帧、每帧两个环境的只读复放均退出 0。逐份核对八个帧归档 SHA，对全部原可见槽核对完整逐点字段，数值精确一致。固定分母和 5 mm 判据保持一致。

| 配置 / 抓法 | 原正确点 | 保留布局正确点 | 固定分母 | 独立候选池 / ≤5 mm | 建议新槽 / ≤5 mm |
|---|---:|---:|---:|---:|---:|
| Front 二指 | 6 | 14 | 2128 | 67 / 8 | 67 / 8 |
| Front 三指 | 0 | 5 | 2165 | 43 / 5 | 43 / 5 |
| Roll20 二指 | 91 | 91 | 2009 | 87 / 64 | 0 / 0 |
| Roll20 三指 | 89 | 89 | 2054 | 84 / 52 | 0 / 0 |

以上数据每帧一致。Roll20 原版的 96 个表面槽已满，试验布局未挤出原槽。Roll20 二指候选来自 54 个已有稠密深度和 33 个软补测深度，后者 11 个通过 5 mm；三指分别为 50、34 和 4。原固定 12 处 × 4 帧仍全部超过 5 mm，保留布局消除了采样替换引起的基线退化，匹配尾部仍未解决。

```sh
export PYTHONPATH="$PWD:$PWD/src" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_wire_sample_retention.py tests/test_dh116_stereo_wire_correspondence.py tests/test_dh116_stereo_wire_depth.py tests/test_dh116_stereo_wire_segmentation.py tests/test_dh116_stereo_wire_tail_audit.py tests/test_dh116_stereo_perception.py
/mnt/mydisk/xk/envs/dh116/bin/python scripts/audit_dh116_wire_correspondence.py --proof <historical-proof> --segmentation-checkpoint <balanced200-checkpoint> --segmentation-sha256 7c9c3a60945e17a8363144d6ceebbd9cb18d676240582d2ac1f0428dc9333920 --retain-baseline-samples --colour-anchor-fallback --output <new-private-output>
```

Roll20 追加 `--tail-fixture configs/dh116_training/stereo_wire_correspondence_roll20_tail_v1.json`。初次复放在旧临时源码副本缺少已存在的几何模块时退出，日志原样保留。最终在 Ubuntu 工作树使用当前完整源码复放；未启动 Isaac、采集或训练。

下一步需要同身份左右原生 ID 和光轴 Z 证据，继续复核候选和固定尾部。原色仍为默认；表面采样不提供材料节点或物理接触资格。

复放摘要 SHA256 为 Front `3918698a06146363427ef1f070b9befb73e812a33799c61d0801fc08d165b413`、Roll20 `c389bed11e2ac652f58533abe9da7cbe0d00bb12e519a17ad06dc291717655cb`。 私有归档目录为 `/tmp/dh116-wire-retention-lf-g4HBlj`。
