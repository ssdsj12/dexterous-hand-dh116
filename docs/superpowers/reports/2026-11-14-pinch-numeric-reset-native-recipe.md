# Pinch 网格 Reset 数值读回和原生复验

上一配方只完成 2 个 SDK 初始化步，物理采样和 PPO 更新均为 0。进程耗时 75.935 秒并返回 1，GDB inferior 同样返回 1，无 signal，`app.close()` 已返回。原生回执 SHA256 为 `6c97bc0cdb03c71f74c7f5a3c0ec350a9c52f64bf622451bf896737d73025027`，失败的 Reset 产物 SHA256 为 `e2f5302ed2dabbb3916b438fb7e3c74bfaa6f7c96e1b4ab5133fdf6794983e41`。栈和库映射仍随失败产物保留。

原 Reset 比较器对仅提供 CUDA array interface 的视图调用了 `np.array()`。复现表明三个视图均生成标量对象数组。内容相同而包装对象不同即可产生不同的字节哈希。该失败没有读回数值前后差，实际杆状态是否改变仍待原生复验。

新入口在同一 Reset 事务的重建前后，使用已有 CUDA tensor 复制路径读取实际位置、速度和 XYZW 四元数。两次采样均以实际 SDK/solver 时钟和同一 owner 对齐，数值来自独立 CPU 副本。保持原严格字节比较，任何数值变化仍触发失败。新 worker 将数值快照、逐字段差异和原错误写入 `mesh_reset_numeric_state.json`，由外层 worker 在自然退出后与 Reset 证据交叉验证。

本轮配方使用 GPU 3、单 Isaac、1 环境、2 个初始化步和 232 个测量步、180 秒、0 次 PPO 更新。单次预算绑定如下。旧原生配方的指纹仍是 `4918ab0a9bb51fc20c0bbcc0bd070fafc7c846dac9a4579a170b1af30cd34601`。

| 指纹 | SHA256 |
| --- | --- |
| 新命令配置 | `54871364c9f86812dd2b0eaebf2462355bca36614c1cf1572f9eec19afc43f93` |
| 新 scheduler 代码及配置 | `c48203afab2a7cb17ca2aaa672a4096ea380c20f85d1146d414a117930d84247` |
| 新物理入口配置 | `1a599dae88cb92c3858332f929b95b0d9b02587e9e321721ebc5b6150503aea8` |

Ubuntu CPU 验证通过 50 项。证据 SHA256 为 `77c36834030f05bcc7777792aedbcfc959494dbc15ab7e4954ccd4fea64d6ffe`。

```bash
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_mesh_reset_numeric_worker.py tests/test_mesh_reset_numeric_state.py \
  tests/test_mesh_reset_health_worker.py
bash scripts/python.sh scripts/run_dh116_mesh_reset_numeric_pinch_work_packages.py \
  --run --scope all --limit 1 --jobs 1 \
  --evidence-dir "$DH116_PRIVATE_EVIDENCE" \
  --native-approval "$DH116_FROZEN_BATCH_APPROVAL"
```

原生结果单独发布。Teacher PPO 准入仍需实际轨迹、R0 和对应抓法的独立证据。
