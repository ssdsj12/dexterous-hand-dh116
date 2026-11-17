# F4 核验后的原始指腹凸包 SDF 与显式注册

本次仅使用 CPU／GPU PhysX parity 通过的 GPU 原生 collision owner 凸包。锁文件验证 GPU 摘要、10 份原生指腹记录、parity JSON、两次进程与对比退出码，以及 F1 编译包的扩展、bridge 和 SDF 源文件 SHA256。10 个指腹均在同一物理 tick 读回，沿用原始构成中重合的 owner／Mesh／link 局部米制坐标和同 tick SDK 世界位姿。SDK 四元数按原 xyzw 合同传递，并核对与读回矩阵一致。

官方 `weld()` 后 10 个凸包均闭合，调用官方 `bake_body()`，使用 1 mm voxel、12 mm margin、0 round、0 thickness。独立注册设定的线缆半径为 5 mm，每半径 5 个 voxel。margin 表示采样域向几何外侧延伸。上游接口没有可配置的 TSDF 截断距离，输出记为 `null`。实际在 RTX 4090 SM89 的 Warp `cuda:0` 运行。现有 Warp 1.12.1 运行日志报告自带 CUDA Toolkit 12.9，固定 `ropesim` 扩展仍是 F1 用 CUDA 12.8 构建的同一文件。

| 指腹 | 左网格维度 | 右网格维度 |
|---|---|---|
| 拇指 | 45 × 39 × 50 | 45 × 51 × 40 |
| 食指 | 40 × 43 × 63 | 40 × 43 × 63 |
| 中指 | 40 × 43 × 63 | 40 × 43 × 63 |
| 无名指 | 40 × 43 × 63 | 40 × 43 × 63 |
| 小指 | 40 × 41 × 53 | 40 × 41 × 53 |

共 1,003,550 个 float32 格点。10 个凸包中心 SDF 均为负（−5.561 至 −4.130 mm），外部探针均为正且大于杆半径（7.747 至 7.877 mm）。六面格点距凸包最小 **12.000 mm**，凸包顶点插值绝对误差最大 **0.388 mm**。逐指保存局部原点和边界、距离极值、原生行 SHA、网格文件 SHA 和世界位姿。离线复核逐个读取 NPZ，核对摘要、维度、坐标、步长和有限性。

一套 `RopeBatch` 只有一根独立杆，处于远离手掌的初始位置。逐指调用官方 `RopeBatch.add_mesh()`，实际返回连续 0–9 号 slot，后端 wrench 读取形状为 `[1, 10, 6]`。该张量形状用于核对注册计数。**没有执行求解步、Isaac、物理接触、力施加或抓取。** 配置的 5 mm 半径只用于本次登记与离散检查。单指接触仍须在独立柔性物理场景进行同一物理时钟的验证。

## 复现

```bash
private=/home/xk/.local/share/dh116-autonomous-3c80166/evidence
f1="$private/flexible-f1-20261112-883RS4"
export CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH="$f1/sdk/lib64"
export PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1
/mnt/mydisk/xk/envs/dh116/bin/python scripts/bake_dh116_deformx_pad_sdfs.py \
  --capture-dir "$private/flex-f4-parity-prep-Vu9H0f" \
  --lock docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json \
  --ropesim-package-root "$f1/build-attempt3/python" --output "$new_output" --execute-gpu
```

默认模式仅核验源证据。实际退出码 0，私有目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-f4-sdf-register-rK7gw0`。原始网格与 GPU 日志仅保留在该目录。

| 文件 | SHA256 |
|---|---|
| 实际 GPU 注册摘要 | `db5d75e780082640a29ae94ab010052dc19a354d574b6d82cf331bf1414355ab` |
| Warp 设备日志 | `667b1ad7c13d76ebf935f25d72b467a70a360c5cc3b5c5ebe8cddd1ca7233737` |
| OpenUSD CPU 回归日志 | `00b0c60c1cfd5e4e8e020924d6299814bb818fdebacbd8a3ba513634b700999b` |

OpenUSD CPU 回归 18 项通过。裸解释器 17 通过、1 跳过（缺 `pxr`）。528 个保护文件和 2 个历史 Checkpoint SHA 核验通过。
