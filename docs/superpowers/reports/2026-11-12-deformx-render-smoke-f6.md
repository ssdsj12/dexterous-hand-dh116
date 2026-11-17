# F6 DeformX 派生场景原生渲染冒烟

原生入口为 `scripts/run_dh116_deformx_render_smoke.py`。它核对固定 DeformX 主仓库 `04f6a11eb4d832df761f631cf6ce1bc00da75eb1` 与派生资源的全部文件摘要，再通过现有 DH116 单 Isaac 实例锁启动。范围是 drop variant 0 和 hang variant 3 的 frame 0，seed 42，各 1 个相机、1×1 累积。使用原生成器的相机、轨迹和线缆引用。应用配置使用 `fast_shutdown=False`。

RGB、`distance_to_camera` 和 `instance_id_segmentation_fast` 三个 Annotator 在同一个原生渲染产品上分步读取。距离为相机径向距离，单位米。ID 保留 32 位整数和 `idToLabels`，原生成器的可见性处理仍适用。该冒烟的单相机输出仅用于验证资产和运行链路。实际生成器持有的 composed stage 和每个导出文件由清单绑定摘要。

## 保留的失败和修复

| Ubuntu 独立目录 | 实际结果 |
|---|---|
| `f6-native-bounded-G7fSVZ` | Drop 生成 RGB 和数值 ID。Isaac 4.5 Annotator 的列表 detach 分支将 `HydraTexture` 对象传给要求路径字符串的 SDK，触发 `.split()` 异常。随后 BasicWriter 已初始化的后台队列在 Python atexit 中触发 SIGSEGV，外层 139。Hang 未启动。 |
| `f6-native-direct-vYc3Ju` | 用稳定路径读写 Annotator，三个产品同步落盘。Drop 的 composed stage 可重新读取，0 条未解析依赖，2 个线缆 ID，798,833 个有限正深度像素。App.close 已返回，外层仍在 Python GC 阶段 SIGSEGV、139。Hang 未启动。 |

已保留两次退出码、日志、实际数据与对应入口源码。第一轮日志 SHA256 为 `e2b4a5d79d5f68267fc3d6fca2e88049948dfcef52d58d5991f1f92a5422f9da`。恢复入口持有明确的唯一渲染产品，读取时复制数组，再使用同一个 USD 路径解绑。为避免原生成器在 `render_frame` 局部变量仍持有 stage 和产品时关闭 stage，恢复入口把实际关闭推迟到运行函数退栈后。顺序为 Replicator stop 和等待、销毁产品、关闭 stage、释放 Python 生成器、GC、App.close。同步文件读写没有后台 Writer 任务，最后 App.close 使用 `wait_for_replicator=False`，避免在 stage 已关闭后重复等待。

## 复现

```bash
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
prepared=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/f6-render-checked-Nmc1SI/prepared
upstream=/mnt/mydisk/xk/projects/deformx-f0-04f6a11/parent
record=$(mktemp -d /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/f6-native-audit-XXXXXX)
for variant in 0 3; do
    code=0
    bash scripts/python.sh scripts/run_dh116_deformx_render_smoke.py \
      --prepared "$prepared" --upstream "$upstream" --output "$record/variant-$variant" \
      --variant "$variant" --execute-native >"$record/variant-${variant}.launcher.log" 2>&1 || code=$?
    printf '%s\n' "$code" >"$record/variant-${variant}.exit_code"
    if [[ "$code" != 0 ]]; then break; fi
done
```

移除 `--execute-native` 可运行同一预检且保持 Isaac 关闭。原生结果由外层退出码和独立读取产物一并验证。入口内 `completion.json` 仅说明 App.close 返回，内部字段 `native_render_verified=false` 和 `outer_exit_verified=false` 保持保守。

## 最终本轮原生证据

第三次运行 `f6-native-owned-release-9Xw638` 从 09:10:53 UTC 至 09:12:07 UTC。运行入口的 SHA256 是 `03b58e47802d81891effeb0d28e894f3edd8da3ec46722bf7ddf858ae86fd083`。Drop variant 0 采集完成后 App.close 返回。外层仍在解释器 GC 阶段 SIGSEGV、139，faulthandler 没有 Python 帧。此次复现支持渲染产物已生成，尚未定位 C++ 析构对象。序列门控使 hang variant 3 未启动。

| 核验项 | 第三次 drop 实测 |
|---|---|
| 图像和原生注释 | 1024×1024 RGB、径向距离和 uint32 Native ID，2 个被观测 ID 对应独立线缆 |
| 径向距离有效像素 | 798,833 个有限正值 |
| Composed stage | Z-up、1 m/单位、1 个代理相机、2 个 2,652 顶点的线缆网格、可解析 DomeLight 纹理 |
| 独立 OpenUSD 重读 | 0 条未解析依赖 |
| Composed stage SHA256 | `3a20ea8e0cd356b18274c69f320fb4e1706b3fea54c6c825bdaaa9be87794293` |
| 外层日志 SHA256 | `fafe7ef72c3bc4d2b422dfd8a71f6f8b8fc3195b69341f793c24d2fa57f9b58c` |
| 三轮独立核验 JSON SHA256 | `233939b34b2413405a0fcaad1de8f1030852175a0cb5af3f503d2590a10aba71` |
| Ubuntu CPU 回归 | 19 通过、0 失败，SHA256 `32bb5ef3de2ab6be3409ace1c5d3cf439884134c42d8aaaff20404a311183a32` |

三轮证据保存在 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/` 下各独立运行目录。F6 的派生 USD 依赖审计已通过，`native_render_verified=false`。下一次诊断应在单实例 GPU 窗口，用相同入口做 variant 0 的 C++ 回溯，随后依据真实进程退出决定是否运行 variant 3。
