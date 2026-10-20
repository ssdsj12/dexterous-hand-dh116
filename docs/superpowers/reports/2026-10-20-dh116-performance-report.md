# DH116 仿真性能对照报告

日期：2026-10-20  
分支：`feat/dh116-simulation-10-20`  
提交：`25190c6`（测量时 verifier 来源提交）

本次使用单个 Isaac Sim 进程串行运行，每组 3 次，`physics_dt=1e-4 s`、DeformX `py_dt=1e-5 s`、40 个杆单元、每单元 16 个接触采样、GPU `cuda:0`，CPU 亲和性为 core 0，OMP/MKL/Numba 为单线程。每次运行 1200 个物理步（0.12 s），桌面和 wire 使用相同 USD、接触网格、材料和碰撞保护。每次 `summary.json` 都报告 `passed=true`、1200/1200 步完成；这些是 settle/diagnostic 运行，没有进入闭指、持续夹持或抬升阶段。

硬件环境：AMD Ryzen 5 7500F 6-Core/12-Thread，NVIDIA GeForce RTX 5070 12 GB，CUDA 2.7.0+cu128，Isaac Lab 0.54.2，Ubuntu 22.04.5，CPU governor 报告为 powersave。

| variant | median wall s | P95 wall s | median SPS | median RTF | 相对 baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline / headless / audit | 36.953951 | 36.954577 | 32.472847 | 0.00324728 | 1.000000× |
| buffered / headless / buffered | 36.933635 | 37.060001 | 32.490709 | 0.00324907 | 1.000550× |
| visual / GUI 15 Hz / audit | 44.014520 | 44.181878 | 27.263730 | 0.00272637 | 0.839585× |

原始运行目录：`runs/perf_20261020_settle/`。机器可读汇总：`runs/perf_20261020_settle/report.json`。每组的物理身份、完成步数和安全状态由 `collect_performance.py` 检查后标记为 `comparable=true`。

结论：证据 buffered flush 在这个 0.12 s 单核场景中只有约 0.055% 的中位改善，处于测量噪声范围，不能称为确定提速。GUI 15 Hz 发布使墙钟时间增加约 19.1%；它用于降低 wire 网格发布频率和保持物理连续性，不是整体加速手段。当前主要热点仍是 DeformX 杆/接触求解（profile 中 `rod_and_contact` 占绝大多数）。

本报告没有改变物理参数，也没有把诊断通过解释为抓取成功。正式 close/hold/lift 仍需完整 5.1 s 以上连续运行并检查四指反作用、滑移、桌面余量和 wire 抬升；在此之前不提交抓取成功结论。
