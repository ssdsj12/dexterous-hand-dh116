# WP5 实际 CUDA 求解器复核

基线 `ae307a55917519138b6da5179e8254ce2e9a7c8f`。2026-11-11 10:18:03–10:18:04 UTC，通过现有 SSH 只读复核 `/mnt/mydisk/xk` 内既有项目和依赖目录。没有修改部署项目、安装环境、下载后端或编译。

| 实际检查 | 本次结果 |
|---|---|
| 项目根 | 22 个 |
| DeformX 候选目录 | 19 个，均不是独立 CUDA solver Git 工作树 |
| 候选中的 CUDA `.cu/.cuh` | 0 |
| 已发现 `.gitmodules` / HEAD gitlink | 0 / 0 |
| 当前实际解释器的 `find_spec('ropesim')` | `None` |
| Python / PyTorch / NumPy | 3.10.12 / 2.7.0+cu128 / 1.26.4 |
| Isaac Sim / Lab distribution | 4.5.0.0 / 0.54.4 |
| `/usr/local/cuda*/bin/nvcc` | 12.2.91、12.4.99；`cuda` 链接也为 12.2 |
| 目标 Toolkit 12.8 | 本次标准路径未找到 |

这些结果只覆盖清单中的有界搜索。19 个目录包含既有 PyElastica/co_sim，不能作为目标 `Cosserat-Rod-Solver-CUDA` 已安装的证据。此前目标提交为 `96b8514c7c55c724f1abba1fbf6ed6642b7db72b`；本轮没有重复尝试已经确认不可访问的 preview 仓库下载，也没有将父仓可见性当作子模块权限。

**阻断保持**：完整目标 CUDA solver 源码及依赖、可验证构建/安装缺失。因此本轮没有创建替代性 solver bridge，没有运行 GPU 杆求解、双向反作用力或动量收支验证。既有 `DeformableRodStateProvider` 继续只是受检的状态入口。`rigid_proxy` 与 `deformable_rod` 身份不合并。

解除阻断需要可读的完整 CUDA 仓库及固定提交/嵌套子模块，或者包含 bindings、kernels、构建文件与来源校验的真实源码目录。取得后按既定独立 CUDA 12.8 安装方案处理，不替换当前 Driver、Python、PyTorch、Runtime 或 Isaac。不能从已安装的 PyTorch cu128 推断 nvcc 12.8 已安装。

本地原始清单为 `artifacts/stereo-wp-20261111/solver_source_inventory.json` 与 `solver_install_inventory.json`。源目录发现脚本沿用已审计的 `artifacts/deformx-20261110/followup/source-inventory.remote.py`，没有向库中再添加一套空接口。原始列表与安装边界见[此前来源审计](2026-11-10-deformx-server-source-access.md)。
