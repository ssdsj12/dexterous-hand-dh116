# DeformX GPU → DH116 11-10：源码、工具链与资产审计

日期：2026-11-10。DH116 基线：`48992ceca5129d94aa50459f70aa1b6bf1fc1e48`，分支 `11-10`。

最新目标更新：用户现已明确最终编译采用 **CUDA 12.8**；现有12.2/12.4只记录为保留基线，不用于冒充最终构建。独立新增Toolkit方案与源码/管理员门禁见[计划](../plans/2026-11-10-cuda128-isolated-toolkit.md)，目前未安装、未编译。

**当前结论：尚未完成原生适配，CUDA 后端源码不可访问是明确阻塞项。** 已取得并校验指定上游版本的可访问源码，完成现有环境与目标 Isaac 接口的只读审计，以及最小资产子集的实际读取。指定 CUDA 子模块处于访问受限的 release preview，Windows 现有 Git 凭据访问返回 404；服务器也未找到 CUDA 12.8 的 `nvcc`。未执行 CUDA 扩展构建、GPU 求解、PhysX 耦合或 DH116 抓取实验。

执行口径：以当前服务器实际Python/Torch/Runtime为保留基线，不升级或替换；用户最新决定是独立增加CUDA12.8作为最终编译工具链，12.2/12.4不作替代。下表的“请求合同”保留最初清单供对照，实际构建须等待完整后端源码及独立12.8 SDK验收。服务器认证与 Windows 认证分别记录，不能混为同一身份。真实 Git/submodule 与安装来源补查见[后续报告](2026-11-10-deformx-server-source-access.md)：Lab distribution 为 0.54.4，而源码根 VERSION 为 2.3.2，没有可验证的源码 Git commit。

## 1. 目标服务器与环境实测

用户确认以当前服务器 `xk@172.31.179.148`、项目 `/mnt/mydisk/xk/dh116` 为准。实际解释器为 `/mnt/mydisk/xk/envs/dh116/bin/python`，不是任务初始清单中的 `/opt/conda/envs/hand/bin/python`。所有目标 Python 检查均通过现有 `scripts/python.sh` 显式设置 `DH116_PYTHON`；没有修改启动器。

| 项目 | 请求合同 | 实测 | 判定 |
|---|---|---|---|
| Python | 3.10.20 | 3.10.12，真实可执行文件 `/usr/bin/python3.10` | 同属 cp310，补丁版本不一致；保留现状，不声称精确满足 3.10.20 |
| PyTorch | 2.7.0+cu128 | 2.7.0+cu128，CXX11 ABI=True | 版本一致；扩展 ABI 仍须取得源码后验证 |
| NumPy | 1.26.4 | 1.26.4 | 一致，未降级 |
| Isaac Sim | 4.5.0 | distribution 4.5.0.0 | 现有安装，未升级 |
| Isaac Lab | 0.54.4 | distribution 0.54.4，editable 源码根 VERSION=2.3.2 | 两层编号不能混用；无 Git commit 可核，现有文件逐项保存 SHA |
| RSL-RL | 3.1.3 | `rsl-rl-lib` 3.1.3 | 一致 |
| GPU | RTX 4090 / SM89 | 4×RTX 4090 / compute capability 8.9 | 未来构建应明确设为 89，不能复制上游 RTX 5090 的 120 |
| CUDA Toolkit | 12.8 | 12.2.91、12.4.99 | 在明确搜索范围内未找到 12.8 `nvcc`；尚未选用其他版本编译 |
| NVIDIA driver | 支持目标运行时 | 580.178.04，驱动报告 CUDA 13.0 | 驱动上限，不是已安装 Toolkit 版本 |
| GCC/G++ | 需符合实际源码 | 12.3.0 | 未进行编译；不等于 C++/CUDA 构建通过 |
| CMake / Ninja | 上游示例列 CMake≥3.24、Ninja | CMake 3.22.1；当前 PATH/env 未发现 Ninja | 与示例配置有差异；真实 CMake 最低要求因缺源码仍未知 |

现有 `nvidia-cuda-runtime-cu12` 12.8.57、`nvidia-cuda-nvrtc-cu12` 12.8.61 和 `torch.version.cuda=12.8` 不能代替 `nvcc`。`/usr/local/cuda` 指向 `cuda-12.2`；12.2/12.4 均有 headers、libcudart、cicc、ptxas，且列出 `sm_89`。本次不通过升级或绕过版本检查消除差异。

## 2. 上游源码身份与取得范围

| 来源 | 固定版本 | 实际取得范围 |
|---|---|---|
| [DeformX 指定分支](https://github.com/DeformX/DeformX/tree/codex/gpu-release-demos) | `8f32ad730c3d321e03397810c1eabd972f38cd5b` | 完整递归 tree；299 个小文件逐 Git blob SHA 校验；29 个大文件为视频/GIF，未下载 |
| `PyElastica-Mesh` gitlink | `a5cfaf60c4a1aa707ed94bdd07641de1a90ae650` | 330 个小文件逐 blob 校验；仅排除 3,472,634-byte `mytest/bunny.stl`；tree 无嵌套 gitlink |
| `Cosserat-Rod-Solver-CUDA` gitlink | `96b8514c7c55c724f1abba1fbf6ed6642b7db72b` | 仓库及指定 tree 在当前认证下均 404；源码与嵌套子模块未取得 |

源码位于独立的 `deformx-gpu-audit-20261110` 审计目录，不是对现有 `vendor/DeformX` 的覆盖。Git HTTPS 获取失败后，采用认证 API 获取 tree/blob 并核对 Git 对象 SHA；没有将部分源码目录冒充成功的 recursive clone。

指定版本 [README](https://github.com/DeformX/DeformX/blob/8f32ad730c3d321e03397810c1eabd972f38cd5b/README.md#L136) 与 [demo/EXAMPLES.md](https://github.com/DeformX/DeformX/blob/8f32ad730c3d321e03397810c1eabd972f38cd5b/demo/EXAMPLES.md#L35) 明确说明 preview 后端需要仓库访问权限，且示例依赖该 pin 的确定性 stepping API。因此不能拿 `main` 的不同 gitlink、旧 Torch/PyElastica 后端或其他求解器替代。

## 3. Python / 构建兼容性矩阵

| 审计项 | 源码证据 / 实测 | 状态与边界 |
|---|---|---|
| 主库 Python 声明 | `pyproject.toml:5`，`>=3.10,<3.13` | 无须通过修改版本声明放行 3.10 |
| CPU 子模块 Python 声明 | `PyElastica-Mesh/pyproject.toml:28`，`>=3.10` | 不代表原生 CUDA 后端也兼容 |
| 已取得 Python 源码语法 | 426 个 `.py` 在实际 Python 3.10.12 上逐一 `compile(..., 'exec')` | 426 通过、0 失败；未导入/执行这些模块，不证明运行期 API 兼容 |
| Python 3.11 专属语法/API | 已取得文件语法检查无失败；CUDA Python 包未取得 | 不能对缺失的 `ropesim` 给出兼容结论 |
| CMake、C++17、CUDA kernel 最低要求 | 示例文档列出测试环境；真实 CUDA `CMakeLists.txt` 与 kernel 缺失 | 未审完，不能把示例测试配置当作编译硬要求或编译成功 |
| PyTorch C++ Extension / pybind 接口 | 现有 Torch headers/ABI 已记录，后端实现缺失 | 未验证，无依据断言零修改可编译 |
| NumPy C API / cp310 wheel | 后端 binding 源码缺失 | 未验证；没有使用 cp311 二进制冒充 cp310 |
| GLM / 其他嵌套依赖 | 示例推荐 GLM 0.9.9.8；CUDA 嵌套子模块缺失 | 需取得完整依赖树再核验，未安装依赖 |

首轮有界搜索访问 1,134 个目录，并检查 19 份旧 `vendor/DeformX` 项目副本；未找到目标 CUDA 源码或已安装 `ropesim`。结论仅覆盖已记录根目录和深度，不声称全盘不存在；后续更针对 Git/vendor/third_party 的补查另记，不将这些目录冒充 Git worktree。

## 4. Isaac 4.5 / Lab 0.54.4 实际接口

检查了当前安装的 Lab/PhysX Python 源码，保存 19 个目标文件的路径、字节数、SHA256。没有按上游 2.3.2 的版本号猜测现有 API。

| 接口边界 | 当前目标源码确认 | 独立适配必须处理 |
|---|---|---|
| `InteractiveScene.reset/write_data_to_sim/update` | 存在；reset 清 buffers 不自动重写物理姿态 | rod 与刚体 reset epoch 配对；每步只写一次已有 actuator 命令 |
| `Articulation.body_names/root_physx_view` | 存在，实际 native link 顺序可读 | 按真实 body/env ID 绑定，不能猜测 mesh ID 等于 link ID |
| `body_link_pose_w/body_link_vel_w` | link 原点 world pose/velocity；Lab quaternion 为 wxyz | 碰撞网格局部偏置只组合一次；raw PhysX xyzw 与 Lab wxyz 显式转换 |
| `body_com_pose_w/body_com_vel_w` | 实测 CoM 状态，与 link 原点不同 | `v_link = v_com + omega × (p_link-p_com)`；力矩必须注明参考点 |
| Articulation/Rigid view `apply_forces_and_torques_at_position` | 实际 binding 有 force、torque、position、indices、is_global 参数 | 完整聚合每 link 反力，显式 world 作用点，逐 physics step 提交；多次调用累加/覆盖行为尚未原生验证 |
| `position_data=None` | binding 文档指 link transform 位置 | 不能假定 CoM；CoM 力矩应显式传实测 CoM world position |
| `set_external_force_and_torque` / wrench composer | 当前 editable 版本实际存在 | 会暂存到后续 write；需要审清瞬时/永久缓冲的清理顺序 |
| PhysX ContactSensor | 提供 PhysX 生成的接触数据 | 外部 GPU rod 的反力不等于旧 `/Wire` 接触传感器会自动报告 rod 接触 |
| `ropesim.isaac_bridge` 完整实现 | 不在主库，缺 CUDA 子模块 | 未审计，不声称旧桥可直接使用或已完成重写 |

目标 `omni.physics.tensors` Python binding SHA256：`bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b`。`Articulation`、`ArticulationData`、`WrenchComposer` 等源码完整清单保留在本地审计 manifest 中。

主库 demo 的已知差异：

- `rope_box_pulley.py:157-165` 先用上次反力推进 PhysX，再 `sync_poses → rope.step → apply_wrenches`，存在一个 exchange 的延迟；不能未经分析复制到 DH116 的 480 Hz 环中。
- `swing_contact_compare` 的机器人碰撞器是 `dynamic=False`，未调用 `apply_wrenches`，不能作为双向 articulation 验收。
- curtain/pile 的 headless 路径只步进 rod，不证明 PhysX 双向接触。
- `demo/cable_extraction/coupling.py:103` 将 `mesh_loads` 标为 surface force estimates、非 integrated impulses。取得真实后端源码后必须确认单位；不能套用旧适配器的 impulse/dt 再除一次 dt。

尚未写入或接通新的生产 Adapter。待完整后端到位后，新模块应独立读取真实 link/CoM 状态，显式处理坐标和单位，并在单个 PhysX step 边界交换载荷；不修改原控制器和策略合同。

## 5. 用户提供的官方资产

[DeformX-Assets](https://huggingface.co/datasets/DeformX/DeformX-Assets) 固定 revision：`9ef440fa263dcd9b39689c97860f1d94d599bc5a`。这是 USD/CAD、轨迹与渲染纹理 bundle，不是 GPU solver 源码，也不是 DH116 成功抓取示范。基础 GPU cable demos 使用程序化 rope，不依赖下载整套约 2.18 GB 资产。

独立目录：`/mnt/mydisk/xk/integrations/deformx-gpu-20261110/hf-assets-9ef440fa`。本次仅保存 README、manifest、来源记录和以下最小数据；没有替换 DH116 原资产或物理参数。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `usd/wire_usdc/wire_usdc/wire_yellow_s20_r0.005_l1_smooth.usdc` | 272528 | `3a0f63105f1aea7b8a765a56f68439b7f21c20ab7318f887e7a46a50ae78912f` |
| `usd/wire_usdc/wire_usdc/textures/color_0C0C0C.exr` | 461 | `e37663c01042f39172be838d2cec5f87dd5a569d57d22517e15130b2bb797526` |
| `wires_traj_data/drop_n2_100.npz` | 407158 | `afa19e40f2c9a80b2788ae91a7fc433cca9d81ffde9dfd67e3a4a03cdbb24983` |

三个数据文件均按该 revision 的 manifest 校验 SHA 与字节数，复制到服务器后再次校验。实际读取结果：USD metersPerUnit=1、Z-up、2,652 mesh points、骨架可读、局部包围盒半径约 5 mm / 长 1 m；NPZ 用 `allow_pickle=False` 加载，记录全部字段 shape/dtype，浮点字段均有限。没有将轨迹数据播放算作物理计算。

该 revision 的 README 内容与 manifest 中 README 的旧 hash 不一致；此文档索引差异被保留记录，未修改或伪造校验结果。三个选定数据文件的校验均通过。资产页面列原作者 USD/轨迹 MIT、HDRI CC0、Pexels 纹理适用其许可；未下载 NVIDIA 数据中心包或 Texturelabs 资源，不能把仓库 MIT 延伸到这些第三方资产。

## 6. 实际执行与未执行的验收

实际目标检查入口：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ForwardAgent=no xk@172.31.179.148
cd /mnt/mydisk/xk/dh116
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
  DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -
```

stdin 检查分别执行 package metadata/静态 Torch 信息、`compile(zip.read(name), name, 'exec', dont_inherit=True)`、`Usd.Stage.Open(...)` 与 `np.load(..., allow_pickle=False)`。426 个 Python 文件的输入 ZIP SHA256：`f093e565cab441140275d04aae9cfe8303e42ad6e6ec17a97af17448d4da1fce`。均 exit 0；未启动 SimulationApp。

| 验收层次 | 本次结果 |
|---|---|
| 上游身份、可取得源码 hash | 主库/CPU 子模块 629 个文件重验一致；CUDA 子模块未取得 |
| 实际 Python 3.10 语法 | 426 通过、0 失败；不是运行期或 CUDA 编译通过 |
| 官方最小资产完整性/实际读取 | 3/3 SHA 校验，USD/NPZ 结构读取通过；未做 EXR 解码或物理验收 |
| cp310 CUDA 扩展构建 | 未开始：精确 CUDA 后端源码尚未取得；须等待独立12.8工具链验收，不能用12.2/12.4替代；不是一次编译失败日志 |
| ropesim 初始化 / RTX 4090 实际求解 | 未执行 |
| 单线重力下垂 / 桌面接触 | 未执行 |
| 单刚体 GPU↔PhysX 双向耦合 | 未执行 |
| 单 DH116 单指 / 多指 | 未执行 |
| 双 DH116 接触 / 抓取提升 | 未执行；不宣称抓取成功 |
| CPU/GPU 性能对比、动量误差 | 未执行，没有可报告性能数字 |

## 7. 保护核验与下一步

没有 pip 安装/升级，没有更换 Python、Torch、CUDA、NumPy、Isaac、Lab 环境，没有启动新的 Isaac 或停止现有作业。部署项目保持旧 branch `11-3`、HEAD `c849cfeb7aae277d8ed56c50f2dce8fb41e48e40`，原 1 个 modified 和 3 个 untracked 状态保留；保护审计覆盖的 79 个文件 hash 无变化，HEAD/status/tracked diff hash 一致。CPU 原后端、`vendor/DeformX`、控制器、Teacher/MPC/Student/PPO、checkpoint 与成功标准均未改动。

继续条件：

1. 提供可访问的精确 CUDA 子模块及其完整依赖，先审 `CMakeLists.txt`、bindings、kernels、`ropesim.isaac_bridge`。
2. 以当前实测 Python 3.10.12、Torch 2.7.0+cu128 和最终CUDA12.8目标审查真实后端要求；独立部署12.8，不升级/替换现有Runtime或12.2/12.4。按最新计划执行准入和完整SDK验收，不能提前声称已满足12.8构建要求。
3. 在独立源码/构建/产物目录执行无依赖替换的 cp310、SM89 构建，保存完整配置/编译/链接结果并重验包基线。
4. 完成独立 Isaac45 适配层后，串行执行用户规定的九级原生验收，分别报告状态、接触力/力矩、动量残差与耗时。未通过双手物理验证，不进入 Teacher/MPC 成功轨迹验收。

本报告不是最终原生适配完成报告；PR 继续 Draft，不合并。
