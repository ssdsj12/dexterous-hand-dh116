# DeformX：服务器源码、GitHub 认证与实际安装来源补查

日期：2026-11-10，时间均为 UTC。目标：`xk@172.31.179.148`，项目根目录 `/mnt/mydisk/xk/dh116`。本次是只读排查及报告更新，没有下载或初始化子模块，没有安装、升级、编译、启动 Isaac 或修改训练。

最新编译决策：用户现已选择最终使用独立 **CUDA12.8**，现有12.2/12.4不作为最终编译替代；不会升级或替换原环境。具体路径与待执行命令见[独立工具链计划](../plans/2026-11-10-cuda128-isolated-toolkit.md)。以下保留本轮实测与历史排查结论。

**明确阻塞：尚未找到可核验的完整 CUDA 后端源码，服务器也没有已验证可用的 GitHub 读取身份。** 父仓库能访问不代表独立的 CUDA 子模块能访问。取得完整后端前，不判断现有环境能否直接编译，不编造 `ropesim` API 或接入实现。

## 1. 真实路径与仓库身份

| 路径 | 实际内容 / Git 核验 | 结论 |
|---|---|---|
| `/mnt/mydisk/xk/dh116` | DH116 Git 仓库，branch `11-3`，HEAD `c849cfeb7aae277d8ed56c50f2dce8fb41e48e40` | 这是部署项目，不能把其 HEAD 当作 DeformX/CUDA 版本 |
| `/mnt/mydisk/xk/dh116/vendor/DeformX` | 既有 PyElastica / `co_sim` 源码，无独立 Git 元数据；父仓普通 tree `82e6dfdf0b4e7578678b3f832c292a9e5ff7badb`，不是 gitlink | 这个 tree hash 不是上游 commit，不能证明目标 CUDA pin |
| `/mnt/mydisk/xk/projects/*/vendor/DeformX` | 另有 18 份同类 vendored 源码；完整路径列在本地 source inventory | 没有独立 Git HEAD，不能用外层目录名或父仓库提交号证明 CUDA pin |
| `/mnt/mydisk/xk/projects/IsaacLab` | 当前 editable Isaac Lab 的源码目录，无 Git 元数据 | 安装来源已定位，但不能确认 Git commit 或官方未修改状态 |
| `/mnt/mydisk/xk/integrations/deformx-gpu-20261110` | 官方 HF 最小资产、语法检查回执和 Python 源码审计 ZIP | 是此前独立审计产物，不是完整 CUDA 后端或安装结果 |
| 项目 `third_party` | 主项目和本次所列项目根均不存在 | 不能当作已安装依赖 |

源目录发现检查覆盖 22 个项目根、1,533 个目录。定名依赖树（`vendor/third_party/thirdparty/deps/external` 等）最多查至 10 层，普通项目最多 4 层；不跟随符号链接，不扫描模型、数据集、环境、缓存和日志。主代理于 `08:00:03 UTC` 另对 `downloads/temp/tmp` 做目录名补扫（最多 5 层、5,000 目录/20 秒上限），实际访问 59 个目录，没有匹配的 DeformX/Cosserat-CUDA/ropesim 目录，未触发上限、无读取错误，没有解压归档。这项补扫与原 inventory 分开留存，不合并冒充同一次扫描。结论只覆盖列明范围，不声称做过无界全盘搜索或解压所有归档。

19 份候选 `DeformX` 均有实际 Python 源文件，不能简单称为空目录；但它们是旧 PyElastica/co_sim 源树，每份 138 个 Python 源文件、源码版本标记 `0.3.3post1`，所查树中没有 `.cu/.cuh/CMakeLists.txt/setup.py/.so/.a`，也没有独立的 Git 仓库。这些信息既不证明安装了原生 GPU 后端，也不能证明目标提交存在。

主项目的实际核验：

```text
git submodule status --recursive       -> exit 0，输出为空
git ls-tree -r HEAD                    -> gitlink 数 0，.gitmodules 数 0
git cat-file -e 96b8514c7c55c724f1abba1fbf6ed6642b7db72b^{commit}
                                      -> exit 128，Not a valid object name
```

这说明该 DH116 对象库中没有目标 commit，且 HEAD 没有声明 CUDA 子模块；不是“子模块已安装”。候选 CUDA 仓库本身未找到，所以其当前检出版本、dirty 状态及嵌套子模块无法核验，没有将“不适用”写成“通过”。

## 2. 父仓库与 CUDA 后端分别是什么

用户提供的 [DeformX 父仓库分支](https://github.com/DeformX/DeformX/tree/codex/gpu-release-demos) 于 `07:41:42 UTC` 再次核实仍为：

```text
DeformX/DeformX @ 8f32ad730c3d321e03397810c1eabd972f38cd5b
Cosserat-Rod-Solver-CUDA gitlink @ 96b8514c7c55c724f1abba1fbf6ed6642b7db72b
子模块 URL: https://github.com/DeformX/Cosserat-Rod-Solver-CUDA.git
```

子模块引用不是 CUDA 文件内容。[固定版本的说明](https://github.com/DeformX/DeformX/blob/8f32ad730c3d321e03397810c1eabd972f38cd5b/demo/EXAMPLES.md#L35) 明确 release preview 后端需要访问权限，并依赖该固定版本的确定性 stepping API。

此前 Windows 本机现有 Git 凭据能够读取父仓库，但读取 CUDA 仓库及其指定 tree 均返回 HTTP 404。这只能说明当时该认证无法看到该资源，结合 preview 说明不能进一步断言仓库永久不存在，也不能把此认证结果套用到服务器身份。本次没有重复请求或下载已返回 404 的后端。

## 3. 服务器现有 GitHub 认证

| 检查 | 服务器实测 | 可得结论 |
|---|---|---|
| `gh api user` | exit 4，CLI 要求登录 | 当前 `gh` 未建立可用认证，未取得 GitHub 用户身份 |
| 非交互 Git HTTPS credential lookup | helper 数 0；lookup exit 128，无可用凭据 | 不能用此路径验证 private/preview 仓库权限 |
| 当前 `GH_TOKEN/GITHUB_TOKEN` | 未设置 | 没有使用环境变量中的认证 |
| 匹配该 URL 的 HTTP extraheader / `core.sshCommand` | 均未配置 | 没有遗漏这两类当前 Git 覆盖认证路径 |
| 当前 SSH agent / 默认与配置的 identity 文件 | 无 agent socket；存在的 identity 文件数 0 | 未发现当前 SSH 配置可用的出站身份；没有读取私钥内容 |
| GitHub SSH | 未认证成功；独立 `github.com:22` TCP 检查 5 秒超时 | 这是网络可达性失败，不能据此断言公钥无权限或仓库不存在 |

因此，服务器上的“已授权账号是否能读取 CUDA 仓库”目前**无法验证**。没有把未登录状态冒充已登录账号的权限拒绝，也没有向服务器复制 Windows 凭据、创建密钥、修改 SSH/credential 配置或输出 Token/密码/私钥。

## 4. 实际解释器、环境与安装来源

实际运行检查使用原启动器：

```bash
cd /mnt/mydisk/xk/dh116
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
  bash scripts/python.sh -
```

`activate.sh` 明确设置上述 `DH116_PYTHON`。`scripts/python.sh` 的旧 fallback 仍是 `/opt/conda/envs/hand/bin/python`，该路径不存在；本次没有采用该 fallback，也没有修改脚本。以下是实际执行该显式覆盖命令后的读回值：

| 项目 | 实测 |
|---|---|
| `sys.executable` | `/mnt/mydisk/xk/envs/dh116/bin/python`，解析至 `/usr/bin/python3.10` |
| Python | 3.10.12；不是原文档的 3.10.20 |
| PyTorch | 2.7.0+cu128 |
| `torch.version.cuda` | 12.8；静态导入后 `torch.cuda.is_initialized()` 为 False |
| NumPy | 1.26.4 |
| nvcc | `/usr/local/cuda-12.2/bin/nvcc` 12.2.91；`/usr/local/cuda-12.4/bin/nvcc` 12.4.99 |
| 默认 CUDA 链接 / PATH | `/usr/local/cuda -> cuda-12.2`；PATH 中无 nvcc |
| GPU / compute capability | 4×RTX 4090 / 8.9（SM89） |
| Driver | 580.178.04；驱动报告 CUDA 13.0，不等于安装了 13.0 Toolkit |
| Isaac Sim distribution | 4.5.0.0 |
| Isaac Lab distribution | 0.54.4 |
| Isaac Lab 源码根 `VERSION` | 2.3.2；与 distribution 编号是不同层级 |
| RSL-RL | `rsl-rl-lib` 3.1.3 |

安装来源核对：目标包的 `INSTALLER` 记录为 `uv`。`isaacsim` 元包加 24 个 `isaacsim-*` 组件，共 25 个 distribution，版本均为 4.5.0.0；其中 7 个 `direct_url.json` 明确指向 `/mnt/mydisk/xk/downloads/isaacsim_*.whl` 本地文件，其余没有记录 `direct_url`，无法从这些 metadata 证明历史下载站点。Isaac Lab 相关 4 个包通过 editable `file://` 来源指向 `/mnt/mydisk/xk/projects/IsaacLab/source/*`。RSL-RL 为普通非 editable 安装，没有记录 `direct_url`。

**不把 Isaac Lab 的 distribution 0.54.4、源码 `VERSION=2.3.2`、Git commit 三者混为一谈。** 源码没有 Git 元数据，不能证明它是未经修改的官方 2.3.2；接口继续以实际安装文件及 SHA 为准。`VERSION` 文件 SHA256 为 `e1d31f14bed34a1c7e6227931673f0bb4ec1e26ee4278d00dac169809c4d9539`。主代理的安装 metadata 与 scratch 回执独立保存在本地 `followup/root-origin-and-scratch-verification.json`，没有把文件存在或版本文本当作 Git 身份证据。

该阶段仅记录实测环境；后续用户明确了最终CUDA12.8目标，采用独立部署方式而非升级现有环境。需要取得真实CMake、bindings和kernels后执行12.8构建；本报告没有证明现有12.4可用，也不把12.4作为最终替代。

## 5. 不依赖后端源码的 DH116 接口梳理

此前实际安装文件审计已确认以下边界，本次仍可作为后续独立适配输入；这不是 `ropesim` API 或原生验证：

- 当前 Lab 可读 DH116 的实际 body names、link world pose/velocity、CoM pose/velocity；link 原点与 CoM 的速度及力矩参考点必须区分。
- 当前 Lab pose 使用 wxyz；raw PhysX pose 使用 xyzw，网格局部偏置、世界坐标与环境原点须显式处理。
- 当前 PhysX 有 articulation/rigid 两种 `apply_forces_and_torques_at_position`；`position_data=None` 的绑定文档指 link transform 原点，不能假定 CoM。
- `write_data_to_sim → step → update` 的控制写入、反力提交和状态刷新顺序必须只由一个物理步边界管理；外部 GPU 反力不能冒充原 `/Wire` PhysX contact sensor 的测量。
- 上游示例中的 `mesh_loads` 标为力估计，不能直接复用旧后端 impulse/dt 的换算。实际后端的力/冲量单位、力矩原点、符号和 CUDA 同步仍待源码确认。

19 个目标 API 文件的 SHA 与详细矩阵见上一份[兼容性报告](2026-11-10-deformx-gpu-compatibility.md)及本地 API manifest。没有向 Teacher/Student 观测或控制器接入空实现，也没有用旧后端冒充新 GPU solver。

## 6. 解除阻塞所需材料

以下二选一即可先解除源码阻塞：

1. 使一个已授权账号能够读取实际 CUDA 子模块仓库及提交 `96b8514c7c55c724f1abba1fbf6ed6642b7db72b`，并能取得其全部嵌套依赖；父仓库链接本身不足以提供这些内容。
2. 提供服务器上完整 CUDA 仓库的真实目录或可核验源码归档，包含构建文件、Python bindings、kernels、`ropesim.isaac_bridge` 及嵌套子模块；若是归档，需要版本和文件校验信息，不能仅有空目录、gitlink 或另一 Python ABI 的二进制。

取得后先只读核对 Git 根、commit、dirty 和子模块状态，保留本地修改，再按当前实际环境分析编译与接口适配。当前无需以升级或替换环境来解除源码访问问题。

远端主项目最后复核仍为原 HEAD/branch 和原 1 处 modified、3 处 untracked；没有执行 checkout、reset、submodule update、安装、模型迁移或训练操作。PR 继续 Draft，未进行 CUDA 构建、GPU 求解、PhysX 耦合或抓取成功验收。

## 附：本次找到的 19 份旧源码实际路径

以下均为上述已核对的 PyElastica/co_sim vendored 目录，不能从列表存在推断 CUDA 后端已安装：

```text
/mnt/mydisk/xk/dh116/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-align-c849cfe/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-contact-geometry-20261106/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-contact-observer-smoke-20261106T1048Z/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-lift-repair-20261106/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-mixed-reset-20261105/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-mpc-contact-v2-20261106/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-native-eval-20261106T0850Z/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-native-smoke-20261106T0834Z/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-palm-lift-20261105/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-reviewed-native-f29eb51-tested-v2/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-reviewed-native-f29eb51/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-tripod-20261107/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-tripod-recovery-20261107/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-vision-coordinate-20261105/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-vision-fixed-20261105/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-vision-student-20261105/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-vision4096-20261105/vendor/DeformX
/mnt/mydisk/xk/projects/dh116-visual-teacher-mpc-20261106/vendor/DeformX
```

本地证据包括 `source-inventory.{json,md}`、`server-github-authentication.json`、`installation-origin.{json,md}` 与 `root-origin-and-scratch-verification.json`。安装来源报告校验了 175 个 distributions 的版本/metadata 路径快照及项目状态前后一致；不将这种快照等同于所有安装 payload 的逐文件校验。没有导入 Isaac，静态 Torch 检查没有初始化 CUDA。
