# CUDA 12.8 独立工具链与源码准入实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute the authorized steps. Do not cross the source-access or administrator-approval gates.

**Goal:** 在不替换现有 Driver、Python、PyTorch、CUDA Runtime、Isaac Sim/Lab 的前提下，为当前 cp310/cu128/SM89 准备独立 CUDA 12.8 工具链，取得完整 CUDA 后端后再重新编译扩展。

**Architecture:** 使用用户可写目录保存 NVIDIA 官方逐组件归档、校验记录和独立 SDK；不运行驱动/系统安装代码，不改全局 CUDA 链接或 loader。源码、Toolkit、构建与输出分别管理；所有实际安装和编译均在源码准入之后。

**Tech Stack:** Ubuntu 22.04.5 x86_64、Python 3.10.12、Torch 2.7.0+cu128、CUDA Toolkit 12.8.0、RTX 4090 SM89，原项目 `scripts/python.sh`。

当前状态：**计划已制定，Toolkit 未安装，Solver 未编译。CUDA 后端访问仍阻塞。** 日期与审计时间均为 UTC。此处 12.8.0 是 12.8 系列的具体冻结版本，不是降用 12.4；不修改原环境中已有版本。

## 0. 已完成的只读复核

- [x] `08:14:29 UTC` 重新检查 `/usr/local/cuda*`、实际 symlink 与 nvcc 版本，以及 59 个有限 glob 模式。发现 12.2.91、12.4.99；边界内未找到 12.8。
- [x] `/usr/local/cuda -> /usr/local/cuda-12.2/`；当前 PATH 无 nvcc。Torch cu128 和 runtime/NVRTC 12.8 元数据不能代替 Toolkit。
- [x] 拟用目录 `/mnt/mydisk/xk/toolchains/cuda-12.8.0` 不存在；最近父目录 `/mnt/mydisk/xk` 对用户 xk 可写、可遍历，可用空间约 3.82 TB。仅核对 metadata，尚未做创建、链接或执行探测。
- [x] 现有 GCC/G++ 12.3.0、glibc 2.35；CMake 3.22.1，Ninja 未发现。GLM headers/dpkg 是 0.9.9.8，pkg-config 字符串为 0.9.9.6，差异保留，不擅自修改。
- [x] 只读核实 Windows 现有 GitHub 身份为 `ssdsj12`；本次没有再次访问或下载已返回 404 的 CUDA 后端，也没有把凭据传到服务器。

主机细节与搜索边界记录在本地 `artifacts/deformx-20261110/cuda128-plan/host-preflight.{json,md}`；没有将旧 79 文件基线冒充本次重算结果。

## 1. 源码与权限准入（当前阻塞）

- [ ] 由 DeformX 维护者向 GitHub 账号 `ssdsj12` 开放 `DeformX/Cosserat-Rod-Solver-CUDA` 的只读访问，包括提交 `96b8514c7c55c724f1abba1fbf6ed6642b7db72b` 及其私有嵌套依赖；或者提供完整可核验源码仓库/归档。父仓库链接不能代替子模块源码。
- [ ] 权限状态发生变化后才进行一次读取验证；不反复重试同一 404。不代用户向维护者发送消息，也不在报告中保存 Token/密码/私钥。
- [ ] 在新目录获取指定提交，核对独立 Git 根、HEAD、dirty、递归子模块，保存真实构建文件和 bindings/bridge 的审计结果。原 `vendor/DeformX` 不动。

可转交维护者的具体请求：

> 请为 GitHub 账号 ssdsj12 开放 DeformX/Cosserat-Rod-Solver-CUDA 提交 96b8514c7c55c724f1abba1fbf6ed6642b7db72b 及其私有嵌套子模块的只读访问；也可提供对应完整源码和版本校验清单。

未来服务器源码目录约定为 `/mnt/mydisk/xk/integrations/deformx-gpu-20261110/source/Cosserat-Rod-Solver-CUDA-96b8514`。以下只读验收命令在源码实际取得后使用：

```bash
set -euo pipefail
backend=/mnt/mydisk/xk/integrations/deformx-gpu-20261110/source/Cosserat-Rod-Solver-CUDA-96b8514
test "$(git -C "$backend" rev-parse --show-toplevel)" = "$(realpath "$backend")"
test "$(git -C "$backend" rev-parse HEAD)" = 96b8514c7c55c724f1abba1fbf6ed6642b7db72b
git -C "$backend" cat-file -e '96b8514c7c55c724f1abba1fbf6ed6642b7db72b^{commit}'
git -C "$backend" status --porcelain=v1
git -C "$backend" submodule status --recursive
```

不能只凭命令 exit 0 判定通过：dirty 必须检查并保留，子模块不能有 `-`、`+`、`U` 状态，嵌套源码和必要 LFS 内容须核实。若目录已有修改，停止自动 checkout/覆盖。**准入未通过时，后续下载、安装、CMake configure/try_compile 和 Solver 编译不执行。**

## 2. 具体版本、目录与官方来源

优先冻结 CUDA **12.8.0**：官方 manifest 的 nvcc 为 **12.8.61**，cudart **12.8.57**、NVRTC **12.8.61** 与当前环境对应组件版本一致。这个选择减少 patch 差异，不代表允许替换 Torch 所加载的 runtime。[NVIDIA 12.8.0 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.0.json)

| 目录 | 用途 |
|---|---|
| `/mnt/mydisk/xk/toolchains/cuda-12.8.0-redist-staging` | 新建归档、inventory、component 解包目录 |
| `/mnt/mydisk/xk/toolchains/cuda-12.8.0/sdk` | 验收并组装后的独立 SDK 根，未来 CUDA_HOME/CUDAToolkit_ROOT 指向这里 |
| `/mnt/mydisk/xk/integrations/deformx-gpu-20261110/source` | 独立后端源码，不覆盖原 vendor |
| `/mnt/mydisk/xk/integrations/deformx-gpu-20261110/build-cp310-cu128-sm89` | 新建构建、日志和产物目录；实际构建系统由源码审计决定 |

官方 metadata 已冻结为 [工具链计划锁文件](../../source/deformx_cuda128_toolchain_plan.lock.json)：33 个 Linux x86_64 用户态组件、逐归档 SHA256/字节数/版本，总压缩字节数 **4,273,865,956**。明确排除 `nvidia_driver`、`nvidia_fs`、`fabricmanager`、`imex`、`libnvidia_nscq`、`libnvsdm`；不能无筛选下载/安装 manifest 全部条目。

该独立纯归档方案预计不需管理员权限，只写新建的用户目录。实际部署前仍须验证 fuseblk 文件系统的写入、链接和执行能力；元数据 W_OK 不等于这些操作已测试通过。所有 GUI/profiler 文件保留在 component 目录，不执行其安装程序或 postinstall。

## 3. 准入后执行：下载、校验与单独解包

**以下为待执行的 Linux bash，不是本次已运行的安装记录。** 使用当前用户 xk，不调用 sudo/apt/conda/pip 安装，不改 `/usr/local/cuda`、系统目录或已有 Toolkit。

- [ ] 首先创建全新 staging，目录或同名 symlink 已存在就停，不覆盖。
- [ ] 下载并冻结官方 manifest；不一致即停，不用重试掩盖版本变化。

```bash
set -euo pipefail
umask 077
test "$(id -un)" = xk
test "$(uname -m)" = x86_64
test ! -L /mnt/mydisk/xk/toolchains
stage=/mnt/mydisk/xk/toolchains/cuda-12.8.0-redist-staging
test ! -e "$stage"
test ! -L "$stage"
mkdir -p /mnt/mydisk/xk/toolchains
mkdir "$stage"
mkdir "$stage/downloads" "$stage/inventory" "$stage/components"
curl --proto '=https' --tlsv1.2 --fail --location \
  --output "$stage/redistrib_12.8.0.json" \
  https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.0.json
printf '%s  %s\n' \
  daa0d766b36feaa933592162c27be5fb63b68fc547ca6886c160a35d96ee8891 \
  "$stage/redistrib_12.8.0.json" | sha256sum --check -
python3 - "$stage" <<'PY'
import json, pathlib, re, sys
stage = pathlib.Path(sys.argv[1])
m = json.loads((stage / 'redistrib_12.8.0.json').read_text())
assert m['release_label'] == '12.8.0' and m['release_product'] == 'cuda'
allow = '''cuda_cccl cuda_cudart cuda_cuobjdump cuda_cupti cuda_cuxxfilt
cuda_demo_suite cuda_documentation cuda_gdb cuda_nsight cuda_nvcc cuda_nvdisasm
cuda_nvml_dev cuda_nvprof cuda_nvprune cuda_nvrtc cuda_nvtx cuda_nvvp cuda_opencl
cuda_profiler_api cuda_sandbox_dev cuda_sanitizer_api libcublas libcufft libcufile
libcurand libcusolver libcusparse libnpp libnvfatbin libnvjitlink libnvjpeg
nsight_compute nsight_systems'''.split()
assert len(set(allow)) == len(allow) == 33
rows = []
for key in allow:
    a = m[key]['linux-x86_64']
    rel = pathlib.PurePosixPath(a['relative_path'])
    assert not rel.is_absolute() and '..' not in rel.parts
    assert rel.parts[:2] == (key, 'linux-x86_64')
    assert re.fullmatch('[0-9a-f]{64}', a['sha256'])
    assert rel.name.endswith('-archive.tar.xz') and int(a['size']) > 0
    rows.append((key, str(rel), a['sha256'], str(a['size'])))
assert sum(int(row[3]) for row in rows) == 4273865956
with (stage / 'archives.tsv').open('x') as stream:
    for row in rows:
        stream.write('\t'.join(row) + '\n')
PY
```

- [ ] 对 33 项逐包下载、硬校验，再保存归档内容清单。不要直接运行官方示例的通用 download-all/flatten；已查看的示例 checksum mismatch 只打印信息，不能当作失败门禁。

```bash
set -euo pipefail
stage=/mnt/mydisk/xk/toolchains/cuda-12.8.0-redist-staging
while IFS=$'\t' read -r key rel sha bytes; do
  name=${rel##*/}
  archive="$stage/downloads/$name"
  test ! -e "$archive"
  test ! -L "$archive"
  test ! -e "$archive.part"
  test ! -L "$archive.part"
  curl --proto '=https' --tlsv1.2 --fail --location \
    --output "$archive.part" \
    "https://developer.download.nvidia.com/compute/cuda/redist/$rel"
  test "$(stat -c %s "$archive.part")" = "$bytes"
  printf '%s  %s\n' "$sha" "$archive.part" | sha256sum --check -
  mv -- "$archive.part" "$archive"
  tar --list --xz --file "$archive" > "$stage/inventory/$key.paths.txt"
  tar --list --verbose --xz --file "$archive" > "$stage/inventory/$key.verbose.txt"
done < "$stage/archives.tsv"
```

- [ ] 停下审查 inventory：确认各包唯一顶层目录、链接目标、相对路径和特殊文件，且不越出各自 component 根；之后才允许以下单独解包。**不能把列清单当作已完成这项审查。**

```bash
set -euo pipefail
stage=/mnt/mydisk/xk/toolchains/cuda-12.8.0-redist-staging
while IFS=$'\t' read -r key rel sha bytes; do
  component_dir="$stage/components/$key"
  test ! -e "$component_dir"
  test ! -L "$component_dir"
  mkdir "$component_dir"
  tar --extract --xz --file "$stage/downloads/${rel##*/}" \
    --directory "$component_dir" --strip-components=1 \
    --no-same-owner --no-same-permissions
done < "$stage/archives.tsv"
```

## 4. 完整 Toolkit 验收与 SDK 布局

- [ ] 依据真正下载到的 archive 内容确认 `nvcc`、`nvcc.profile`、`cudafe++`、`fatbinary`、`nvlink`、`ptxas`、`cicc`、`nvvm/libdevice`、compiler CRT、CUDA/CCCL headers、cudart/device runtime 和其余 Toolkit libraries 齐备。
- [ ] 在全新 `/mnt/mydisk/xk/toolchains/cuda-12.8.0/sdk` 内组装实际需要的 SDK 相对布局。保留原 component/license；相同目标文件先比较，不静默覆盖。新 symlink 只位于新 prefix 内；不改任何系统 symlink。
- [ ] 不借用 12.2/12.4 的 headers、cicc、ptxas、NVVM 或库补齐。缺文件时保留 staging，不能只凭 nvcc 或 runtime wheel 宣称 Toolkit 完整。
- [ ] SDK 真正存在后，核对其中 `nvcc --version` 为 12.8.61、内部工具与 headers/libraries 的归属，并检查 link/runtime 路径。

本次没有下载这些归档，不能提前证明它们的具体目录结构；因此不提供虚构的批量 cp/ln 合并命令。以上是安装计划与验收步骤，尚不是验证通过的全自动安装器。[NVIDIA tarball 与 CMake 导入说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html#tarball-and-zip-archive-deliverables)

## 5. cp310/cu128/SM89 构建合同（准入后才执行）

以下是后续配置值，**不是已执行的 CMake configure 或 Solver 构建命令**。源码到位后先核其实际构建后端与参数传递方式，不用 `CMAKE_ARGS` 环境变量存在冒充参数已生效。

```text
Python interpreter = /mnt/mydisk/xk/envs/dh116/bin/python
Python version = 3.10.12
PyTorch = 2.7.0+cu128
CUDA_HOME = /mnt/mydisk/xk/toolchains/cuda-12.8.0/sdk
CUDACXX = /mnt/mydisk/xk/toolchains/cuda-12.8.0/sdk/bin/nvcc
CUDAToolkit_ROOT = /mnt/mydisk/xk/toolchains/cuda-12.8.0/sdk
CMAKE_CUDA_COMPILER = /mnt/mydisk/xk/toolchains/cuda-12.8.0/sdk/bin/nvcc
CMAKE_CUDA_ARCHITECTURES = 89
TORCH_CUDA_ARCH_LIST = 8.9
```

- [ ] 只在单次构建子进程设置这些值，并明确 12.8 SDK 的 headers 与链接库路径。实际 CMakeCache、compile/link 命令和工具 provenance 必须证明未回落到 12.2/12.4。
- [ ] 使用当前解释器经 `/mnt/mydisk/xk/dh116/scripts/python.sh` 构建。若使用 PEP517/pip wheel，禁止依赖解析替换 Torch/NumPy/Isaac；实际命令须在 pyproject/CMake 审计后确定。缺失的 CMake/Ninja 等构建工具另置独立目录，不写入当前 DH116 site-packages。
- [ ] 核对构建系统实际采用的 Python 可执行文件与 Torch include/library 路径；CMakeCache 中对应的 Python interpreter 必须是当前 venv 路径，不能只凭同为3.10或环境变量声明判定正确。
- [ ] 扩展必须从源码重新编译并验证 cp310、当前 Torch CXX11 ABI 与 SM89；不复用 cp311/其他 CUDA 构建产物，不覆盖历史 checkpoint。
- [ ] 不改 `.bashrc`、activate.sh、原 python.sh、全局 PATH、ld.so.conf 或 CUDA alternatives。**不把新 SDK/lib64、lib 或 stubs 加进全局 LD_LIBRARY_PATH。**
- [ ] 由同一原项目启动器采集构建前后的动态库路径基线；构建后检查扩展 `readelf -d` 的 NEEDED/RPATH/RUNPATH，以及导入时 `/proc/<pid>/maps` 的实际库文件。若 Torch/runtime/driver 路径或版本偏离基线，停止验收，不用升级 Runtime 解决。
- [ ] 配置或编译失败保留完整日志，不通过升级原依赖、放宽检查或改用 12.4 冒充 12.8。

## 6. Runfile / 管理员权限备选

NVIDIA 官方的 Toolkit-only 用户目录候选命令如下。**本次未下载或执行；纯归档方案优先，不自动切换到该备选。**

```bash
sh /mnt/mydisk/xk/toolchains/downloads/cuda_12.8.0_570.86.10_linux.run \
  --silent --toolkit \
  --toolkitpath=/mnt/mydisk/xk/toolchains/cuda-12.8.0-runfile \
  --defaultroot=/mnt/mydisk/xk/toolchains/cuda-12.8.0-external-libs \
  --tmpdir=/mnt/mydisk/xk/toolchains/cuda-12.8.0-tmp \
  --no-man-page
```

该命令不含 `--driver`；官方 runfile 是 `https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run`，官方 MD5 为 `c71027cf1a4ce84f80b9cbf81116e767`，不是官方 SHA256。[官方 checksum](https://developer.download.nvidia.com/compute/cuda/12.8.0/docs/sidebar/md5sum.txt)

`--defaultroot` 不是全局写入 sandbox；官方文档描述创建 `/usr/local/cuda` 可能需要 root，已查参数没有可据以保证禁用该链接的开关。若需要管理员介入，必须先列出实际 installer 参数、拟写系统文件/链接与影响范围，等待用户确认；**不自行补 sudo，不装 Driver，不改系统链接或现有 Toolkit**。当前没有提出或执行任何管理员命令。[NVIDIA Runfile 参数与权限说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html#advanced-options)

## 7. 完成状态与验证层次

本次完成的是只读 Toolkit 路径复核、GitHub 身份确认、官方 metadata/安装方法核对与独立计划。没有新 Toolkit、没有 Solver configure/build、没有 GPU 或 Isaac 验证。

本地计划校验已执行：33项lock与官方manifest逐项相符，排除名单及4,273,865,956字节合计一致；5个bash代码块通过远端`bash -n`语法检查（UTF-8/LF stdin，不执行命令）；嵌入Python通过3.10静态grammar检查，并在本地仅使用已保存manifest生成33行TSV。没有下载Toolkit二进制、执行解包或安装。详细回执保存在本地`artifacts/deformx-20261110/cuda128-plan/plan-validation.json`。这些检查不是Toolkit/Solver编译验收。

后续依次验收：源码/嵌套依赖 → 归档 hash 与 SDK 完整性 → 12.8/cp310/cu128/SM89 编译 → 独立扩展导入与 runtime 保持 → 原生 GPU/PhysX/DH116 测试。不得跳过源码门禁，不得把安装器或语法检查成功写成 Solver/抓取成功；仍一次只运行一个 Isaac 实例。
