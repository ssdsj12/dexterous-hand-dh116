# F1 独立 CUDA 12.8 编译

在 Ubuntu 22.04.5 上使用已核验的 DeformX CUDA 提交 `96b8514c7c55c724f1abba1fbf6ed6642b7db72b`。原 DH116 Python 3.10.12、Torch 2.7.0+cu128、NumPy 1.26.4、Isaac Sim 4.5、Isaac Lab 0.54.4 和 Driver 580.178.04 都沿用现有安装。工具仅安装在用户私有目录。系统 `/usr/local/cuda` 继续指向 12.2。

独立 SDK 由 NVIDIA `cuda-12.8.0` channel 的 13 个编译相关包组成。锁文件逐项保存包字节数及 SHA256。SDK 包含 `nvcc 12.8.61`、CRT、NVVM、CCCL、cudart 12.8.57 和二进制检查工具。范围是编译该 Solver 所需的 SDK。CMake 3.31.6、Ninja 1.11.1.4 和 pybind11 2.13.6 仅解包安装到私有 build-tools。主机编译器 G++ 12.3.0，GLM 0.9.9.8。CMakeCache 确认 CUDA 编译器、cudart 和架构 `89` 都来自指定 SDK。源码仓库未改动。

初次 SDK 解包遇到发行版 tarfile 将相对软链接按归档根解析。改为分开验证和恢复按成员父目录解析的相对链接。配置尝试 1 缺少 nvcc.profile 在目标布局中查找的 `cicc`，尝试 2 缺少 `cuda-crt-tools` 内的 `bin/crt/link.stub`。两个日志均保留。修正目标目录布局并加入已锁定的 CRT 包后，先编译最小 SM89 程序，再完成 12 步 Solver 构建。上游格式化字符串告警仍保留。再次使用仓库准备脚本从锁重组 SDK，独立配置及构建同样通过。

| 核验 | SHA256 / 结果 |
|---|---|
| 构建输入锁 | `89d65e89a88ec6c8f195a8faa5898f0c5da73db183e11c81abbf0c579d87ea27` |
| 首次配置失败日志 | `f5e44e08510e282214f015f5e98ad7cac8e4de05262d33a38ea414544c143413` |
| 第二次配置失败日志 | `ec34e4b48d06bb7eb570e05015e459d97364325978333a2736b67c770bce4cb8` |
| 首次成功构建日志 | `3e76b3f4f511ab74141df2fca5ff67a2e51df8e004741e74de0387d73fc4980d` |
| cp310 原生扩展 | `c47942c4b15b3dfedc47212e71f8193825c0347df3c1d73fb76b648f7bdedb53` |
| 独立 SDK 重建扩展 | `ea07c29341fcac91236b2d9065f7d7b8ee5e21cc69adbf181fa211fdaa9d44d3` |
| 最终 SM89 机器码 | cuobjdump 读到 7 个 `sm_89` cubin |
| Ubuntu CPU 回归 | 18 通过，0 失败 |

两次构建输出目录不同，扩展 SHA 也不同。两次均核验了 cp310 文件名和 SM89 设备代码。本报告记录编译层验收。GPU 数值运行及 PhysX 耦合使用后续独立证据。

## 复现

将锁中的 NVIDIA 包和三个工具 wheel 校验后分别放入 `$archives` 和 `$wheelhouse`。使用独立的新输出目录和同一固定后端 checkout。

```bash
python=/mnt/mydisk/xk/envs/dh116/bin/python
"$python" -m pip --isolated install --no-deps --no-index --target "$private/build-tools" "$wheelhouse"/*.whl
"$python" scripts/prepare_dh116_deformx_sdk.py \
  --lock docs/contracts/dh116_flexible_cuda_build_inputs.json \
  --archives "$archives" --output "$private/toolchain"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$backend"
export CUDAHOSTCXX=/usr/bin/g++ GLM_INCLUDE_DIR=/usr/include
"$python" scripts/build_dh116_deformx_cuda.py \
  --source "$backend" --source-commit 96b8514c7c55c724f1abba1fbf6ed6642b7db72b \
  --toolkit "$private/toolchain/sdk" --output "$private/build" \
  --cmake "$private/build-tools/cmake/data/bin/cmake" \
  --ninja "$private/build-tools/bin/ninja" \
  --pybind "$private/build-tools/pybind11/share/cmake/pybind11"
```

Ubuntu 私有证据目录 `flexible-f1-20261112-883RS4` 保留构建材料、完整日志、两个产物和原始包。源码及历史 Checkpoint 核验通过。仓库发布范围为脚本、测试、输入锁和报告。
