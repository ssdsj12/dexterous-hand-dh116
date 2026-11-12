# F0 DeformX 柔性后端源码锁定

开发基线为 DH116 `4ef3faca8550a6c6280a05f4622a91bf5deb2bf1`。从官方 DeformX 的 `refs/heads/main` 读取主仓库和 Gitlink。Ubuntu 验证目录为 `/mnt/mydisk/xk/projects/deformx-f0-04f6a11/parent`，与 DH116 工作树分离。

| 仓库 | 固定提交 | 源码许可证 |
|---|---|---|
| DeformX | `04f6a11eb4d832df761f631cf6ce1bc00da75eb1` | MIT |
| Cosserat-Rod-Solver-CUDA | `96b8514c7c55c724f1abba1fbf6ed6642b7db72b` | 包元数据声明 GPL-3.0-or-later |
| PyElastica-Mesh | `a5cfaf60c4a1aa707ed94bdd07641de1a90ae650` | MIT |

后端包含七个 `.cu`、两个 `.cuh`、两个构建文件、两份许可证和 `csrc/bindings.cpp` 的 pybind11 模块。后端 Git 树跟踪 `third_party/yarnball/` 源码及 GPL-3.0 文本。该目录的 `VENDOR.md` 记录 YarnBall-Linux 上游提交 `fafa4fc8b323eb335679f03857bbf1d3fbefd9ce` 及本地补丁。主仓库 MIT 和后端 GPL 许可证分开记录。DH116 交付仅包含锁文件、验证器、测试和报告。

## 本地材料与运行方式

Ubuntu 直连 HTTPS 返回主机证书不匹配，TLS 握手实际证书主体为 `net.szu.edu.cn`。经 Windows 用证书校验下载官方 Git 对象，Ubuntu 按 SHA256 校验转发文件。仅对本次验证的三个已知路径设置进程内 `safe.directory`，以处理该挂载对文件所有权的映射。

```bash
p=/mnt/mydisk/xk/projects/deformx-f0-04f6a11/parent
export GIT_CONFIG_PARAMETERS="'safe.directory=$p' 'safe.directory=$p/Cosserat-Rod-Solver-CUDA' 'safe.directory=$p/PyElastica-Mesh'"
export PYTHONDONTWRITEBYTECODE=1
git -C "$p" submodule status --recursive
/mnt/mydisk/xk/envs/dh116/bin/python scripts/lock_deformx_source.py "$p"
PYTHONPATH="$PWD/src:$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_deformx_source_lock.py tests/test_dh116_deformx_cuda_source.py
```

锁文件固定三个 Git 树、子仓库 Gitlink、源 URL，以及 CUDA 内核、构建文件、绑定、桥接和许可证的文件 SHA256。验证器要求干净工作树、全部 Gitlink 已物化，复用现有 CUDA 材料审计排除符号链接和 LFS 占位。CPU 测试覆盖缺少 Gitlink 锁、许可证篡改和仓库 URL 不符。

实际递归状态显示两个 Gitlink 均在指定提交，三个 Git 树均通过 `git fsck --full --strict` 且状态干净。CPU sibling 为单提交浅克隆，固定提交的 331 个树条目均在本地。CUDA 后端和父仓库具有非浅历史。Windows 的初次 CPU Git 传输中断，改用带 TLS 校验的定点浅克隆完成，未替换 CPU 源码。

| 校验材料 | SHA256 |
|---|---|
| Windows 转发的主仓库 Git bundle | `1b1d0b6dd32133e3f8b4a6f53c808348bea6a73eb0b0bce946ab1ee2d6f52b04` |
| Windows 转发的 CUDA Git bundle | `738a1236dfb4a775305bc60be284c7c0eb106c0416d78365388b74208f2af0cf` |
| Windows 转发的 CPU 浅仓库归档 | `32a16a746b558ca8b1f644747f0b5a6e602f76235ebd949d6deadb8a0f2d5cf2` |
| Ubuntu 实物验证结果 | `837297090288d819930d82a66a98e334d1bb4205d923e0979640b5d00258e6d3` |
| Ubuntu CPU 回归 | `61e382cc24abc6f9ef372e607127032e5cca5fd704814e59a69375246ec9da6d` |

CPU 回归 9 项通过，0 项失败。真实验证结果为 `source_locked=true`。528 个受保护文件及 2 个历史 Checkpoint 核验通过。Git 转发包和原始证据保存在 Ubuntu 验证目录。

## F1 边界

源码中的最低 CMake 版本为 3.24，编译使用 C++17、CUDA、OpenMP、GLM、pybind11 和 scikit-build-core。Ubuntu 默认工具读回 gcc/g++ 12.3.0、CMake 3.22.1，默认 PATH 无 Ninja。系统路径可见 CUDA 12.2 与 12.4 的 nvcc。F1 独立核验 Toolkit 12.8、SM89 和兼容工具链。本工作包只核验源码和 CPU 测试，CUDA 编译、kernel 执行和 Isaac 双向耦合均待相应工作包。
