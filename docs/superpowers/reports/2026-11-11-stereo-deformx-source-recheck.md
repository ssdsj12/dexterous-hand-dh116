# DeformX CUDA 来源复查

状态为 `blocked`。目标提交为 `96b8514c7c55c724f1abba1fbf6ed6642b7db72b`。服务器只读复查覆盖 22 个项目根和 19 个 DeformX 候选，未发现候选内独立求解器 Git 工作树、CUDA `.cu/.cuh`、`.gitmodules` 或已检出 Gitlink。搜索未触及目录或时间上限，无读取错误。原始清单留在本地，SHA256 为 `7910993e311c6fd421d93d9c8aa139926eb0329950ea6736be20c45c16a81dc4`。搜索边界沿用此前的项目及依赖目录发现脚本。

新增 `scripts/verify_dh116_deformx_cuda_source.py`。取得候选路径后，该命令只读核验独立工作树、当前固定提交、递归子模块、未提交修改和实际 Git 跟踪的非空源码。清单同时列出 CUDA、CMake、Python 构建元数据、可能的绑定入口及许可证。绑定特征与许可证文件存在只是供后续审查的证据，不能证明 API、构建、使用或再发布资格。用该工具检查服务器现有 `vendor/DeformX`，结果是 `not a standalone Git checkout`。不修改该目录。

服务器实际解释器为 `/mnt/mydisk/xk/envs/dh116/bin/python`，Python 3.10.12，PyTorch 2.7.0+cu128，`find_spec('ropesim')` 为 `None`。GPU 3 是 RTX 4090、SM89。标准 `/usr/local` 路径仍仅找到 Toolkit 12.2 和 12.4。`cu128` 是 PyTorch 的构建/runtime 信息，不能证实 CUDA Toolkit 12.8 存在。

CPU 验证 `python -m pytest -q -p no:cacheprovider tests/test_dh116_deformx_cuda_source.py` 共 2 项通过。临时 Git fixture 验证空路径、固定提交、可见 bindings/kernels、构建文件及删除源码后的阻断。这些 fixture 不计入原生证据。

解除阻断需要可读的真实求解器仓库及指定提交、完整递归子模块、bindings、kernels、构建文件和许可证。收到材料后先运行只读清单并审查许可证与 Python 3.10/PyTorch C++ ABI，然后才制定独立 Toolkit 12.8/SM89 构建执行。若需要安装 Toolkit，使用独立路径，不替换现有 Driver、Runtime、Python、PyTorch 或 Isaac，按资源和权限门槛另行批准。编译、GPU 求解和 Rod/Isaac 双向耦合均尚未执行。
