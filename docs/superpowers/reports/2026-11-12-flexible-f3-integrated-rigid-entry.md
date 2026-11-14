# F3 原刚体命令的集成原生复验

Ubuntu 集成工作树运行原入口 `scripts/verify_dh116_deformx_rigid_bridge.py`，使用 `--steps-per-phase 120`、实际 GPU 3 和固定的 F1 cp310/SM89 扩展。该入口现复用已验证的 E 生命周期。控制与动态各 120 步，合计 240 条原生样本。应用返回自然关闭标记，Bash 外层退出码 0。

独立 `cmp` 确认该命令的实际物理账本与 E/E2 和已知旧退出 139 的同物理对照逐字节相同。物理账本 SHA256 为 `ae3934bd61f2a98ef8d267f102c1c2a48e17326dd5372294d171a1908d97e206`。关闭回执 SHA256 为 `80a042b70b8ed0ae5955d671080ede69f15366e01496881dfa81ed62d8d59752`。启动日志 SHA256 为 `a25fcfc5f0f4be1f06b01e9bf57aeec6405803b4b7fa8a6d09f44b120eacf8f2`。原始证据留在 `flex-f3-integrated-entry-20261112-IdCvxP`。

该复验覆盖原命令实际执行路径及单刚体生命周期。后续真实 DH116 接触和 F6 AppWindow 自然退出依照各自原生证据登记。
