# Windows 发布与 Ubuntu 队列

基线 `600b74bc930a56118f18616d6f0a0fe34b2690f9`。Ubuntu 的工作包调度入口始终走本地提交路径。`--plan` 读取本地已核验的远端跟踪引用。`--run` 与 `--prepared` 核对本地 HEAD 后验证并排队。`--resume` 恢复本地提交事务，或者报告等待 Windows 发布的队列。旧的直接发布事务保留供核验。

Windows 裸仓库负责 GitHub push、fetch、提交和 PR 身份核验。`--relay-plan` 与 `--relay-ack` 保留两阶段回执合同。最后一次回执经本地祖先关系和 compare-and-set 检查后更新跟踪引用，再清除队列。

Ubuntu 回归命令 `python -m pytest -q -p no:cacheprovider tests/test_dh116_work_packages.py`。新增检查在修复前复现 6 项失败，修复后 22 项通过，退出码 0。日志 SHA256 为 `1c51da291d533fcfcdae0c4c758ad9b126436744ba41ecd37b8a428ec809280e`。测试使用本地裸仓库，检查计划、排队、异常恢复及第二阶段回执。原生实验由既有门槛管理。
