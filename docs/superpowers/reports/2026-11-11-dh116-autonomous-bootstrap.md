# DH116 Ubuntu 工作包调度器

基线 GitHub SHA 3c80166da75fb5496701c9f3ff673296c5db5615。原始服务器项目 /mnt/mydisk/xk/dh116 位于 11-3，原有本地修改保留。新集成与六个任务工作区均在 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166 下。FUSE 挂载不支持 Git 初始化的 chmod，因此使用 /home/xk/.local/share/dh116-autonomous-3c80166 中的 ext4 Git 元数据。Git 历史来自已核对的开发分支与原始仓库只读祖先对象。

Ubuntu Python 3.10.12 CPU 验证 14 项通过，0 失败。包含依赖拒绝、失败日志、允许发布路径、断网后继续同一提交，以及待发队列验证后才进入清单。保护的 528 个源文件均与 11-10 的 Git 内容相同，Ubuntu 字节 SHA 库绑定当前检出。两份历史 Teacher Checkpoint 按原始路径只读核对，原生 Isaac 与训练均未启动。

Ubuntu 承担编辑、测试与提交。服务器 GitHub HTTPS 当前返回校园门户的过期证书。GitHub 推送采用用户指定的本机 Git 裸仓库中转，逐个核验 Ubuntu 提交与远端。未推送结果保留在私有队列，发布清单不写入未经读回的提交。受保护的 11-10 与 Draft PR #2 保持原有发布约束。

运行命令：`python -m pytest -q -p no:cacheprovider tests/test_dh116_work_packages.py`。下一步集成待发布的 WP1、WP2、WP6，并运行 WP3、WP4、WP5 的独立 CPU 配方。
