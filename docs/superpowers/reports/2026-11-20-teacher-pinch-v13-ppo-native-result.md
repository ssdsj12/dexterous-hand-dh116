# Pinch v13 有界 Teacher 原生结果

Ubuntu 实测提交 `9186e65e6ce21a37fb66fa944ea5ab130a81d0e2`，尝试指纹 `b22bc5fb2aca4c2b65ff7e3d783c80189f574f4d1af48438c7c5f0385d56b899`。Registry 启动前发现 v13 的独立 CPU 包缺少 dependencies 声明。补齐后运行真实调度器预检，505 文件执行指纹不变，Registry 的 60 项 CPU 回归通过。启动前错误记录在独立日志，没有启动 Isaac 或消耗配方执行次数。

## 原生执行与终止

GPU 3，单实例，1 环境。2 个 SDK 初始化步，116 个测量 SDK 步，1392 个求解器子步。共 14 个完整控制窗和 1 个在第 4 个物理子步终止的控制窗。左右初始净空分别为 2.3372216 和 2.6531715 mm。SDK 117 右手最低净空 2.102741 mm，SDK 118（控制索引 14，子步索引 3）右手 `finger13_link` 原始碰撞包络降到 2.096886 mm，低于 2.100 mm 保护线 3.114 µm。该步原手腕投影原因左右均为 0，实测右腕跟踪误差 0.019153 mm。账本关联了 24D 请求、下发与实际读回、左右桌面净空和终止时钟。仅凭此行不能区分姿态目标造成的最低点下降与 SDK 动态跟踪贡献。

终止 transition 保留，GAE bootstrap 和跨回合 carry 均关闭。现配方只允许一次 Reset，训练记录 `fresh_verified_reset_factories_required` 后停止，没有更新或 clean checkpoint。App close 返回，外层与 GDB inferior 均为自然退出 1，未见 SIGSEGV。墙钟 184.173 秒。账本保留 57072 个 solver contact pair 子步记录，按 mesh 的实际计数见随附 JSON。原 PhysX pair 字段缺失，不能据此授予双侧力、抓取或 Teacher 资格。

## 独立计时复读

v12 的 10 个完整控制窗中位数 6.845 秒，v13 的 14 个完整控制窗中位数 4.484 秒，下降 34.5%。这是两次原生尝试的描述性对照，Reset 和轨迹不同，无法独立归因到编码或复制优化。v13 唯一更新启动前读回剩余墙钟 75.431 秒。按完整窗实测中位数估算需约 143.5 秒采集 32 个控制窗，另需更新和封存时间。即使处理净空终止，仍需先解决剩余墙钟不足。保护线、物理和 30 mm / 5 s / 0.5 mm 的成功门槛保持原合同。

66 项 CPU 回归通过。原始账本 SHA `3b46bc46f0b615eca40e26c8bafec9c5df9b873cb89ddaf45b171889f5cf2b7b`，生命周期日志 SHA `25bb69a94d3758db1db6d7e9ad350898a9eaa59d72a881e15c8abc61d9ec0aa9`，外层退出 SHA `5b582cdc617522619d056785e40e49e31aab02d1b888fb450ae84d1d8b199d00`，执行回执 SHA `e8a9c8c735f20714aa59d557792fa268705c32324b1ca11bece84627de8d0d72`，计时复核 SHA `405f8725199a5ab4fd1dcf57f4a43be9d6f135731c046417ae4836c01f6db315`。实际测试日志 SHA `77120be136f0da5a466f4d905f3bb607a7df52a8b3932baa28d011c704a97a5f`。历史原始数据留在 Ubuntu 独立运行目录。
