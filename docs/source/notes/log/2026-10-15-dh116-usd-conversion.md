# DH116 左右手 USD 结构转换

Baseline / Candidate: b5666c2 + dirty worktree。资产位于用户指定的主仓库 assets，不在 worktree 内。未修改 O6、控制入口或源文件，未提交上传。

使用 go2 环境 IsaacLab UrdfConverter，headless GPU0，taskset core0、OMP/MKL/OpenBLAS 单线程。输入桌面左右 URDF；固定基座、保留固定关节、保留 URDF 线性 mimic，禁用驱动（target none、零增益），开启自碰撞、convex hull 诊断碰撞几何。没有执行物理 step。

输出：

- `/home/xk/coding/M1/assets/dh116_v2/left/DH116_left.usd`
- `/home/xk/coding/M1/assets/dh116_v2/right/DH116_right.usd`
- 同目录 manifest.json：源树/输出 SHA、结构检查、警告及 motion_authorized=false。

两侧重开 USD 并遍历 instance proxies：各 34 个有顶点网格、17 rigid bodies、11 revolute joints、metersPerUnit=1。首次普通 Traverse 计数零的原因是未遍历实例内部，非源网格丢失。

USD 依赖检查仍报告 OmniPBR.mdl 未解析；各几何/物理分层均位于输出目录。当前只通过结构几何检查，未完成无外部依赖资格或渲染检查。不得把未解析材质忽略为完整验收。

仍未完成：驱动限值、拇指零态限位冲突、URDF/MJCF 耦合一致性、Panda 安装、DH116 Jacobian 与指腹标定、真实抓握。禁止直接复用 O6 参数或权重。下一步处理材质依赖并建立 DH116 控制合同。

Isaac 日志：kit_20261015_104040.log（首次检查拒绝）、kit_20261015_104116.log（左右结构检查通过），位于 go2 Isaac-Sim/5.1 日志目录。
