# DH116 portable USD 依赖闭合

Baseline / Candidate: b5666c2 + dirty worktree。阶段：DH116 资产迁移；未修改 O6 运行代码或原始导入层。

原始警告根因：OmniPBR.mdl 是 Isaac 安装内材质库依赖，并非手部网格丢失。生成独立 portable 版，使用 UsdPreviewSurface，保留 diffuse_color_constant，其他原型默认材质无该值时采用灰色，粗糙度统一为诊断值 0.5。不宣称外观渲染完全一致。

首次仅处理常规 Material 漏掉 flattened prototype 内的 MDL Shader；失败产物保留为 portable_failed_material_check，不作为可用入口。之后处理所有 prim（包括 abstract prototype）上实际 sourceAsset shader，原始左右导入版不变。

生成并重新打开：

- `/home/xk/coding/M1/assets/dh116_v2/portable/DH116_left.usd`
- `/home/xk/coding/M1/assets/dh116_v2/portable/DH116_right.usd`

每侧 34 个带顶点 mesh、17 rigid bodies、11 revolute joints、36 材质 shader 转换。ComputeAllDependencies 每侧 1 layer、0 external assets、0 unresolved。manifest.json 记录 SHA 与 motion_authorized=false。左 SHA 31ba8d541058e89fc17e20af03c317499726b41f4a304f6c0af315a37dcf8f32，右 SHA 63d98413dd870ab58f0775170b47e1db45146db0ab9f24fa839d80c28bd3ad60。

验证使用 go2 IsaacLab headless GPU0、core0、数值库单线程；仅 USD 操作，无物理步。成功日志 kit_20261015_104427.log。退出码不能独自作为证据，已检查 CHECK 和 PORTABLE_COMPLETE 输出。

仍阻塞动态型号确定：URDF 四指线性耦合与 MJCF 二次项在 q=1.57 时相差 0.200535*1.57²≈0.4943 rad（28.3°）；实际力矩/速度上限缺失，拇指零态与从动限位矛盾。用户批准结构转换不等同批准任意动力学近似。下一步需明确仿真以哪份源为基准，或者补厂家关节耦合/驱动数据；不能默默继承 O6。

双 Panda 安装、DH116 Jacobian、指腹接触、开合与抓握仍未验证。相关父节点 T500/DH116 迁移 OPEN。
