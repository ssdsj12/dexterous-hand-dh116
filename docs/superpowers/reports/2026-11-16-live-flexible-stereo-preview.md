# DeformX 实时柔性杆与双 RGB 预览：代码集成

基线 07b848d4b8b0dc77c945b17a15f03c86165f9e57。独立预览路径复用当前双手物理构建、DeformX 节点与材料朝向读回及 WireMeshPublisher；在独立限定控制边界更新显示，再以一个 tiled RGB render product 读取左右相机。reset 帧明确标记 reset，后续帧按原 physical exchange 标识；display skin 不是碰撞体，输出图像不附带训练用特权输入。原控制路径和资产未变。

渲染守卫核对 SDK 与 rod 时钟、相机 frame、标定和分辨率；检测到超额、状态变化、缺失来源即拒绝。默认 640×480，支持校准缩放/重新定义镜头。preview 明确 0 PPO、零 24D 请求，使用 singleton lock，不进入成功池。相机资源代码做所有权检查并调用已安装版本的 detach/destroy；若构造内部提前失败无法获得所有权，不虚报清理成功。

本包按 Ubuntu DH116 Python 运行相关 CPU 测试（115 新接口及既有相机测试、加原双手控制/Teacher session 回归，合计 144 passed）。测试使用 CPU fixture，不能证明实际 USD、GPU 动态物理、双目图像、WebRTC、接触、抓取或自然关闭成功。对 CameraView 部分构造失败、Kit 内部渲染、进程阻塞时 180 秒协作截止不构成硬超时，仍需原生证据。

下一步：发布后在目标服务器的实际源码树重新生成冻结合同，核对 source、experience、backend 与预算，然后向用户请求本次单实例短时原生预览授权。未授权前不启动 Isaac 或训练。成功标准保持 30 mm / 5 s / 0.5 mm；本预览不评估任务成功。
