# 柔性 Teacher 首帧双 RGB 与特权坐标对齐

新入口 scripts/diagnose_dh116_teacher_stereo_alignment.py 在现有 Pinch 初始化诊断同一场景加载 Student 使用的 stereo_camera_front_640x480_v3.json。采用两台固定的 RTX TiledCamera 及同一渲染边界。分辨率 640×480，基线 80 mm，使用原始 Student 标定 SHA。记录实际 SDK 内外参，并以现有镜头读回契约审核。这个首帧方案只处理一个环境。

源特权状态来自 GPU Teacher 原版 get_obs() 的 67D 输入。代码从这个向量反解两个掌位和五个选定线缆节点，与同边界的实测 base_link 姿态及 DeformX 节点校验。另一个 NativeTeacherCollector 使用 371 槽状态账本及不同模型入口，本工作包尚未接入该收集器。保存两个 RGB PNG 及其 SHA、原始 67D、41 个线缆材料节点、两个实测掌 link 姿态和同一状态摘要。坐标标签含世界、左右掌局部、左右光学坐标深度及亚像素投影。图像内掩码只表示投影落在画幅，不能判断遮挡。特权记录独立于 RGB。

读取、渲染和读回均受 SDK 与 DeformX 物理时钟约束，同一状态摘要在渲染边界前后比较。显示线缆以同帧材料节点生成视觉曲线，Student 仍使用自己的可视网格。已有 cooked 原始数据先写盘。相机资源在原清理流程前关闭，原严格几何错误照常记录。该计划未加载 Teacher 权重，模型查询、控制、求解器和 PPO 更新预算均为 0。

CPU 回归 591 项通过，专项 129 项通过。新增测试覆盖标定、左右投影、环境平移不变性、图像时钟与状态来源错配、从实际 Teacher 输入布局反解坐标、独立冻结指纹。一次受控实机首帧仍待执行，连续轨迹和 Student 训练资格仍未验证。
