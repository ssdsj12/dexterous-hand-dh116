# Pinch 非零动作中的运动学/跟踪分离
基线 0779cf8d300faccce9ae7de1ca9f7cd7b98bd41a。现有零动作FK核验不代表非零动作；已有16步完整实际link姿态可在不新增物理预算下进行对照。
1. 先失败测试。复用原USD joint frames和实际左右独立来源，禁止修改旧FK/控制器。
2. 同一实际base pose下，比较实际11个关节FK、6个active加原mimic FK、SDK active target加原mimic FK与实际link transforms。
3. 每个env/episode/reset/policy/physics/timestamp保留位置/旋转误差、q/qd/targets和mimic残差，隔离坐标模型、被动联动、目标跟踪三部分。
4. 对四个必要pad保留actual SDF和三个模型SDF；均为几何，不是contact。无隐含接触预测资格。
5. 只读固定实际receipt/账本，独立输出SHA，CPU回归、报告、Registry、Ubuntu commit、Windows同commit发布。
不新增几何搜索、不改变动作/参数、不运行Isaac，不从16步短轨迹推断长期稳定或Teacher资格。
