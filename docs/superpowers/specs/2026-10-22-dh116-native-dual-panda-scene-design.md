# DH116 原生双 Panda 场景重建设计

## 目标

重建一个干净的 Isaac 场景，使其稳定显示并验证参考图中的目标姿态：双 Panda、左右分开、掌心向下、手指朝前、wire 位于桌面上。动作从已验证的 close 状态开始，按 `close -> hold -> lift` 执行，不重跑 approach。

## 资产边界

- Panda 使用 Isaac/IsaacLab 原生 Franka/Panda articulation，左右各一个独立实例。
- DH116 使用厂家 DH116 V2.0 左右独立 USD，分别以独立 articulation 根加载，并把整个 DH116 根 prim 对齐到对应原生 Panda `panda_hand` 世界位姿；不再加载 `fixed_dual_panda_dh116.usd`，也不创建嵌套 articulation。
- 桌面和 wire 是独立场景物体，不嵌入 Panda 或 DH116 USD。
- 停用逐 link 的 `visual_pose_sync`。关节状态由各自 articulation 写入，安装关系只通过 DH116 根 prim 的显式挂载变换同步；不修改 DH116 子 link。
- 左右关节按本侧名称映射；禁止把一侧的关节数组复制到另一侧。

## 姿态与动作

1. 使用 `runs/forward_feasibility3/pose.json` 对应的 candidate 3 作为初始关节状态；它必须按用户提供的 `pose_preview.png` 显示为双 Panda、掌心朝下、手指朝前。该姿态目前只是离线几何候选，不能预先标记为物理通过。
2. 启动后先进行姿态门禁：检查左右 Panda、左右 DH116、安装关系、掌心法向和手指方向；不通过则停止。
3. 直接从保存的 close 状态进入闭指；不执行 approach。
4. hold 阶段保持接触和承托，Panda 不漂移。
5. lift 阶段由 Panda 双臂同步沿世界 +Z 抬升，DH116 维持抓握稳定和接触负载，不生成新的奇异姿态。

## 物理验证标准

- close：四个目标指腹点连续满足接触窗口至少 100 ms。
- hold：接触和负载稳定保持 1 s，滑移不越界。
- lift：双臂同步抬升至少 30 mm，并保持 1 s；同时检查整根 wire 离桌间隙和整体滑移。
- 每阶段失败即停止并保存当前阶段状态、诊断和轨迹；不自动创建第二个 Isaac 实例，不回退到 approach。
- 输出必须区分离线几何检查、Isaac 物理验证和完整抓取成功。

## 实现边界与测试顺序

实现拆成四个独立组件：

1. 原生双 Panda 场景加载器；
2. DH116 左右 USD 固定挂载器；
3. candidate 3 姿态映射和 close/hold/lift 阶段入口；
4. 阶段证据与失败恢复记录器。

验证顺序固定为：无 wire 资产/关节检查、桌面+wire 静态姿态检查、close、hold、lift。每一步都生成机器可读 JSON，并保存可见截图；未通过前一阶段不得宣称后一阶段通过。

## 非目标

- 不删除或覆盖现有 USD、运行记录和旧分支资产。
- 不修改 DH116 厂家额定驱动力矩/速度。
- 不把视觉回放或离线几何结果当作物理抓取成功。
- 不在本设计中重新训练 teacher/student 或修改 wire 材料模型。
