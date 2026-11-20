# Power 右掌原始三角网 SDF

复用已封存的右掌三角扇和原 DeformX 烘焙路径。先用独立有符号距离确认顶点均值在内部。实际上游 weld 后，37 个顶点、70 个有向三角面均保留，重新通过自交检查。三角面仍使用原固定扇形三角化。

CUDA 在 RTX 4090 独立创建一个 RopeBatch，右掌槽位 0，wrench buffer 为 [1, 1, 6]。1 mm 体素，110 × 79 × 153，零 Isaac、物理和求解器步，自然退出 0。与点到原三角面的独立 float64 距离和固体角符号比较得到：

- 212 个面心、边中点及表面顶点，P95 / 最大绝对误差为 0.262 / 0.328 mm。
- 840 个原三角面法向偏移样本，范围 ±0.025 至 ±5 mm，符号一致。±5 mm 组最大误差 0.270 mm。
- 样本误差最大点位于原顶点。结果属于这些查询样本，未建立整个网格的误差上界。

下一步需对运行中的原 PhysX 右掌使用相同链接坐标的定向查询，核对原面与物理碰撞面的关系，再验证掌面和五指共同受力。Power 正式环境和 Teacher 准入仍未通过。

运行 `python -m scripts.bake_dh116_power_right_palm_sdf --capture <verified-capture> --sealed-audit <verified-triangle-audit.json> --ropesim-package-root <locked-backend> [--output <new-directory>]`。109 项工作包 CPU 测试通过，原生结果 SHA 见 readiness。
