# 同一 episode 双手 Pinch：原24D控制与几何预检

基线：a4e8c2d19a7240beda5c42adee312972401ddc26。无新原生步、无PPO更新；旧左右单手记录未重跑。

## 实际修改
- prepare_dh116_bilateral_owned_pinch.derive_pair/load_pair：独立配置复用真实左右自身姿态、原Reset和自由杆；禁止将历史单手接触拼成双手证据。
- action_schedule/expected_boundary：逐维合成原左闭合和右接近，逐480Hz校验双侧关节目标及右侧-0.5mm根位移。29×24，其他腕平移及全部旋转为零。
- BilateralOwnedGeometry：完整冻结动作、物理时钟、请求/投影/SDK目标匹配；原碰撞、关节、Mimic、深度和跟踪保护不变。额外保存全部source17+cooked5顶点构成的整手分离下界，不能仅由掌心距离推断。
- audit_dh116_bilateral_owned_action_contract：实际调用原控制内核，CPU理想反馈逐232子步演算两侧执行目标；明确不是SDK实测。

## 验证
先运行24项缺失实现失败测试、3项控制审计缺失失败测试，日志保留Ubuntu私有目录。
新旧合同/几何42项通过；新增双手目标审计及新合同/几何27项通过（集合重叠，不相加冒充总数）。发布前重新执行45项去重回归。
命令：DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_bilateral_owned_contract.py tests/test_dh116_bilateral_owned_geometry.py tests/test_dh116_bilateral_owned_action_audit.py tests/test_dh116_right_approach_contract.py tests/test_dh116_right_approach_geometry.py

CPU演算：/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/bilateral-original-controls-20261114-run01/result.json
命令：bash scripts/python.sh -m scripts.audit_dh116_bilateral_owned_action_contract --output <独立新目录>
232行，投影0次；双侧关节目标误差最大5.7732e-15rad，根位置误差5.5511e-17m，本地/世界回环误差0。
整手静态分离下界最小318.207mm，原始全部碰撞形状桌面净空最小2.117711mm。
Reset SHA c9ef9bf12a65fb89d597ff6f4ad3e64dce09bdf3d74ff8b58717fa8fb95e3b69；
配置SHA 5d33440492525b2955e175aa8baeb8f8d1961fb8bc47a9a303545d6ef1c25553。
左右原独立全路径证书仍只证明对应静态路径；变形杆、实际跟踪和同刻接触须新原生验证。

## 尚未通过与下一包
本包不授予native/Teacher/grasp资格。右侧旧实验仍有极小非目标间隙，不能把CPU理想反馈当成实际安全余量。
下一包：独立原生入口、真实右父回执和左事后审计SHA绑定，保留原退出校验；冻结234步/180秒/1环境/0更新新配方。发布后按既有批准范围仅执行一次。
R0完整有限奖励、材料锚滑移、稳定承托及371D缺失来源仍待资格。234步不足5秒Hold。30mm/5s/0.5mm标准不变。
