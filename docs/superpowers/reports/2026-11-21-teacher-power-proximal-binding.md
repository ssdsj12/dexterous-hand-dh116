# Power 双侧小指近端碰撞接入预检

在原两端支撑、原掌面及指垫和单根自由任务杆的布局之后，按左、右追加各自的 `finger52_link`，总计 17 个碰撞体槽。原 15 个槽编号不变，只有按独立诊断开关运行才启用。读取 0.5 mm SDF 前核对原生捕获、烘焙结果和双侧数组的 SHA，并用运行时刚体路径、碰撞体归属、原 mesh-to-body 和几何摘要逐侧检查来源。每侧刚体力和力矩现在可按第 7 个绑定逐行路由。掌面和五指原有资格条件继续执行，新增近端接触独立入账。

`python -m scripts.verify_dh116_power_supported_task --verify-near-open-settling --proximal-capture CAPTURE --proximal-bake BAKE ...` 冻结成功。单次预算 1 环境、2 个初始化 SDK 步加 238 个测量步、2856 求解器子步、1 次 Reset、0 控制/训练步、180 秒软上限。保留第一个求解器交换的静止位姿绑定。计划从原 15 槽同初态验收作对照，逐行检查支架、桌面和手部的载荷，再确定可执行的靠近路径。

`python -m pytest -q tests/test_dh116_power_supported_task.py tests/test_dh116_power_palm_contact_evidence.py tests/test_dh116_power_palm_raycast.py tests/test_dh116_deformx_pad_sdf.py tests/test_dual_hand_rod_runtime.py tests/test_dh116_work_packages.py` 通过 126 项。反向夹具覆盖任一侧文件被替换、网格间距不符及左右错位。当前记录仅为 CPU、真实证据文件和冻结预检。暂无 17 槽原生任务读回，本工作包亦未产生抓取证据。
