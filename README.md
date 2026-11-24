# DH116

唯一开发、构建和运行目录：/mnt/mydisk/xk/dh116。开发分支：11-17。

当前包为src/dh116/：24个实现模块、6个包初始化文件及2个CUDA源文件，按env / physics / rl / hand分层。底层物理、接触、控制、几何守卫与生命周期由Teacher和Student共用。模型使用eager执行，关闭torch.compile。

## 环境与入口

~~~bash
cd /mnt/mydisk/xk/dh116
source activate.sh
python -m pip install -e . --no-deps --no-build-isolation
bash scripts/python.sh -B -m pytest -q tests
~~~

当前已安装环境：Python3.10、Isaac Sim4.5、IsaacLab0.54.4、PyTorch2.7.0+cu128、Warp1.12.1；原有环境未升级。CUDA12.8工具链与编译产物位于本地忽略的build/。

| 入口 | 用途 |
|---|---|
| scripts/python.sh | 当前Python/SDK库路径与线程设置 |
| scripts/build_native.py | ropesim、FP64几何核构建与真实flat接触对照 |
| scripts/prepare_scene.py | 实测场景/SDF/Reset数组及资产SHA验证 |
| scripts/train_teacher.py | 无相机共享物理Teacher PPO及有界控制窗验证 |
| scripts/train_student.py | 封存数据的状态监督/BC，保留独立初态持出与准入 |
| scripts/run_student.py | Actor-only双RGB部署推理 |

构建使用vendor/ropesim和当前源码；build/scene_bundle保存真实SDF、完整凸包与独立的计划初态/实测Reset，不上传原始数据。新克隆需配置经过SHA核验的本地输入，不会自动伪造几何或监督。

Pinch仅以拇指/食指作为目标SDF；其余三指仍保留完整非目标守卫。Tripod/Power保留独立配置。30mm Lift、5s Hold、0.5mm Slip、0.15mm穿入门槛、2.1mm桌面保护线保持原值。

## 验证与资格

当前验证摘要见configs/runtime/readiness.json。有界控制窗通过不授予抓取或完整回合成功资格；正式Student训练仍需合格Teacher与双目来源，R2和未验证Force Proxy保持关闭。

测试覆盖PPO/Adam/RNG恢复、原网络数值基线、4096样本KL、局部Reset隔离、真实flat接触规约、1µm守卫和SDF边界回退、Student信息隔离及固定双RGB时序合同。

## 源码树

~~~text
src/dh116/__init__.py
src/dh116/cuda_measurements/__init__.py
src/dh116/cuda_measurements/hull_support.cu
src/dh116/cuda_measurements/hull_support.py
src/dh116/cuda_measurements/pad_support.cu
src/dh116/cuda_measurements/pad_support.py
src/dh116/env/__init__.py
src/dh116/env/cameras.py
src/dh116/env/contracts.py
src/dh116/env/lifecycle.py
src/dh116/env/scene.py
src/dh116/hand/__init__.py
src/dh116/hand/articulation.py
src/dh116/hand/kinematics.py
src/dh116/hand/safety.py
src/dh116/physics/__init__.py
src/dh116/physics/diagnostics.py
src/dh116/physics/geometry.py
src/dh116/physics/runtime.py
src/dh116/physics/solver.py
src/dh116/physics/stereo_match.py
src/dh116/physics/vision.py
src/dh116/physics/wire.py
src/dh116/rl/__init__.py
src/dh116/rl/device.py
src/dh116/rl/ppo.py
src/dh116/rl/qualification.py
src/dh116/rl/reward.py
src/dh116/rl/student_data.py
src/dh116/rl/student_model.py
src/dh116/rl/teacher.py
src/dh116/rl/training.py
~~~

## 数据和发布

assets/、resources/、deploy/、vendor/保留。旧实验和恢复证据位于/mnt/mydisk/xk/archive/dh116-root-cleanup-20261124，项目内不保留历史runs/数据；runs/用于新的显式运行输出。

生成目录、缓存、原始数据、模型与日志不提交。Ubuntu提交后，由Windows导入同一commit并推送，核对GitHub SHA。PR #2保持Draft，不自动合并。
