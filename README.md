# DH116

唯一开发、构建和运行目录：/mnt/mydisk/xk/dh116。开发分支：11-17。

当前包为 src/dh116/，按 env、physics、rl、hand 分层。底层物理、接触、控制、几何守卫与生命周期由Teacher和Student共用。模型使用eager执行，关闭torch.compile。

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
| scripts/train_teacher.py | 无相机 Teacher PPO、控制窗验证、冻结 Actor 的原生双目回放 |
| scripts/train_student.py | 封存数据的状态监督/BC，保留独立初态持出与准入 |
| scripts/run_student.py | Actor-only双RGB部署推理 |

构建使用vendor/ropesim和当前源码；build/scene_bundle保存真实SDF、完整凸包与独立的计划初态/实测Reset，不上传原始数据。新克隆需配置经过SHA核验的本地输入，不会自动伪造几何或监督。

Pinch、Tripod 和 Power 各自绑定原生指垫、桌面和双端支架。严格几何审计继续提供 FP64 凸包和 SDF 对照。完整抓取训练的接触与终止规则见下节。

## 单模板建场（SCENE-INIT-SPEEDUP R1）

共享运行时默认使用 Cloner 的 USD 继承克隆：env0 完成原资产、mount/mimic、材质与34碰撞体校验，再复制其余环境；原配置的支架存在时一并复制。精确复用 SDK 原作者的坐标量化顺序，保留 float64 origins；碰撞隔离使用 inverted groups，并保留全局物体碰撞。

128路建场准备实测由205.716s降至6.658s（30.9倍），不含诊断属性导出；包含SDK启动、全量对照写盘和关闭的整进程为40.976s。2/128路固定同组24D动作的两窗原生对照自然exit0：USD属性/世界变换、Reset、关节/线缆/奖励/守卫精确一致；接触行按真实身份排序后相同，float64力规约差异最大9.99e-16（原路径重复也存在求和顺序差异），未放宽任何门槛。

可显式选择 --scene-build-mode reference 进行旧路径对照。replicated 仅保留诊断入口：本次2路测试在app.close返回后以-6退出（GIL finalization），禁止进入PPO训练，默认不使用。完整CPU/CUDA回归182项通过（169 CPU、13 CUDA），证据及SHA见configs/runtime/readiness.json；本轮仅本地commit，不push。

## 三抓法 Teacher PPO

同一个训练入口使用 `--full-grasp` 启用双腕线缆局部参考、一次性参考姿态到位、实测双侧对向力、完全离架高度和连续保持。Pinch 与 Tripod 的非参与指保持原目标。左手在 −Y、右手在 +Y，两侧拇指相向。三个任务均从同钟 DeformX 接触对读取桌面和两个端支架载荷。恢复旧模型前应核对配方和来源 SHA。

完整抓握奖励使用同钟 DeformX 接触对生成有界的逐指轻载进度。双手已对向承载时，由手对线缆的向上受力、桌面与两端支架的竖向卸载以及线缆中位点和低点共同引导预抬升。离架前后使用同一势函数。成功须满足实测对向承载、完全离架、净抬升 30 mm、连续保持 5 s 与滑移不超过 0.5 mm。非目标载荷超过 0.05 N 或原始手部反力达到 40 N 时重新累计 Hold，探索阶段仍按连续成本计分。

完整抓握训练在每局内只对首次改善的接近、手势、左右轻载、预抬升、离架高度和保持时间追加有界奖励。新一局用实测 Reset 值初始化峰值。离架奖励要求无外部支撑和真实双侧承载。手腕目标按实际局部驱动行程核对，修正后重新读回到同一世界坐标系，保留 2.1 mm 桌面投影。

当前三抓法各支持 2048 环境。生产运行使用各自同源、同规模且自然退出的两窗收据，64,000 控制窗分成 2,000 次 PPO 更新。沿用无相机采样，恢复点和原生账本只写本地。

完整抓取 PPO 对每个候选更新计算实际 rollout 的步后精确 KL，超限时一并回滚模型、Adam 和 RNG 再退避。`latest.pt` 保存已完成更新的恢复点，`best.pt` 保存产生评分轨迹的策略。双目回放从经过来源核验的训练检查点提取 Actor，在两路环境执行。

腕部指令限速为平移 0.05 m/s、旋转 1.0 rad/s，主动指关节为 1.2 rad/s，被动关节限速为 3.0 rad/s。动作仍为 24D，教师与学生部署按同一动作 SHA 校验。

完整抓取模式使用 FP32 指垫 SDF 和原生接触对。指垫顺应超过 0.8 mm、非目标手部网格载荷超过 0.05 N、跟踪滞后超过 10 mm、mimic 误差超过 0.08 rad、接近关节限位及原生反力超过 15 N 时产生有界连续扣分。提交给 PhysX 的每网格反力限幅为 40 N，奖励使用原始测量。线缆掉出桌面、关节超过硬限位 0.25 rad、指垫间隙低于 -3 mm、达标后复触支撑及成功分别统计。活跃环境的数值异常，以及共享求解器或来源异常，拒收本窗。已终止环境保留最后一次有限的测量，随后在窗末局部 Reset。

物理 480 Hz，控制 60 Hz。每窗 8 步，PPO 每 32 窗更新，完整回合默认 600 窗（10 秒）。硬终止在窗末只重置受影响的环境。超时截断保留价值自举。默认使用 `official_cache` 单次读回与 minibatch KL 早停。每 25 轮、最佳更新及最后一轮原子保存 checkpoint。实例按物理 GPU UUID 加锁。

### Native stereo replay

用当前配方的已核验 2 或 512 环境 checkpoint 在两路环境中回放冻结 Actor。命令中的路径和 SHA 取自同一次经过验证的训练。

~~~bash
bash scripts/python.sh -B scripts/train_teacher.py \
  --preview-checkpoint /path/to/latest.pt --preview-checkpoint-sha256 SHA256 \
  --grasp power --gpu 3 --output runs/stereo-preview \
  --preview-windows 90 --preview-every 15
~~~

三视图采用同次原生渲染。`configs/cameras/student_stereo.json` 固定 80 mm 双目基线和 86° 水平视场，两台相机从操作者 +X 侧上方朝操作区俯视。原始 RGB 时钟与测得的 SDK Reset 后时钟逐帧校验，容差为 1 µs。橙色管状显示跟随实测线缆节点。左手位于线缆 −Y 侧，右手位于 +Y 侧，双手拇指朝内。每个采样窗口保存第三人称和左右 RGB、帧时钟及 `student_rgb.pt`。原始 RGB 包供 `run_student.py` 使用，`train_student.py` 读取通过资格校验并封存的数据集。

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
