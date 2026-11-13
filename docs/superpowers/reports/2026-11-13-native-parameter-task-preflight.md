# 独立原生参数后端：双手真实任务读回准备（2026-11-13）

基线 `95ecd3008ab4863a4e41879df6c3b4de69ee54ad`。该包仅准备新的有界实验，不把 CPU fixture 当成原生结果。

## 实际修改

- `prepare_dh116_native_parameter_task.py`：复用现有 guarded closure/director 的真实 owner、Reset、物理循环和释放顺序，显式选择独立 getter 后端；仅在该入口临时绑定，异常后恢复旧默认。原动作不变：2 初始化 + 8 零动作 + 8 原手指动作，无腕部抬升/旋转。
- `validate_dh116_native_parameter_task.py`：独立重放原始参数，核验时钟、源二进制、构造 rest、EA/EI、质量、重力、摩擦、半径、自由节点和 director；复用原来的关节、碰撞、账本、Tensor View、GDB 和自然退出审计，不放松保护。参数读取失败保留 terminal 物理记录，Reset 失败则不能发动作。
- 独立 verify/worker/JSON 配方：真实父回执及全部依赖输入绑定，新读回对象仅持有 weakref，无隐藏 native view；要求17包（Reset +16物理边界）且读取不推进时钟。
- 两个 CPU 测试文件：逐包篡改拒绝、owner完整/失败/Reset路径、作用域恢复、实际 worker 子进程重入、外层审计、预算及真正调度器配方加载。

旧二进制 SHA `92a50ba3747192b72fd607741ff1465c17874f7c8f8fb0cbed82d99d5985d8c0` 不变。新二进制 SHA `2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec`；原72个源码文件锁定，除只读 bindings getter 外源码字节相同。新 getter 的0步独立GPU探针已通过，这是历史前置，不能替代本次真实任务验证。

## CPU 证据

命令：
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_native_parameter_task.py tests/test_dh116_native_parameter_task_worker.py tests/test_dh116_native_parameter_endpoints.py tests/test_dh116_deformx_parameter_getters.py tests/test_dh116_deformx_parameter_probe.py tests/test_flexible_parameter_readback.py tests/test_dh116_parameter_packet_validation.py
```

合并回归146 passed /0 failed /191.00s，日志：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-parameter-task-regression-hrj7MD/pytest.log`。
另补2项实际调度器配方测试，最终worker回归19 passed /0 failed /19.79s，日志 `native-parameter-task-final-worker-wTsnYO/pytest.log`；与前146项有17项重叠，共覆盖148个独立测试，不能相加为165。

初始失败测试保留于 `native-parameter-task-red-aUpGla`。冻结前发现 protected USD 被误加入 publishable source_files，真实 scheduler 拒绝；现改为继承已审查源码清单，资产仍按原始SHA只读核验，新增拒绝不合法资产发布路径测试。没有修改 USD。

## 新配方（未执行）

GPU3、单实例、1环境、0 PPO、18物理步、180秒上限、最多一次；既往配方预算不复用。
- 配方 SHA：`81accf84f883f51c4a0425c5e94adf41ba34034faeb511d96e739b9948171346`
- scheduler代码配置 SHA：`bc73ea4aba22065b14830640743d21f1133eee28bedd0821fa191320f3a8ceb3`
- worker代码配置 SHA：`1e91af45356148106e49643c1194a2ab98bc62333bb426c10fb3ecbfcb9bfe12`

执行前在发布HEAD记录用户“批准，后续自动批准”对该确切配方的范围绑定，检查单实例锁和资源。新后端只有 getter 先前验证通过；真实任务接入尚待本包后继原生执行与独立磁盘审计。失败不重跑，保留账本/栈。

## 未解决与晋级限制

Teacher仍0/3，PPO更新0；没有形成新的抓取成功、双侧接触或材料标定证据。端点 director 不能替代 solver 子步接触材料点历史；滑移、R0全部来源和全371D资格仍需后续工作。不能将原生半径读回等同实测材料或将18步诊断等同完整episode。Pinch接触几何、Tripod/Power各自可达性与相机/WireSeg任务保持独立；原30mm/5s/0.5mm标准不变。

发布按 Ubuntu commit → Windows 导入同commit push；PR #2 Draft。保护清单和两份历史Checkpoint在每次集成/发布核验。

## 集成发现与修复

首次集成35 passed /1 failed，日志 `flex.f7-native-parameter-task-cpu-gza3f5t9/pytest.log`。失败为同字节配置的工作目录绝对路径进入指纹，导致工作树与集成树SHA不同；没有启动原生、没有消耗原生预算。先补跨真实工作树失败回归，再将该配置归入相对local_sources并保留其内容SHA核对。外部二进制、SDK和原始证据仍保持绝对路径与SHA。额外位置回归使累计独立测试149项；补充最终CPU结果见Registry。未执行的旧配方指纹作废。
