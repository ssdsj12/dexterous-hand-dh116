# Stage 7：默认关闭的 mounted descriptor 与完整17D FK/J假设

本单元是只读测量接口与CPU运动学实现，**新增原生作业为0，原生FK/J校准尚未通过**。默认M0/Teacher、原四个控制器、3844D Actor、371D Critic、24D Tanh Gaussian、物理/材料/执行器及30mm/5s/0.5mm标准不变。PR继续Draft，不开放B3/PPO、Teacher替换或新专家池。

## 实现范围

- `--vision-mpc-model-descriptor-capture`默认关闭，必须同时启用既有model capture；消费同一次原refresh后的articulation sample，仅使用已存在的stage、左右hand view和`world.w.sim.physics_sim_view`。不创建World/view，不advance/refresh/set，不把特权信息加入Actor。
- 额外读取gravity、disable-gravity、link transform及Jacobian，保存当前composed joint graph、属性来源、完整17 DOF/23 links/22 joints布局、实际q/qd/惯性原始读回及每次getter前后时钟。未知量及失败partial保留，当前raw不能回退成上一帧。
- 左右手分别根据实际关节轴和各自root读回计算完整FK/3×17 point Jacobian；主动与被动SDK行保持独立，不应用仿射mimic代填，不假定左右完全镜像。根姿态xyzw、USD joint frame、坐标符号/零参考等都要求显式假设。
- 只提供当前姿态的数学假设，不包含RNEA、未来驱动/物体/接触、约束、求解器或控制接线。旧11D几何与控制路径保留。源码中的valid/structure及qualification标记不等于原生校准或成功。

## 实际CPU验证

独立source12基于只读source11复制，10个明确overlay文件；源文件清单SHA256为`9cc35b36af42f2adae8018bc51aaad05c464577f1aff8856b2c661f833c5f72a`。

实际执行入口为PRIVATE `prepare_stage7_model_descriptor_capsule12.py --remote-prepare --overlay-sha f39c4c07ea2bd1b17aa3a0916348e908cb99382b2ddb5ea7c3e0c509fcdd258f --manifest-sha 2834a2504749f672ccb83eef8d4af6c538cbe3937c634c08b4dc1b59ee6e5b4d`，测试在独立`/tmp/dh116_stage7_mpc_model_descriptor12_cpu`副本运行，CUDA关闭、无Isaac。实际测试命令由该入口保存，主体如下：

```text
bash scripts/python.sh -m pytest -q -p pytest_mock -p no:cacheprovider --junitxml=<independent output>/pytest.xml <27 explicit test files>
```

27文件覆盖state/measurement/capture、原Teacher采集、scene/contact/aux、articulation/model、objective及新descriptor/FK/integration/options。实跑**769 passed、1 CUDA-unavailable skip、238 subtests passed，54.21s，exit0**。源副本、父source11和原四个控制器/资产SHA前后一致；没有以修改运行中源码完成测试。

| 证据 | SHA256 |
| --- | --- |
| source12 CPU validation | `2415fc4a8b0c01aaf76f3c73157194122a9ba33ac76f05e429fffc91eb048dad` |
| pytest log | `1d5e53085e3500220de118837e78895f72d88c2455256b3f63ff05ea7ce56627` |
| descriptor source | `c98703fd3f47b2846f7601defca054fb1c39f5de85689078d42b6d6835417959` |
| full17 FK/J source | `fe979453af9cc4284064bc5faff01643b24441e629e78c34964d14896d62f60a` |

独立review另实际运行251项integration/相关测试、113 subtests（22.35s），以及19项FK测试（4.58s）和独立分叉树有限差分。修复了两个实际RED→GREEN边界：现有World/scene/hand别名变化必须在新getter前拒绝；坏dtype getter不得让partial复制异常覆盖原始来源异常。仅允许保存同次事务已取得的raw，不把它标成有效测量。

其后source13的独立CPU联合回归包含同字节source12及另外四个未在本提交发布的H8/contact文件：**887 passed、1 skipped、238 subtests，70.13s，exit0**。这不是本提交新增原生验证，也不把后续模块算成本单元实现。source13清单`cbbc1f8f3bcbb2e68783108b6ca5716630c331aba5a946a65739a064939ea1d9`，validation`679ae0f9e74b6a06a1acc8c40056c5a106174bac7e4760a2655231855c1820b8`；包含真实已安装SciPy下原恢复Runtime的15项兼容性测试。

## 原生状态及限制

本单元的有限descriptor1计划已经独立准备，但**尚未执行**，不会在另一Isaac仍运行时自动排队启动。原source11的180步接触数值采集继续独立运行；它不能替代本descriptor的原生读回。

已有source09左右17D读回及source11接触1/16步结果分别见此前报告，不作为本次FK/J的新结果。现有短测零力值不能证明无接触/承托。`carb.Float3`重力API没有tensor dtype合同，unknown来源标签不能被JSON数值类型冒充成backend float64。

下一步按单实例约束，在原作业自然退出和闭合核验后执行独立descriptor1；封存后先检查raw/DTO/graph/clock/SHA，再比较左右当前FK和两种预先声明的SDK Jacobian参考点假设。两者均保留，禁止按较小误差自动选择并宣称校准。若实际来源/API/图结构失败，保留失败，在新独立源码/目录修复，不覆盖source12或旧数据。

未来H8命令/响应、完整移动腕连续碰撞、力和载荷转移、共同暂停完整回合、每种抓法100个不同配对初态以及新Teacher/Student资格仍未完成。
