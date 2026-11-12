# PhysX GPU 接触报告读回与左右拇指标定

在 Isaac Sim 4.5.0、Isaac Lab 0.54.4 和 GPU 3 上，沿用隔离的实际 DH116 finger14 网格、固定运动学安装座与自由下落球体。原任务的物理配置、控制器和成功标准未变。两个标定进程顺序执行，每轮 240 个 480 Hz 物理步，每轮自然退出码 0。左侧 04:52:44–04:52:59 UTC，右侧 04:53:48–04:54:03 UTC。

`--enable-contact-report-readback` 仅用于独立标定。既有 `--enable-full-report-processing` 仍把 `/physics/disableContactProcessing` 设为 false。新增选项在建场后、reset 前单独将 `/physics/suppressReadback` 从 true 设为 false，随后核对 reset 后及每条记录的值。安装版本 `physics_context.py` 通过 GPU pipeline 选择该读回设置。默认标定调用保持原设置。

| 指标 | 左侧 | 右侧 |
|---|---:|---:|
| 完整物理记录与 SDK 步号 | 240，3–242 | 240，3–242 |
| 已解析 object/partner actor 与 collider 的 poll headers | 37 | 43 |
| 非零 Tensor / poll 物理步 | 30 / 30 | 32 / 32 |
| ordinary callback / full callback receipts | 37 / 37 | 43 / 43 |
| full callback contact entries / friction anchors | 36 / 38 | 42 / 43 |
| 以 actor0 为受力侧，contact 加 friction 的最大动量残差（N·s） | 2.1061e-10 | 1.0322e-9 |
| 同侧只计 contact 的最大动量残差（N·s） | 5.2587e-4 | 6.2458e-4 |
| 反号 contact 加 friction 的最大动量残差（N·s） | 7.7377e-3 | 8.1770e-3 |

两侧 header 的 object 都在 actor0，partner 都解析到对应已知安装座。ordinary API 不提供 friction 数组。保留 full、ordinary 与 poll 的三套原始记录；动量对照没有重复加计 callback。两轮在原生循环完成后 fsync，并保留了关闭返回、回调释放和 context 清理记录。与 v3 左侧比较，240 步 SDK 时间、物体位姿及每步前后速度逐值相同；法向 Tensor 力的最大差异为 1.49e-8 N。v3 关闭读回时 0 个可识别 headers。

结论限于两个隔离刚体对：开启 GPU→CPU 接触报告读回后，原生 pair 身份和两类冲量通道可观测。两轮的 actor 顺序相同，未实测反序；任务中的逐指映射、完整外部接触、附着及施力证据仍需在现有 `solid_cylinder` 场景逐步核验。抓取资格、R1、R2 与 Force Proxy 保持关闭。

| 证据 | SHA256 |
|---|---|
| 冻结源码变更 | `a7d1ff1b0d5b262d49b425152eb4dc39009524dba56c1b5b5ff02d9f7f0a2770` |
| 左 raw JSONL | `9ad911494f4d84296cb743b979d4294b1600491cafaf405daeaf2e55e10f2f85` |
| 右 raw JSONL | `57fb5018546a15c1bc312663b73534fb8e192a47c5e5955910da174f00586b30` |
| 左只读复核 | `d7306b360553b410c6e7d162a54ba0af465de11289986f1d229cb2cbd34a` |
| 右只读复核 | `9f4f02c71721ba3cf1b97f8d7f4ebd4c7798b07a813ee374ae4168aa66a9be2b` |

复现使用服务器现有解释器、CUDA_VISIBLE_DEVICES=3、冻结快照中的 `bash scripts/python.sh scripts/calibrate_dh116_stereo_contacts.py --execute-native --side left --steps 240 --enable-full-report-processing --enable-contact-report-readback --full-contact-callback --ordinary-contact-callback --monitor-sigterm --output "$new_output"`。右侧参数为 `--side right` 且使用新的独立输出目录。只读复核使用 `scripts/audit_dh116_physx_contact_sources.py`，同时校验原始 JSONL、inventory 和 USD SHA256。

Ubuntu CPU 测试使用 `python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_stereo_shutdown_validation.py tests/test_dh116_physx_polled_contact_audit.py`。新增读回接受与拒绝回归先观察 2 项失败，修复后 67 项通过，0 失败。原始日志、冻结源码和物理证据仅保存在 Ubuntu 私有证据目录。
