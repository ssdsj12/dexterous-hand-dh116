# DH116 独立工作目录

## 最近上传：2026-10-18

当前阶段：独立工程隔离与离线验证；实际双手接近试验进行中，尚未确认闭指、持续夹持和抬升成功。

本次改进：独立整理 DH116 控制、USD 依赖、接触配置和 DeformX 后端；增加上传排除规则及持续进度记录。本次本地回归 **559 passed in 51.52s**，运行配置预览通过，未启动第二个 Isaac。详细变更见 [CHANGELOG.md](CHANGELOG.md)。

GitHub 版本不包含约 4.7 GB 的 `archive/` 历史证据、运行日志及缓存；原件保留本地。部分测试和配置重建工具依赖该归档，**仅克隆仓库不能复现全部本地测试**。当前运行配置仍含此机器的绝对路径与指纹，异机需重新生成配置，不可直接修改校验值冒充兼容。厂家资料与第三方资源保留原有权利归属，本仓库不授予额外再分发许可。

当前为固定双 Panda + 普通 DH116 V2.0 + 桌面 DeformX 柔性线缆项目。不是 DH116S，不包含机器狗任务或 O6 资产。`fixed_dual_panda_o6` 命名空间内仅保留 DH116 使用的接触桥和线缆可视化两个共享模块。

## 当前迁移状态

- 1380 个静态来源文件已复制并 SHA 校验，原目录未删除；详见 `migration_manifest.json`。
- 主模型的 5 个 USD 层已隔离到 `assets/`，依赖相对引用；`OmniPBR.mdl` 由 Isaac 软件环境提供。
- 新运行配置已重新校验指腹表面坐标、接触几何、源路径 94 个区间和 loader 指纹。
- 主运行入口读取本目录的 DeformX；Python 启动脚本重置 PYTHONPATH，仅使用本项目及安装的软件库。
- 主运行、空载验证、资产生成及测试资源路径已适配，独立目录全量 **559 passed in 50.95s**，见 `regression.log`。共享测试中仅排除了不适用的旧 O6 资产专项测试，DH116 接触身份测试保留。
- **实际运行迁移尚未完全验收**：新目录实际 Isaac 验证尚未执行。原目录当前试验结束前不启动第二个实例。
- 原活动试验 `table_margin_raise1mm_trial1` 未纳入静态归档，结束后才能同步最终证据。

## 内容

`src/` 控制与共享依赖；`scripts/` 运行/校验；`tests/` 回归；`assets/` 新运行 USD；`configs/` 新配置；`vendor/DeformX/` 实际后端；`resources/manufacturer/` 厂家资料；`archive/` 历史原件；`docs/source/` 设计与记录；`tools/` 可审计迁移工具。

归档中的旧路径和 SHA 保持不变。它们不是新目录可直接恢复的状态；禁止修改旧 stage pin 冒充兼容。新配置的模拟参数不代表实机额定参数。

## 环境和命令

使用已有 `/home/xk/miniconda3/envs/go2/bin/python`、Isaac Sim/Isaac Lab、GPU0。可用 `DH116_PYTHON` 指定兼容 Python 环境；不复制整个 conda 环境。所有入口通过 `scripts/python.sh` 限定 core0，数值库单线程。

```bash
cd '/home/xk/coding/dexterous hand'
bash scripts/python.sh -m pytest tests/test_migration_copy.py tests/test_asset_isolation.py tests/test_standalone_layout.py -q
# 完整回归（包含真实 DeformX 后端测试，不启动 Isaac）
bash scripts/python.sh -m pytest tests -q
# 只打印命令并验证配置，不启动仿真
bash scripts/python.sh scripts/run_current.py
# 原实例结束后再运行；存在其他 DH116 verifier 时会拒绝启动
bash scripts/python.sh scripts/run_current.py --run
# GUI 版本需指定独立输出目录
bash scripts/python.sh scripts/run_current.py --run --gui --output runs/gui_trial1
```

这些入口尚待新目录实际物理验收，不能将命令可打印当作抓取成功。

## 抓取进度

旧恢复试验成功加载状态，但接近末段桌面余量不足触发保护，未闭指/夹持/抬升。当前调整为左手路径上移1mm，名义桌面余量约2.20mm；实际试验仍在原项目运行。未使用绑定伪造抓取，1mm安全阈值不变。
