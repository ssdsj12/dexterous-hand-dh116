# F6 官方 DeformX-Assets 的 drop/hang 资产审计

Ubuntu 独立缓存固定为 Hugging Face `DeformX/DeformX-Assets@9ef440fa263dcd9b39689c97860f1d94d599bc5a`。已下载方案中四个指定背景的 **109 个文件、176,253,735 字节**子集，包含全部线缆 USD、drop/hang 场景及轨迹、18 张地面纹理、4 个背景、脚本和仓库元数据。完整约 2.18 GB 的数据中心与背景集合不在本次子集。来源为[官方固定版本](https://huggingface.co/datasets/DeformX/DeformX-Assets/tree/9ef440fa263dcd9b39689c97860f1d94d599bc5a)。

Ubuntu 直连源端返回过期 TLS 证书，下载使用验证 TLS 的 Windows 中继，实际读取与审计在 Ubuntu 完成。首次批量下载发生传输中断，按固定路径重试与续传。108 文件首批中继核验 SHA 后解包，最后一个背景独立验证并传输。对最终 109 文件缓存从头执行官方 SHA256 和字节数检查。README 当前内容与官方内部旧清单不一致，单独锁定固定版本的实际字节，差异保留在输出中。其他下载内容未放宽摘要要求。

| 检查 | Ubuntu 实测 |
|---|---|
| 文件完整性 | 109/109；最后补齐的 `empty_workshop_4k.hdr` SHA256 `2cb701b87912452d8de97a261edec8c6b76e05f3dfe65366d119bcebd1b765e5` |
| 线缆 USD | 72/72 可解析，Z-up、1 m/单位；每个真实网格 2,652 个有限顶点，20 个骨架关节。均验证网格索引和依赖 |
| 模板 USD | 3 个 drop 与 1 个 hang，均可用 OpenUSD 22.11 读取。每个模板有一个未解析的原作者 DomeLight 默认纹理，其余未发现未解析或根外依赖 |
| Drop NPZ | 2、4、8 杆各 100 条几何记录、100 个 seed、31 个时间向量样本（约 0–1.5 s） |
| Hang NPZ | 2、4、8 杆各 200 条几何记录、100 个 seed、201 个时间向量样本（约 0–2 s）；逐记录 hanging/sweep 根数之和等于对应杆数 |
| 几何维度 | 6 份 NPZ 的每杆节点为 21、director 为 20，radius 数值为 0.001–0.005；按原字段保留时间向量、记录和 seed 身份，未将时间向量用作逐行时间戳 |
| 原生 Isaac / CUDA / 双手交互 | 均未执行 |

三个 drop 模板引用 `usd/textures/color_0C0C0C.exr`，hang 引用 `usd/textures/industrial_pipe_and_valve_01_4k.hdr`。审计保留这四条原始未解析边，`authored_usd_dependencies_closed=false`。允许名单仅匹配指定模板里可核验的 `DomeLight` 纹理。其他未解析和根外依赖使 CLI 返回非零。上游 DeformX 固定提交 `8f32ad730c3d321e03397810c1eabd972f38cd5b` 的生成器先引用原线缆 USD，再给 `/World/DomeLight` 选择背景并禁用原 DomeLight。生成器行为来自固定源码读取，渲染效果待原生检查。[生成器源码](https://github.com/DeformX/DeformX/blob/8f32ad730c3d321e03397810c1eabd972f38cd5b/dataset_generator/drop_hang/generator.py)

官方来源说明将 DeformX 原创 USD、轨迹标为 MIT，Poly Haven HDRI 为 CC0，地面 Pexels 图像适用 Pexels License。NVIDIA Datacenter_NVD 和 Texturelabs 原始贴图因外部来源未随发布包提供，当前也未下载。[资产来源与许可](https://huggingface.co/datasets/DeformX/DeformX-Assets/blob/9ef440fa263dcd9b39689c97860f1d94d599bc5a/README.md#provenance-and-licensing)

## 复现

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
  PYTHONDONTWRITEBYTECODE=1 DH116_NUM_THREADS=2 \
  bash scripts/python.sh scripts/audit_dh116_deformx_assets.py \
  --asset-root /mnt/mydisk/xk/integrations/dh116-flex-f6-9ef440fa/snapshot \
  --lock docs/contracts/dh116_deformx_assets_lock.json \
  --output /tmp/dh116-flex-f6-audit.json
```

验收使用现有环境的 `scripts/python.sh` 仅导入 OpenUSD。F6 新测试、CUDA source 和资产隔离合计 14 通过。连同旧蒙皮测试重跑合计 29 通过、0 失败。旧测试原始路径指向服务器缺少的外部夹具，初始基线有 14 个 setup 错误。复跑仅通过私有 pytest 夹具将其指向此次字节等价的原版橙色 USD（SHA256 `990c18a99064c700fb4a742070aefff6036a8efd5bbb7c6bf8e56f873398d5a9`）、同源纹理和项目已保留的许可证。旧源码与测试未编辑。裸解释器测试配方不加载 `pxr`，F6 + CUDA source 11 通过、1 跳过。

| 原始证据 | SHA256 |
|---|---|
| 完整 109 文件审计 JSON，外层退出 0 | `7d4c69c8bffccffe938c19130e513d2488a61c8fe254f84694a4ab2a9013837a` |
| OpenUSD 诊断日志 | `f4cf88c08af56c48ab9ce3d5214c06857db5568050fe345641b2ee611ef70543` |
| 29 项 CPU 回归日志 | `dd59f644e2523a6af3ab27f572f0ea01e2e34cb5050c489bd79ee7c1ab7af840` |
| 官方 manifest | `509ed09a36393b40ff6b0775789a9aa08505176a7f3dc2e9014cf6ec16e55836` |

私有证据目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-f6-assets-audit-WUUmwa`。源文件、HDRI 和审计日志保存在 Ubuntu 缓存或私有证据目录。528 个保护文件及 2 个历史 Checkpoint 通过核验。

后续渲染冒烟使用固定 DeformX 的 `dataset_generator/drop_hang/cli.py`，以 `DEFORMX_ASSET_ROOT` 指向核验缓存，分别检查 drop variant 0 和 hang variant 3 各一帧，原生运行应单 Isaac 实例且独立输出。现有 USD 骨架和轨迹用于渲染资产及初态来源。Cosserat 材料节点、时间同步、接触力、DH116 观测、实际 24D 动作和训练标签需要后续运行证据。
