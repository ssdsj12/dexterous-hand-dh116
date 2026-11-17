# F5 WireSeg-36K 数据接入和单目预训练诊断

## 数据身份和空间预算

固定 `DeformX/WireSeg-36K@1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b`。Hugging Face 同版本 blob 目录提供 45 个合成 Parquet（34,736,075,305 字节）、100 个深度 gzip shard（61,321,341,597 字节）、3 个 real Parquet（1,263,480,631 字节）。用于脚本验收的 148 项 blob 三元组（相对路径、字节数、LFS SHA256）总量为 97,320,897,533 字节，规范化 SHA256 为 `569022f7f5c1985cba7eedb6cd563619a55417001e60f81fabff1cdb5b68f7ac`。独立 COCO annotations 约 222 MB，Parquet 已提供同用途的逐行标签。完整 36,000 个深度矩阵按 float32 1024² 计算，未压缩约 151 GB，加上源数据约 248 GB。Ubuntu 数据盘检查为 3.5 TiB 可用，Windows D 盘检查为 817 GB 可用。仓库只保留小型代码、测试和报告。

原始资料按 CC-BY-4.0 提供。real 300 张为官方外部图像测试。当前采用整族隔离，`flying_wire` 为 train，`wire_on_plane` 为 validation，`data_center` 为 test，`real_easy` 和 `real_hard` 为 real_test。数据源未提供已核验的轨迹和渲染组身份，整族隔离避免按相邻帧切分。该划分只用一个合成族训练，跨族评估需要保留其分布差异。后续只有在获得真实渲染组身份后才采用场景族和线缆数分层。

## 实现与使用

`tabletop_wireseg_pretraining.py` 保留每实例 COCO 压缩 RLE 和原像素位置，按文件相对路径连接 float32 NPY 深度。正且有限值才进入监督。缺失深度没有替代值，配对摘要包含文件身份、RGB 原字节 SHA 和 NPY 原字节 SHA。流式抽取拒绝链接、非规范路径、同一 shard 重复标识和冲突内容。命令行将使用的已抽取路径限定在当前经 LFS 校验的 shard 清单。

单目模块直接复用既有 StereoEncoder 的 GN-ResNet18 和高层投影结构。附加 stride-8 浅层特征与实例、可见区域、边界、metric depth 头，训练损失独立记录。实例匹配保持每图实例身份，深度只参与有效像素损失。显式迁移 API 只加载共享 backbone 与 projection。左右同步、跨注意力和双目对应需要各自的数据，当前诊断未验证。这个工件无 371D、24D、机器人动作、左右相机对或力学标签。

在 Ubuntu 独立数据工具目录准备 cp310 `pyarrow==19.0.1`。原 DH116 Python 及其依赖未替换。将锁定修订的 API blob 目录放入 `hf_revision_manifest.json`，完整下载件保留源相对路径。

```bash
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python
root=/mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86
export PYTHONPATH="$root/tools:$PWD/src:$PWD"
"$DH116_PYTHON" scripts/pretrain_dh116_wireseg.py audit --dataset-root "$root" \
  --extract-depth --output "$root/audit-pinned-run"
"$DH116_PYTHON" scripts/pretrain_dh116_wireseg.py pretrain --dataset-root "$root" \
  --extract-depth --steps 1 --device cpu --output "$root/pretrain-smoke"
```

不加 `--extract-depth` 时只允许分割监督。所有运行使用独立输出目录。CLI 当前训练入口为固定步数和单图批次的诊断；正式训练仍需完整源文件、持出集评估、细线和实例 AP 指标、批量训练及训练预算。

## 实测和失败保留

Ubuntu Hugging Face 直连证书校验失败。Windows 原生 curl 超时；经系统受信连接的 PowerShell 请求可读取修订、blob 目录和 206 Range。首个 depth gzip 前缀的 Content-Range 为 `bytes 0-16777215/1230080936`，两端前缀 SHA256 一致，`947e8664a0baa5deee587901a6e7796ec28bfcc7a5fb4d45137d3b04e6156692`。全文件 GET 尝试未完成，完整源文件须经清单中的 LFS SHA256 校验才进入 CLI。大文件传输仍受当前网络路径吞吐限制。

Ubuntu 私有目录 `/mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86/viewer-audit-aNUOzq` 保留真实输入诊断。Dataset Viewer 派生 JPEG 保持原像素尺寸，其字节身份未与原 Parquet 图像字段建立等同性。4 行真实数据含 3 张合成图像和 1 张官方 real 图像，实例数 3、3、2、2。两张 `data_center/4_cable` 图与同名深度精确配对。两张深度分别有 1,048,576 个正且有限的 float32 像素，原值范围为 28.818–152.261 和 40.511–176.738。原值未裁剪或转作 DH116 场景尺度。官方 real 图无深度。该运行只在 `flying_wire` 图上执行 1 次 CPU 分割优化，instance 2.0421、visible 1.8204、boundary 2.1968、total 4.4117，metric depth 实际训练步数 0。保留的 result.json SHA256 为 `62caa30ef0ff4b1d63e4d6808afff410f1939ab00f4b76fc56ca4bf7d51a2fb8`。脚本 SHA256 为 `840d01ab0113cd73991ddf12dc19b78f6446fb214d01e8926a5fc22941054f0a`。

Ubuntu CPU 回归 33 项通过。测试覆盖压缩 RLE 差分、精确文件名、深度缺失、归档防逃逸、锁定目录摘要、真实和合成族隔离、四个受监督头梯度、显式 backbone 迁移。完整数据下载、正式预训练和真实泛化待核验。此次无 Isaac、原生抓取或视觉部署资格。
