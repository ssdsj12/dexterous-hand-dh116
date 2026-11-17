# F5 WireSeg-36K 分块传输和首个完整分片

Ubuntu 的正式工作树新增标准库 CLI `scripts/relay_wireseg_blobs.py`。视觉预训练入口复用同一份固定 revision、148 项目录、完整 LFS 校验。传输检查 HTTPS、响应状态 206 和精确 Content-Range，记录每块字节数及 SHA256。重启后重新验证已存在的块和回执。超时只重试同一范围，达到预算即保存已完成块。完整合并件通过官方 LFS SHA256 和尺寸校验后才进入数据缓存。Windows 和 Ubuntu 分别运行校验，无需安装其他数据依赖。

## 首个完整 blob

固定 `DeformX/WireSeg-36K@1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b`。Windows 使用已安装的 uv Python 3.14 和系统受信 HTTPS。首次运行按预算收到两块，第二次复用它们并得到另外三块，最后一个范围发生读超时。独立重试只请求余下范围，Windows 完成组装，再经现有 SSH 配置把完整文件和六张回执传入 Ubuntu。Ubuntu 按六个连续 Content-Range 从完整文件独立重读每块，同时对完整文件做 LFS 校验。完整来源保存在数据盘，日志与审计保存在 Ubuntu 私有目录。

| 证据 | 实测 |
|---|---|
| 原始分片 | `depth/depth-00019-of-00100.tar.gz`，22,541,803 字节 |
| 完整 LFS SHA256 | `a3b12e0269c626e2c6983cc861f46eaa3c3ae53b543338ba4f8eba9d12c352c5` |
| Windows 分块／Ubuntu 独立块哈希 | 6／6 |
| gzip、tar 和 NPY | 360 个唯一成员，每个 1024×1024 float32 |
| 正且有限深度像素 | 5,726,719／377,487,360，源值 0.360272–1.827117 m |
| Ubuntu 整 blob 验证记录 SHA256 | `c70e66dca222471b9fd3b258c153a252333c414b73fae2d6e57bed11ad986064` |
| Ubuntu 逐块和深度数组审计 SHA256 | `e6272929d5b85e7d4085af1430c02107361feeaa97c08758accfbe06be3f3178` |
| 整目录状态 | 148 个请求，1 个完整，100 个深度分片中 1 个完整 |
| 关联原始 RGB、正式训练 | 当前此分片 0 个原始 Parquet RGB 配对，0 次训练更新 |

首批两块、随后的超时和最后仅补一块是实际续传。新增单范围超时自动重试已通过 CPU 回归。完整数据吞吐仍受链路限制，100 个深度分片与 148 个源 blob 仍按实际完整性单独登记。先前 Viewer 诊断结果保持独立。

## 复现

以独立 Windows 中继缓存为 `$relay_cache`，将固定 manifest 存在缓存根。两次调用使用相同分块尺寸，便于继续上次断点。发布仅包含代码、测试和报告。

```text
python scripts/relay_wireseg_blobs.py fetch --cache RELAY_CACHE --chunk-mib 4 --blob depth/depth-00019-of-00100.tar.gz --max-new-parts 2
python scripts/relay_wireseg_blobs.py fetch --cache RELAY_CACHE --chunk-mib 4 --blob depth/depth-00019-of-00100.tar.gz --max-new-parts 4 --timeout 75
python scripts/relay_wireseg_blobs.py verify --cache UBUNTU_CACHE --blob depth/depth-00019-of-00100.tar.gz
python scripts/relay_wireseg_blobs.py verify --cache UBUNTU_CACHE --all
```

Ubuntu CPU 回归 40 项通过，包含续传缓存完整性、Content-Range 拒绝、LFS 不匹配拒绝、同一范围超时重试和既有视觉训练入口。私有证据目录 `flexible-f5-relay-20261112-0x4SDr` 保留独立核验结果。
