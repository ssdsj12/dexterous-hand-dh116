# F5 第二个官方深度分片与传输速率

在已有 Windows HTTPS 续传入口上，用现有 Windows 本地代理发起一次独立的三块预算诊断。保留正常 TLS 证书校验。下载同一固定 revision 的 `depth/depth-00024-of-00100.tar.gz`，分块尺寸 8 MiB，三个 Content-Range 成功并完成官方 LFS 校验。Windows Python 3.14 和 Ubuntu cp310 均使用相同入口。

| 检查 | 实测 |
|---|---|
| 传输大小与 Windows 墙钟 | 22,972,279 字节／224.78 秒，约 0.102 MB/s |
| LFS SHA256 | `9e7c7280f6aa0e75b2f0799a6d3e950b08217b40aa5724958882bc93b2478eba` |
| Ubuntu 独立校验 | 完整大小和 SHA256 一致，gzip 可解，tar 列出 360 项 |
| Ubuntu 官方缓存全表 | 148 个 blob 中 2 个完整，均为深度分片，完整深度 2/100 |
| 独立 Ubuntu 核验 SHA256 | `3ac01a75d5c3dc2beb221e4cba0dfba1d57290f3f1faeaeeac709ea04dbf03b2` |
| tar 成员索引 SHA256 | `e4a2e8c85d218608c121eb60db9d508ace250c87071b1470c641417da607bc4a` |

单次 4 MiB Range 探测一度在 3.53 秒返回，完整小分片的实际墙钟明显更长。当前有可恢复的完整性流程，速率依然是完成全量数据的阻塞。此诊断没有原始 Parquet RGB，因此没有新增正式 RGB–深度配对或训练更新。原始文件和日志仅保留在数据缓存及 Ubuntu 私有证据目录。
