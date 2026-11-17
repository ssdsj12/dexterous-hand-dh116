# WireSeg 配对深度持出验证

Ubuntu 入库新增官方 `depth/depth-00062-of-00100.tar.gz`，913337418 字节，固定 revision `1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b`，原始 LFS SHA256 `b730d6c70377585055522a52cb1c246ac88592a36d65ad470f4a365f8a54bccb`。Windows 四路续传耗时 118.4 秒，Ubuntu 独立整包 SHA 校验通过。当前缓存 5/148 个完整源分片，后续保持按固定清单续传。

独立评估命令为 `python -m scripts.pretrain_dh116_wireseg evaluate --mode seg-depth --extract-depth --validation-manifest <heldout> --checkpoint <seg-depth-16> --dataset-root <verified-cache> --output <independent-output>`。Checkpoint SHA `055ca13e11d3cc55a778d6fcde368685a3ee0c24aa80a4becf25a92c520832bd`，持出子集 SHA `5d5cc1331115737bfd039fa8223bd0f23cd67ea385595ffc0f03e71e6115f78f`。评估只用 128 张 wire_on_plane，训练只用 flying_wire。

| 指标 | 结果 |
|---|---:|
| 精确配对持出图像 | 128 |
| 裁剪区有效参考深度 | 4718592 像素（100%） |
| 可见线缆 IoU | 0.118849 |
| 深度 MAE | 1.003469 m |
| 深度 RMSE | 1.008793 m |
| 深度 P95 | 1.213317 m |

深度指标覆盖全部有效中央裁剪像素。16 步诊断模型的误差仍大。原生双目光轴深度、线缆三维精度和 Student 资格继续独立验证。结果 SHA256 `80b06c08119e4d0bb358c3ce14d925e0bd479f77f9735b13f314f0c99e6cc9a0`。CPU 回归 18 项通过。
