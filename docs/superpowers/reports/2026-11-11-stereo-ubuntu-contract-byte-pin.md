# 371D 状态合同的 Ubuntu 字节校验

基线提交 3c80166da75fb5496701c9f3ff673296c5db5615。Ubuntu 检出的状态合同使用 LF，原 v1 配置保存的是同一文件的 CRLF SHA。将 Ubuntu 合同的每个 LF 换成 CRLF 可准确复现原 SHA。旧测试直接比较两个原始字节 SHA，因此在 Ubuntu 失败。

v1 配置校验明确接受旧 CRLF pin 及它对应的两个经审计字节序列。CRLF SHA 为 `a7af9f3f214a646f2f95dce45aa4744f378488de34237431c914bd4137b2ee86`，LF SHA 为 `9f145cf8e68b1dfe8e2c7464ec5fc0cb00e81a2707115c2fe148ab49d77bbb62`。校验返回实际文件 SHA。混合换行、额外空白、字段修改和其他配置 pin 均拒绝。语义 SHA 校验仍独立运行。原合同、配置及 Checkpoint 字节身份未修改，已有适配器仍计算实际原始文件 SHA。

Ubuntu Python 3.10.12 上原始失败已复现。专项 CPU 回归 78 项通过，用时 16.31 秒。528 个保护文件和两份历史 Teacher Checkpoint SHA 均核对一致。未启动 Isaac、采集或训练。

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_state_multicohort.py tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_cold_candidate_admission.py
```

日志位于独立服务器目录 `/tmp/dh116-autowp3-sha-tWeCXU`，green.log SHA256 为 `2ecede97b9f6db6cf65faa7cdbace451b03b8c973ab94a9fcf6355a8fdfec3c6`。下一步是在集成 Autowp2 后运行相同 CPU 回归。原生训练仍须各自满足资格及批准门槛。

集成 Autowp2 后在 Ubuntu 重跑以上五个文件，82 项通过，0 失败，19.47 秒。之前失败的固定 SHA 用例已通过。
