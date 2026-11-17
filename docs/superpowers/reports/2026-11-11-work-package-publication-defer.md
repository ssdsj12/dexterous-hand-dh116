# Ubuntu 调度器离线提交与发布恢复

集成基线 `d4da7c102fd96062b81a62b978bbf0911d7d6bba`。Ubuntu 到 `api.github.com` 的 TLS 校验报 hostname mismatch，旧版默认模式会在 CPU 验证前退出。现在发生连接、TLS 或超时时，调度器在本地分支和提交链通过核对后继续验证并提交工作包。队列收据记录 `remote_verified: false`。远端返回不合格的 PR、HTTP 错误或已知分支分歧时保持阻断。中途推送失败仍使用事务日志恢复。证书校验保持启用。

## CPU 验证

`python -m pytest -q -p no:cacheprovider tests/test_dh116_work_packages.py`

- 修复前新增离线回退及本地提交身份测试 2 项失败，PR 身份阻断测试通过。
- 修复后 Ubuntu 17 项通过、0 失败，退出码 0。日志 SHA256 `3898d4ef8d298b435da22a0370ae383b83c26fe401e6ed12b6d41f59c7ed4084`。
- 使用临时裸 Git 仓库检查队列、恢复、提交范围及发布清单。该测试不授予任何原生资格。

GitHub 连接恢复后从 Ubuntu 执行 `--resume`，按提交顺序读回并写入发布清单。
