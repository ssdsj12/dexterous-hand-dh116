# 验证三种 Actor 的独立双目部署

基线 `3bc2014f165592263b6887ea098ff7ccc17cb0b7`。CPU 状态 `passed`，通过 47 项，失败 0 项，跳过 0 项，退出码 0。此项不授予原生或训练资格。

命令 `python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_inference.py tests/test_dh116_stereo_runtime_actor_modes.py tests/test_dh116_stereo_student_ppo.py`。私有日志 SHA256 `dbe521125ef8bb9ee2fad31cf76e4a423e101a66dd0c43ac717a8e65cd0c85f5`。后继工作包 wp6.bc-dagger-ppo。
