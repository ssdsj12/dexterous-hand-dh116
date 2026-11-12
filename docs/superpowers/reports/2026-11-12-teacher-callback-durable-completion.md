# Teacher 回调结束与外层退出证据

先前任务级原生接触探针有实际退出 0、场景 summary 和回调内 `teacher_evaluation.json`，回调外的 `completion.json` 缺失。原 Isaac launcher 的 App close 可以结束解释器，回调外代码不能作为完整运行证明。

Teacher 训练、候选采集、有界评估及独立冷重置 Teacher 评估现在在原生回调中 fsync 已关闭或已 flush 的状态、动作、训练记录、模型或评估账本，并写入和 fsync `completion.json` 与包含新目录项的目录。回执使用 `completion_scope=native_callback_finished_before_app_close`，独立外层退出仍由进程观察器记录。保留原子任务、引擎和关机路径。冻结评估仍需原始 480 Hz 和原生成功资格。

Ubuntu 回归 82 通过，0 失败。新测试使用明确 CPU fixture 模拟 launcher 在回调后立即退出，直接验证回执已可读并限制其声明范围。项目挂载点的文件与目录 fsync 另行验证通过。日志 SHA256 `b89630e11382056dd0859e741161a257053d1b89d2578a6ac4eddc17d647be9b`。本包未执行 Isaac，真实关闭路径待独立原生验证。
