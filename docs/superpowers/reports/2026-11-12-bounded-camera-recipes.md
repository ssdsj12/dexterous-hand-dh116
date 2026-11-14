# 相机生命周期 V0–V3 有界执行配方

四格固定 Isaac Lab headless rendering Kit 和已安装 Replicator 源文件 SHA。每格单进程、GPU3、一次尝试。V0 空启动，V1 单相机 product，V2 RGB，V3 一次渲染读 RGB、径向距离、Optical-Z 和 Native ID。每格只依赖上一格的独立外层退出和范围匹配的原生资格。V3 只验证最小单相机路径。当前柔性任务双目仍需要原生同步与自然重启的独立资格。

配方审批文件与原始产物存放在私有证据目录。退出码 139 时保留产物和日志并停止依赖链。采用原 `native_instance_lock`，每个进程自然退出。
