# Snapshot02: native bootstrap normal-exit return

本次只修改新模块`tabletop_stereo_native.py`中的`launch_native`，新增纯CPU回归`tests/test_dh116_stereo_native_bootstrap.py`。原launcher、原任务/控制/物理源码以及单实例锁行为保持不变。本次native执行数为0，未训练、未启动Isaac，未授予任何模型或传感器原生资格。

## 问题与修复

原launcher在清理和退出状态发布之后执行`raise SystemExit(code)`。新bootstrap原先未捕获成功退出，导致即使callback已经生成结果，调用者的`completion.json`写入和冷试验CLI输出仍不可达。

新bootstrap仅接受`SystemExit(None)`或整数`SystemExit(0)`。非零、字符串和其他非整数退出payload原样继续抛出；普通异常、`KeyboardInterrupt`和bridge清理失败同样继续抛出。成功返回还要求callback执行、结果收集和bridge关闭均已完成；使用独立完成标记，允许合法的空字典结果，同时拒绝未执行callback的正常退出。原有`finally`继续恢复被隔离的training callback、`sys.argv`和工作目录，单实例锁仍覆盖整个launcher生命周期。

## TDD与CPU验证

测试使用真实`runpy.run_path`执行临时CPU launcher脚本，替换的仅为NativeBridge与Linux实例锁依赖，未把runpy或SystemExit替换成mock。脚本会真实改变工作目录和argv，再正常返回、抛`SystemExit`或异常，以验证恢复路径。

先RED：18项中7失败、11通过，失败准确复现成功`SystemExit(None/0)`不能返回、空结果误判和无callback的成功退出未被校验。最小修复后18项全部通过。覆盖成功结果/空结果、`None`/整数0/普通返回、非零及非整数退出、无callback、异常、中断、callback失败和bridge清理失败；失败路径不产生调用者completion文件。

最终联合命令（PowerShell，repository cwd）：

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_native_bootstrap.py tests/test_dh116_privileged_cli.py tests/test_dh116_stereo_camera_cli.py tests/test_dh116_stereo_runtime.py tests/test_dh116_stereo_resolution_benchmark.py tests/test_dh116_stereo_trials.py tests/test_dh116_stereo_resolution.py tests/test_dh116_stereo_geometry.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_dagger.py
```

结果：`164 passed in 16.23s`，exit0，无warnings。该结果只证明CPU软件控制流与既有回归；实际Linux/Isaac shutdown兼容性仍需后续独立原生流程验证。

## 冻结指纹

|文件/范围|SHA256|
|---|---|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_native.py`|`238c0e23a47c9518ea51aa383fab5a6a2b6b2a82a3cebc5d5e16304df163728c`|
|`tests/test_dh116_stereo_native_bootstrap.py`|`3e7b77b8b0cb37505d46be1f2b50ec31bb3f325fdd29762d9902eb8db0f9258b`|
|未改原`train_dh116_tabletop_gpu_ppo.py`|`4353281ea7ebdb076af472a4e57832789809128fedf245d27c3e99d4d488d644`|
|未改`native_instance_lock` AST，`ast.dump(include_attributes=False)`|`b1109436d8395d186ea0ed2ad7d49b8c090c26749fda684268a19ae993b256b4`|

Phase0 action manifest绑定的原30个source SHA仍一致；旧224配置SHA仍为`f5c53f5c9d3042ee0ce97453a3420dda3ecb7417469d7362be20604b0171d494`。本报告完成后本agent再次冻结源码，后续采集须绑定snapshot02实际文件SHA。
