# R06/R07/R08 Execution Chain Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 修复生命周期故障收尾、非中心相机投影和独立保留证据封存；不改写旧原生结果。

**Architecture:** R06新增opt-in lifecycle模块，读取调用时的旧模块绑定；旧消费源码和recipe保持原样。R07提取USD aperture数学，由实际author/readback使用同一组正确公式并用Gf真实投影检查。R08完成现存独立重审scaffold，直接依赖scheduler.file_hash，校验outer/工件/数值Reset并冻结重审源码身份。

**Tech Stack:** Python3.10、pytest、torch、numpy、CPU pxr.Gf、既有DH116源读取器。

---

## 实施位置和共同命令

本文代码/命令只在实施阶段执行。准备worktree时使用using-git-worktrees；分支前缀codex/。三个独立工作树：

- R06：/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r06-lifecycle-20261115
- R07：/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r07-camera-20261115
- R08：/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-postexit-replay-20261114。先确认HEAD=5eac3ac40ffaaae69a4458b7838d4dca9e806f04，保留两个现有untracked文件，禁止覆盖。R08先在原producer版本完成，避免将当前新源码fingerprint误套历史recipe。

测试统一使用以下环境；不能直接PYTHONPATH=src:.调用PY，否则缺pxr不是源失败：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_dh116_pinch_v2.py --tb=short
~~~

每包先核对HEAD/status和528保护文件/两个历史Checkpoint，之后再次核对。旧receipt、旧config、已消费wrapper、模型和账本不改。source/config是否变动与readiness更新由root精确审查；子包仅stage下文明确文件，不广泛git add。发布由Windows处理。本文无native启动；CPU通过不授Teacher/Student/native资格。

## Task 1 / R06：新版本故障安全生命周期

**Files:**

- Create: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r06-lifecycle-20261115/scripts/dh116_fault_safe_lifecycle_v3.py
- Create: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r06-lifecycle-20261115/tests/test_fault_safe_lifecycle_v3.py
- Read only: scripts/dh116_pinch_acquisition_v2.py及所有旧native recipe。

### 1.1 写失败测试

- [ ] 新增以下完整测试。run_owned_call使用现有真实实现；FakeTrace只代替CPU所有权观察和checkpoint文件格式，不代替待测run_lifecycle。

~~~python
import importlib
import importlib.util
import json
import sys
import weakref
from types import SimpleNamespace
import pytest
from scripts import dh116_pinch_acquisition_v2 as old

def api():
    name = "scripts.dh116_fault_safe_lifecycle_v3"
    assert importlib.util.find_spec(name) is not None, "fault-safe lifecycle missing"
    return importlib.import_module(name)

@pytest.mark.parametrize("damage", ["none", "acquire", "clear", "stop", "close", "manager", "freeze"])
def test_real_lifecycle_failure_order_and_alias_release(tmp_path, monkeypatch, damage):
    module=api();events=[];refs={};frames=[];freeze_count=[0]
    class Sim:
        current_time_step_index=2
        current_time=2/480
    class Owner: pass
    class Trace:
        def __init__(self,path):self.references={}
        def watch(self,name,value,**kwargs):
            self.references[name]=refs[name]=weakref.ref(value)
        def checkpoint(self,phase):pass
        def event(self,phase,**kwargs):pass
    def acquire(sim,owners,trace,cleanup):
        value=None
        try:
            value=owners.keep("runtime",Owner());trace.watch("runtime",value)
            if damage=="acquire":raise RuntimeError("injected acquire")
            return dict(physics_rows=0,code_config_sha256="a"*64)
        finally:value=sim=None
    def clear(sim):
        try:
            events.append("clear");frames.append(sys._getframe())
            if damage=="clear":raise RuntimeError("injected clear")
        finally:sim=None
    def stop():
        events.append("stop")
        assert refs["simulation"]() is None
        if damage=="stop":raise RuntimeError("injected stop")
    def freeze():
        freeze_count[0]+=1
        return damage=="freeze" and freeze_count[0]==1
    def close():
        events.append("normal_close" if (tmp_path/"before_app_close.json").exists() else "fault_close")
        assert refs["simulation"]() is refs["runtime"]() is None
        assert frames[0].f_locals.get("sim") is None
        if damage=="close":raise RuntimeError("injected close")
    monkeypatch.setattr(old,"LifetimeTrace",Trace)
    monkeypatch.setattr(old,"OWNED_NAMES",("runtime","simulation"))
    monkeypatch.setattr(old,"snapshot_weakrefs",lambda rows:dict(resources={
        k:dict(alive=v() is not None) for k,v in rows.items()}))
    kwargs=dict(make_sim=Sim,acquire=acquire,synchronize=lambda:None,
        clear_sim=clear,stop=stop,manager_is_none=lambda:damage!="manager",
        freeze=freeze,output=tmp_path,cleanup="explicit")
    app=SimpleNamespace(update=lambda:events.append("update"),close=close)
    if damage=="none":
        result=module.run_lifecycle(app,**kwargs);assert result["failure"] is None
    else:
        with pytest.raises(RuntimeError):module.run_lifecycle(app,**kwargs)
        result=json.loads((tmp_path/"result.json").read_text());assert result["failure"]
    assert events.index("clear")<events.index("stop")
    assert sum(x in ("normal_close","fault_close") for x in events)==1
    assert result["native_qualification"] is result["teacher_admission"] is False
    stages=json.loads((tmp_path/"cleanup_stages.json").read_text())
    assert stages["version"]==module.VERSION and stages["qualification_granted"] is False
    if damage in ("clear","stop","manager","freeze"):
        assert stages["close_mode"]=="fault" and "fault_close" in events
        assert not (tmp_path/"before_app_close.json").exists()
        assert not (tmp_path/"app_close_returned.json").exists()
        assert (tmp_path/"fault_app_close_returned.json").exists()
    else:assert stages["close_mode"]=="normal"
    if damage=="clear":
        assert stages["stages"]["clear"]["error"]
        assert stages["stages"]["stop"]["returned"] is True
    if damage=="stop":assert "update" not in events
    if damage=="close":
        assert not (tmp_path/"app_close_returned.json").exists()
        assert stages["stages"]["close"]["error"]

@pytest.mark.parametrize("raise_inside",[False,True])
def test_opt_in_scope_restores_legacy_binding(raise_inside):
    module=api();before=old.run_lifecycle
    try:
        with module.repaired_lifecycle_scope():
            assert old.run_lifecycle is module.run_lifecycle
            if raise_inside:raise RuntimeError("fixture")
    except RuntimeError:assert raise_inside
    assert old.run_lifecycle is before
~~~

- [ ] RED：在R06工作树执行 `DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= bash scripts/python.sh -B -m pytest -q tests/test_fault_safe_lifecycle_v3.py --tb=short`。预期9 failed，原因fault-safe lifecycle missing。不得以pxr缺失作为该RED。

### 1.2 实现新模块

- [ ] 写入以下完整代码；不改旧模块。新caller必须显式使用repaired_lifecycle_scope；当前历史入口继续保留原身份。未来新native recipe采用及新自然退出验证属于D阶段。

~~~python
"""Explicit opt-in cleanup repair; consumed recipes remain unchanged."""
from contextlib import contextmanager
import gc
from pathlib import Path
import traceback
import weakref
from scripts import dh116_pinch_acquisition_v2 as previous
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

VERSION="dh116-fault-safe-lifecycle-v3"

def run_lifecycle(app,*,make_sim,acquire,synchronize,clear_sim,stop,
                  manager_is_none,freeze,output,cleanup):
    if cleanup!="explicit":raise ValueError("unsupported cleanup case")
    output=Path(output);trace=previous.LifetimeTrace(output/"ownership.jsonl")
    sim=None;sim_ref=None;sim_created=False;failures=[];clocks=[];acquired={};stages={}
    def attempt(name,callback):
        record=dict(attempted=True,returned=False,error=None);stages[name]=record
        try:
            value=callback();record["returned"]=True;return value
        except BaseException as error:
            text=traceback.format_exc();record["error"]=text;failures.append(text)
            traceback.clear_frames(error.__traceback__);error.__traceback__=None
            return None
        finally:callback=None
    def frozen():
        if freeze() is not False:raise RuntimeError("Kit physics freeze did not read back false")
        return True
    def observe(phase):
        target=sim_ref() if sim_ref is not None else None
        try:clocks.append(dict(phase=phase,clock_available=target is not None,
                              **(previous.clock(target) if target is not None else {})))
        finally:target=None
        trace.checkpoint(phase)
    def acquire_owned():
        nonlocal sim,sim_ref,sim_created,acquired
        sim=make_sim();sim_created=True;sim_ref=weakref.ref(sim)
        trace.watch("simulation",sim,owner=VERSION,release="clear local alias then STOP",callbacks=True)
        received=previous.run_owned_call(lambda owners:acquire(sim,owners,trace,cleanup),
            synchronize=synchronize,ownership=trace)
        if not isinstance(received,dict):raise ValueError("owned acquisition dictionary required")
        acquired=received
        if acquired.get("failure"):raise RuntimeError(str(acquired["failure"]))
    attempt("acquire",acquire_owned)
    initial_frozen=attempt("freeze_before_stop",frozen) is True
    clear_returned=stop_returned=False
    if sim_created:
        attempt("observe_before_stop",lambda:observe("before_stop"))
        try:
            attempt("clear",lambda:clear_sim(sim))
            clear_returned=stages["clear"]["returned"]
        finally:sim=None;gc.collect()
        attempt("stop",stop);stop_returned=stages["stop"]["returned"]
    after_stop_frozen=attempt("freeze_after_stop",frozen) is True
    if stop_returned and after_stop_frozen:attempt("post_stop_update",app.update)
    attempt("observe_after_stop",lambda:observe("after_stop"))
    manager_released=attempt("manager_readback",manager_is_none) is True
    if not manager_released:
        def reject_manager():raise RuntimeError("manager view release false or unknown")
        attempt("manager_guard",reject_manager)
    attempt("observe_before_close",lambda:observe("before_app_close"))
    def ownership_readback():
        resources=previous.snapshot_weakrefs(trace.references)["resources"]
        names=tuple(dict.fromkeys((*previous.OWNED_NAMES,"simulation")))
        return bool(names) and all(isinstance(resources.get(n),dict) and
            resources[n].get("alive") is False for n in names)
    released=attempt("ownership_readback",ownership_readback) is True
    if not released:
        def reject_owners():raise RuntimeError("owned resource release false or unknown")
        attempt("ownership_guard",reject_owners)
    close_frozen=attempt("freeze_before_close",frozen) is True
    cleanup_failed=any(v["error"] for k,v in stages.items() if k!="acquire")
    normal=bool(initial_frozen and clear_returned and stop_returned and after_stop_frozen
                and manager_released and released and close_frozen and not cleanup_failed)
    mode="normal" if normal else "fault"
    before_name="before_app_close.json" if normal else "fault_before_app_close.json"
    returned_name="app_close_returned.json" if normal else "fault_app_close_returned.json"
    def close_app():
        before=dict(case=cleanup,clocks=clocks,lifetime_released=released,
            play_simulations=False if close_frozen else None,process_exit_verified=False)
        if not normal:before.update(close_mode="fault",qualification_granted=False)
        previous.write_json(output/before_name,before);app.close()
        returned=dict(case=cleanup,app_close_returned=True,process_exit_verified=False)
        if not normal:returned.update(close_mode="fault",qualification_granted=False)
        previous.write_json(output/returned_name,returned)
    attempt("close",close_app)
    attempt("observe_after_close",lambda:observe("after_app_close"))
    if not normal and not failures:failures.append("normal close admission failed")
    previous.write_json(output/"cleanup_stages.json",dict(
        version=VERSION,close_mode=mode,stages=stages,qualification_granted=False))
    steps=[row["sdk_step"] for row in clocks if row.get("clock_available")]
    result=dict(version=previous.VERSION,cleanup_version=VERSION,case=cleanup,
        acquisition=acquired,clocks=clocks,close_mode=mode,
        physics_steps=max(steps) if steps else None,measurement_steps=acquired.get("physics_rows"),
        rod_steps=acquired.get("physics_rows"),code_config_sha256=acquired.get("code_config_sha256"),
        physics_clock=acquired.get("physics_clock"),physics_budget=acquired.get("physics_budget"),
        ppo_updates=0,play_simulations=False if close_frozen else None,
        lifetime_released_before_close=released,failure="\n".join(failures) or None,
        process_exit_verified=False,native_qualification=False,grasp_qualified=False,teacher_admission=False)
    previous.write_json(output/"result.json",result)
    if failures:raise RuntimeError("\n".join(failures)) from None
    return result

@contextmanager
def repaired_lifecycle_scope():
    with bindings((previous,"run_lifecycle",run_lifecycle)):yield
~~~

VERSION/OWNED_NAMES/LifetimeTrace在调用时从previous读取，避免from-import冻结别名破坏Tripod既有bindings。故障close只作单独有记录的收尾尝试；异常不被写成正常释放。STOP失败不执行app.update。clear异常后强引用在finally释放，原异常与各收尾异常分别保存。

### 1.3 GREEN、旧路径回归、精确提交

- [ ] 重跑RED命令：预期9 passed。
- [ ] 运行以下完整回归，预期exit0，实际计数写入实施证据：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_dh116_pinch_v2.py tests/test_dh116_pinch_v3.py tests/test_dh116_dual_pinch_entry.py tests/test_dh116_bilateral_owned_owner.py tests/test_tripod_native_scope.py --tb=short
~~~

- [ ] diff检查、保护哈希确认后，仅stage两个新文件：

~~~bash
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r06-lifecycle-20261115 diff --check
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r06-lifecycle-20261115 add -- scripts/dh116_fault_safe_lifecycle_v3.py tests/test_fault_safe_lifecycle_v3.py
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r06-lifecycle-20261115 diff --cached --stat
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r06-lifecycle-20261115 commit -m "Add opt-in fault-safe lifecycle cleanup"
~~~



## Task 2 / R07：非中心主点的实际USD投影

**Files:**

- Modify: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r07-camera-20261115/src/go2_pvcnn/control/dh116/tabletop_stereo_calibration.py
- Modify: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r07-camera-20261115/src/go2_pvcnn/control/dh116/tabletop_stereo_camera.py
- Create: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r07-camera-20261115/tests/test_stereo_usd_principal_point.py

### 2.1 写真实Gf投影回归

- [ ] 新增以下完整测试。左右slot各自变换world点，覆盖3种分辨率、中心/非中心主点。不能只让author/readback两个错误公式互相抵消。

~~~python
import importlib
import numpy as np
import pytest
import torch
from pxr import Gf

@pytest.mark.parametrize("width,height",[(64,32),(224,96),(640,480)])
@pytest.mark.parametrize("off_centre",[False,True])
@pytest.mark.parametrize("side",["left","right"])
def test_actual_usd_projection_matches_requested_K(width,height,off_centre,side):
    m=importlib.import_module("go2_pvcnn.control.dh116.tabletop_stereo_calibration")
    assert hasattr(m,"usd_aperture_offsets"),"USD projection helpers missing"
    fx,fy=.75*width,.9*width
    cx=width/2+(.11*width if off_centre else 0)
    cy=height/2-(.07*height if off_centre else 0)
    k=torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=torch.float64)
    focal=24.
    horizontal,vertical=m.usd_aperture_offsets(k,width,height,focal)
    camera=Gf.Camera()
    camera.focalLength=focal
    camera.horizontalAperture=width*focal/fx
    camera.verticalAperture=height*focal/fy
    camera.horizontalApertureOffset=horizontal
    camera.verticalApertureOffset=vertical
    optical=np.eye(4)
    optical[:3,3]=[-.04 if side=="left" else .04,.7,1.2]
    usd=optical@np.diag([1.,-1.,-1.,1.])
    camera.transform=Gf.Matrix4d(*usd.T.reshape(-1).tolist())
    view=camera.frustum.ComputeViewMatrix()
    projection=camera.frustum.ComputeProjectionMatrix()
    for xyz in ((0.,0.,1.),(.03,-.02,.7)):
        world=optical@np.array([*xyz,1.])
        ndc=projection.Transform(view.Transform(Gf.Vec3d(*world[:3].tolist())))
        actual=np.array([(ndc[0]+1)*width/2,(1-ndc[1])*height/2])
        expected=np.array([fx*xyz[0]/xyz[2]+cx,fy*xyz[1]/xyz[2]+cy])
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=0)
    readback=m.usd_aperture_intrinsics(width,height,camera.focalLength,
        camera.horizontalAperture,camera.verticalAperture,
        camera.horizontalApertureOffset,camera.verticalApertureOffset)
    torch.testing.assert_close(readback,k,atol=2e-4,rtol=0)
    if not off_centre:assert horizontal==vertical==0.
~~~

- [ ] 在R07工作树用共同命令运行tests/test_stereo_usd_principal_point.py。预期12 failed，原因helpers missing。已确认的旧原式Gf复现为640x480/fx400/cx350/cy260光轴约290/220，不是native渲染结果。

### 2.2 实际数学和调用点补丁

- [ ] calibration.py追加以下完整两个函数。原dataclass字段、to_dict和digest不变：

~~~python
def usd_aperture_offsets(k,width,height,focal_length):
    k=_matrix(k,(3,3),"USD lens K");focal=float(focal_length)
    if (type(width) is not int or type(height) is not int or min(width,height)<1
        or not math.isfinite(focal) or focal<=0 or k[0,0]<=0 or k[1,1]<=0):
        raise ValueError("positive USD lens dimensions and scales required")
    return ((width/2-float(k[0,2]))*focal/float(k[0,0]),
            (float(k[1,2])-height/2)*focal/float(k[1,1]))

def usd_aperture_intrinsics(width,height,focal_length,horizontal_aperture,
                            vertical_aperture,horizontal_offset,vertical_offset):
    values=tuple(float(x) for x in (focal_length,horizontal_aperture,
        vertical_aperture,horizontal_offset,vertical_offset))
    if (type(width) is not int or type(height) is not int or min(width,height)<1
        or not all(math.isfinite(x) for x in values) or min(values[:3])<=0):
        raise ValueError("finite USD lens data and positive dimensions required")
    focal,horizontal,vertical,ox,oy=values
    fx,fy=width*focal/horizontal,height*focal/vertical
    return torch.tensor([[fx,0.,width/2-ox*fx/focal],
                         [0.,fy,height/2+oy*fy/focal],
                         [0.,0.,1.]],dtype=torch.float64)
~~~

- [ ] camera.py原calibration import替换为：

~~~python
from .tabletop_stereo_calibration import (
    StereoCalibration, usd_aperture_offsets, usd_aperture_intrinsics)
~~~

- [ ] _NativeRGBPairReader.__init__中原CreateHorizontalApertureOffsetAttr和CreateVerticalApertureOffsetAttr两行替换为：

~~~python
horizontal_offset,vertical_offset=usd_aperture_offsets(
    k,calibration.width,calibration.height,focal)
camera.CreateHorizontalApertureOffsetAttr(horizontal_offset)
camera.CreateVerticalApertureOffsetAttr(vertical_offset)
~~~

- [ ] _NativeRGBPairReader._readback整体替换为以下真实方法。SDK focal getter读取focalLength/10，因此保留乘10；不改变相机轴/槽顺序：

~~~python
def _readback(self):
    from .tabletop_depth_camera_probe import usd_pose_to_optical
    positions,quats=self.view.get_world_poses(camera_axes="usd",usd=True)
    transforms=np.stack([usd_pose_to_optical(p,q)
        for p,q in zip(_array(positions),_array(quats))])
    focal=np.asarray(self.view.get_focal_lengths())*10.
    horizontal=np.asarray(self.view.get_horizontal_apertures())
    vertical=np.asarray(self.view.get_vertical_apertures())
    c=self.calibration;K=np.tile(np.eye(3),(self.count,1,1))
    cache=self.UsdGeom.XformCache(self.Usd.TimeCode.Default())
    for i,path in enumerate(self.paths):
        camera=self.UsdGeom.Camera(self.stage.GetPrimAtPath(path))
        K[i]=usd_aperture_intrinsics(c.width,c.height,focal[i],horizontal[i],vertical[i],
            camera.GetHorizontalApertureOffsetAttr().Get(),
            camera.GetVerticalApertureOffsetAttr().Get()).numpy()
        authored=np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T@np.diag([1.,-1.,-1.,1.])
        if not np.allclose(authored,transforms[i],atol=1e-5,rtol=0.):
            raise ValueError("USD/runtime stereo calibration mismatch")
    index=self.order.cpu().numpy()
    K=K[index].reshape(self.n,2,3,3)
    transforms=transforms[index].reshape(self.n,2,4,4)
    self.calibration.validate_readback(
        K[:,0],K[:,1],transforms[:,0],transforms[:,1],origins=self.origins)
    return K,transforms
~~~

### 2.3 GREEN、回归、提交

- [ ] 重跑12项：预期12 passed。中心默认offset为0，原中心config序列化/digest保持。
- [ ] 完整回归：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_dh116_stereo_geometry.py tests/test_dh116_stereo_resolution_benchmark.py tests/test_dh116_stereo_rig_experiments.py tests/test_dh116_stereo_runtime_contract.py --tb=short
~~~

预期exit0，实际计数留实施后记录。只证明Gf数学，不授native accuracy资格。

- [ ] 保护核对及diff审查后精确提交：

~~~bash
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r07-camera-20261115 diff --check
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r07-camera-20261115 add -- src/go2_pvcnn/control/dh116/tabletop_stereo_calibration.py src/go2_pvcnn/control/dh116/tabletop_stereo_camera.py tests/test_stereo_usd_principal_point.py
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r07-camera-20261115 diff --cached --stat
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r07-camera-20261115 commit -m "Correct USD principal point offsets and readback"
~~~



## Task 3 / R08：独立保留证据重审封存

**Files:**

- Finish existing untracked: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-postexit-replay-20261114/scripts/verify_retained_dh116_pinch_v3.py
- Extend existing untracked: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-postexit-replay-20261114/tests/test_retained_pinch_v3.py
- Read only: scripts/run_dh116_mesh_reset_numeric_pinch_work_packages.py、scripts/run_dh116_pinch_fast_evidence_work_packages.py、原config、receipt、intent和所有原工件。

现存scaffold已经定义verify_original_attempt、verify_numerical_reset、producer_profile、verify_saved_run、assemble_verified_sources与CLI。它锁定：

- 原receipt：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-pinch-fast-evidence-native/f0b5bc8cd80210db27052f4f96f435c7ce48a7ff23d53fd533a53fd3ed63502f/attempt-2ynpy5si/receipt.json
- 原receipt SHA：6bc124165c6cd907e84ac013b07c9f9df45b2ee1cd4f9499253b42d924adc4ef
- 原失败原因：independent lifetime audit rejected evidence: AttributeError: module 'scripts.verify_dh116_mesh_reset_numeric_pinch' has no attribute 'sha'

不在旧wrapper加sha alias；这会改已消费source身份。新重审直接调用更底层mesh/health审查及numeric验证，以scheduler.file_hash封存。这是历史物理源重新验证，不是把原scheduler失败改成成功。

### 3.1 写outer和真实工件负向测试

- [ ] 在现有test_retained_pinch_v3.py末尾追加完整以下代码。原wrapper AttributeError复现测试保留；tamper只在内存改变读取摘要，不写原工件：

~~~python
ROOT=Path(__file__).resolve().parents[1]
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f8-pinch-fast-evidence-native/"
    "f0b5bc8cd80210db27052f4f96f435c7ce48a7ff23d53fd533a53fd3ed63502f/"
    "attempt-2ynpy5si/receipt.json")

def outer_fixture():
    debug=dict(debugger_status_verified=True,debugger_backtrace_captured=False,
        debugger_inferior_exit_code=0,debugger_inferior_signal=None)
    outer=dict(duration_seconds=161.766509436,outer_exit_code=0,
        native_lock_observed=True,log_sha256="d"*64,**debug)
    return dict(outer,exit_code=0),outer,dict(budget=dict(timeout_seconds=180)),debug,"d"*64

@pytest.mark.parametrize("damage",["outer_exit","receipt_exit","signal","inferior_exit",
    "unverified","duration","nan_duration","lock","log","debug_mismatch"])
def test_outer_failure_not_waived(damage):
    assert hasattr(replay,"verify_outer_evidence"),"explicit outer verifier missing"
    receipt,outer,spec,debug,actual=outer_fixture()
    if damage=="outer_exit":outer["outer_exit_code"]=1
    elif damage=="receipt_exit":receipt["exit_code"]=1
    elif damage=="signal":debug["debugger_inferior_signal"]=11
    elif damage=="inferior_exit":debug["debugger_inferior_exit_code"]=1
    elif damage=="unverified":debug["debugger_status_verified"]=False
    elif damage=="duration":outer["duration_seconds"]=181
    elif damage=="nan_duration":outer["duration_seconds"]=float("nan")
    elif damage=="lock":outer["native_lock_observed"]=False
    elif damage=="log":actual="f"*64
    else:receipt["debugger_inferior_exit_code"]=2
    with pytest.raises(ValueError,match="natural exit"):
        replay.verify_outer_evidence(receipt,outer,spec,debug,actual)

def test_outer_complete_fields_pass():
    assert hasattr(replay,"verify_outer_evidence"),"explicit outer verifier missing"
    assert replay.verify_outer_evidence(*outer_fixture()) is True

@pytest.mark.parametrize("artifact",["physical.jsonl","sdk_clock.jsonl",
    "mesh_reset_state.json","mesh_reset_numeric_state.json"])
def test_real_artifact_mismatch_rejected_in_memory(monkeypatch,artifact):
    from scripts import run_dh116_work_packages as scheduler
    before=RECEIPT.read_bytes();receipt=json.loads(before)
    damaged=Path(receipt["output"])/artifact;original=scheduler.file_hash
    monkeypatch.setattr(scheduler,"file_hash",
        lambda path:"f"*64 if Path(path)==damaged else original(path))
    with pytest.raises(ValueError):replay.verify_original_attempt(ROOT,RECEIPT)
    assert RECEIPT.read_bytes()==before

def test_real_debugger_signal_rejected(monkeypatch):
    from scripts import run_dh116_work_packages as scheduler
    original=scheduler.debugger_status
    def status(path):
        return dict(original(path),debugger_inferior_exit_code=None,debugger_inferior_signal=11)
    monkeypatch.setattr(scheduler,"debugger_status",status)
    with pytest.raises(ValueError,match="natural exit"):
        replay.verify_original_attempt(ROOT,RECEIPT)

def test_complete_saved_sources_preserve_all_original_artifacts():
    from scripts import run_dh116_work_packages as scheduler
    before=RECEIPT.read_bytes();receipt=json.loads(before);output=Path(receipt["output"])
    watched={RECEIPT:scheduler.file_hash(RECEIPT)}
    watched.update({p:scheduler.file_hash(p) for p in output.iterdir() if p.is_file()})
    result=replay.verify_saved_run(ROOT,RECEIPT)
    assert result["verified_saved_physical_sources"] is True
    assert result["original_scheduler_status"]=="failed"
    assert result["native_qualification"] is result["teacher_admission"] is False
    assert result["new_physics_steps"]==result["ppo_updates"]==0
    assert result["replay_source_sha256"],"new replay source identity missing"
    assert result["original_receipt_sha256"]==replay.RECEIPT_SHA
    assert RECEIPT.read_bytes()==before
    assert all(scheduler.file_hash(p)==sha for p,sha in watched.items())
~~~

- [ ] RED，仅outer新测试：

~~~bash
cd /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-postexit-replay-20261114
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_retained_pinch_v3.py -k 'outer_failure or outer_complete' --tb=short
~~~

预期11 failed，原因explicit outer verifier missing。现有已经通过的格式检查不需要人为制造失败。

### 3.2 完成独立验证器

- [ ] 在verify_retained_dh116_pinch_v3.py增加import math与以下完整函数：

~~~python
def verify_outer_evidence(receipt,outer,spec,debug,actual_log_sha256):
    duration=outer.get("duration_seconds")
    keys=("duration_seconds","outer_exit_code","native_lock_observed","log_sha256")
    if (type(duration) not in (int,float) or not math.isfinite(duration)
        or not 0<duration<=spec["budget"]["timeout_seconds"]
        or any(outer.get(k)!=receipt.get(k) for k in keys)
        or any(type(row.get("outer_exit_code")) is not int or row["outer_exit_code"]!=0
            or row.get("native_lock_observed") is not True for row in (outer,receipt))
        or type(receipt.get("exit_code")) is not int or receipt["exit_code"]!=0
        or not isinstance(actual_log_sha256,str) or not scheduler.SHA256.fullmatch(actual_log_sha256)
        or actual_log_sha256!=outer.get("log_sha256")
        or debug.get("debugger_status_verified") is not True
        or type(debug.get("debugger_inferior_exit_code")) is not int
        or debug["debugger_inferior_exit_code"]!=0
        or debug.get("debugger_inferior_signal") is not None
        or any(row.get(k)!=v for row in (outer,receipt) for k,v in debug.items())):
        raise ValueError("actual wall time, native lock or natural exit differs")
    return True
~~~

- [ ] verify_original_attempt保留原receipt SHA、intent/approval/recipe/source和native_artifact检查；将现有以 “if (not 0 < outer” 开始、对应raise结束的outer条件段完整替换为：

~~~python
verify_outer_evidence(
    receipt,outer,recipe["spec"],debug,scheduler.file_hash(native_log))
~~~

- [ ] verify_saved_run整体替换为以下真实方法：

~~~python
def verify_saved_run(project,receipt_path):
    project=Path(project)
    receipt,output,outer,generic=verify_original_attempt(project,receipt_path)
    with producer_profile(project,output):
        with entry.identity_scope(),bindings(
            (entry.operation,"VERSION",entry.VERSION),
            (entry.operation,"SCOPE",entry.SCOPE),(parent,"entry",entry)):
            audit=parent.audit_outcome(output,outer,"explicit")
        numeric=verify_numerical_reset(output)
    result=assemble_verified_sources(receipt,generic,audit,numeric)
    names=tuple(dict.fromkeys(("scripts/verify_retained_dh116_pinch_v3.py",
        *parent.AUDIT_SOURCES,*entry.FILES)))
    sources={}
    for name in names:
        path=scheduler.native_artifact(project,name)
        sources[name]=scheduler.file_hash(path)
    unchanged=scheduler.file_hash(Path(receipt_path))==RECEIPT_SHA
    if not unchanged:raise ValueError("original failed receipt changed during replay")
    result.update(replay_source_sha256=sources,
        qualification_scope="retained_pinch_v3_saved_physical_source_reaudit_only",
        original_receipt_unchanged=unchanged,
        native_execution_performed=False,new_solver_steps=0)
    return result
~~~

现有verify_numerical_reset直接使用entry.prior.numeric和scheduler.native_artifact；不引入entry.prior.sha。producer_profile继续校验原producer路径与本地锁定源码；缺原源或不一致就失败，不替换历史hash。重审脚本不调用native启动/worker/注册函数，不重置任何预算，不更新registry中的原scheduler结果。

### 3.3 完整重放、回归及只读封存

- [ ] 完整测试：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_retained_pinch_v3.py --tb=short
~~~

当前scaffold29项加新增17项，预期46 passed；以实际pytest收集输出为准，不抄历史数字。完整真实保留账本重放与小型格式fixture分开报告。原AttributeError隔离复现测试应仍通过，证明旧消费运行没有被改写。

- [ ] 回归：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_mesh_reset_numeric_state.py tests/test_mesh_reset_numeric_worker.py tests/test_mesh_reset_health_worker.py tests/test_flexible_health_postexit.py tests/test_dh116_dual_pinch_postexit.py --tb=short
~~~

预期exit0，实际计数实施后记录；pxr错误必须先修正启动环境再判断源。

- [ ] 私有新目录封存独立重审JSON，原receipt前后hash必须一致。此命令只运行CPU重审，没有新native：

~~~bash
EVIDENCE_ROOT=/home/xk/.local/share/dh116-autonomous-3c80166/evidence
REAUDIT_DIR="$(mktemp -d "$EVIDENCE_ROOT/pinch-v3-independent-reaudit-20261115-XXXXXX")"
RECEIPT=/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-pinch-fast-evidence-native/f0b5bc8cd80210db27052f4f96f435c7ce48a7ff23d53fd533a53fd3ed63502f/attempt-2ynpy5si/receipt.json
sha256sum "$RECEIPT" > "$REAUDIT_DIR/original-before.sha256"
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B scripts/verify_retained_dh116_pinch_v3.py --receipt "$RECEIPT" --output "$REAUDIT_DIR/reaudit.json"
sha256sum "$RECEIPT" > "$REAUDIT_DIR/original-after.sha256"
cmp "$REAUDIT_DIR/original-before.sha256" "$REAUDIT_DIR/original-after.sha256"
sha256sum "$REAUDIT_DIR/reaudit.json"
~~~

预期CLI和cmp均exit0。JSON须保留original_scheduler_status=failed、native_qualification=false、teacher_admission=false、new_physics_steps=0、new_solver_steps=0、ppo_updates=0。失败时留私有诊断，不能重跑相同已消费identity的native或改原账本使其通过。

- [ ] 保护核对、diff审查和精确提交：

~~~bash
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-postexit-replay-20261114 diff --check
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-postexit-replay-20261114 add -- scripts/verify_retained_dh116_pinch_v3.py tests/test_retained_pinch_v3.py
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-postexit-replay-20261114 diff --cached --stat
git -c safe.directory=/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-postexit-replay-20261114 commit -m "Add independent retained Pinch source re-audit"
~~~

只提交两文件；不提交原始日志、JSON、账本、模型。root集成代码及独立重审摘要，原失败scheduler状态保持，不能加入Teacher/Student成功池。

## 自审

- 三项独立任务，各有RED、完整真实方法补丁、GREEN、旧路径回归、精确git add/commit范围。
- R06新scope不自动改变旧入口；D阶段新recipe必须冻结新scope源码并独立验证自然退出。不能称当前历史链已切换。
- R07真实Gf projection决定符号正确，不能仅测试数学互逆；中心默认digest不改。
- R08旧源码、receipt、消耗预算均只读；新重审源清单和结果身份独立。
- native0次、optimizer更新0次；240步诊断不能替代2400步Hold，Teacher成功不是PPO探索的循环前置。
- 本计划起草阶段只写Markdown，没有实现生产代码、测试或commit。实施后的实际测试数量、SHA、日志路径必须重新记录。


## R08补充：用户PPO详细文档§2.3的真实fast替换上下文验收

本节按2026-11-15用户文档补充，优先于Task3.2中旧verify_saved_run代码和Task3.3旧预期计数；R06/R07不变。前版Markdown已保存为lifecycle-camera-sealing-plan-before-ppo-doc-v1.md。用户文档来源：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/ppo-reward-doc-20261115/extracted.txt，段62–77。

确认旧numeric wrapper存在两处错误：父成功分支访问fast entry缺少的entry.numeric；父失败但artifact存在时访问entry.prior.sha，目标numeric verify也没有sha。原测试父audit整体mock失败，没有验证真实成功分支。不得通过修改冻结旧wrapper修复；新独立重审固定直接依赖numeric模块和scheduler.file_hash。原旧wrapper错误必须可以在真实上下文继续复现，以证明没有重贴旧任务身份。

### R08-A：新增真实parent audit的五类artifact测试

- [ ] 在tests/test_retained_pinch_v3.py追加以下完整测试，所用ROOT、RECEIPT由Task3.1定义。完整copy仅在pytest临时目录；五类变体不编辑原receipt或原产物。保留真实parent.audit_outcome，不能mock父审查返回值。

~~~python
import shutil

@pytest.mark.parametrize("case",["success","failure","missing","corrupt","numeric_changed"])
def test_real_fast_numeric_context_preserves_actual_parent_audit(tmp_path,case):
    assert hasattr(replay,"audit_retained_numeric"),"stable retained numeric audit missing"
    receipt=json.loads(RECEIPT.read_bytes())
    original=Path(receipt["output"]);output=tmp_path/"native"
    shutil.copytree(original,output)
    assert output.resolve().is_relative_to(tmp_path.resolve())
    assert output.resolve()!=original.resolve()
    artifact=output/replay.entry.NUMERIC_ARTIFACT
    if case=="failure":
        path=output/"result.json";value=json.loads(path.read_bytes())
        value["failure"]="injected parent lifecycle failure"
        path.write_text(json.dumps(value)+"\n")
    elif case=="missing":artifact.unlink()
    elif case=="corrupt":artifact.write_bytes(b"{")
    elif case=="numeric_changed":
        value=json.loads(artifact.read_bytes());value["byte_equal"]=False
        artifact.write_text(json.dumps(value)+"\n")
    numeric_wrapper=legacy.prior
    with replay.producer_profile(ROOT,original),replay.entry.identity_scope(),bindings(
        (numeric_wrapper,"entry",replay.entry),(replay.parent,"entry",replay.entry),
        (replay.entry.operation,"VERSION",replay.entry.VERSION),
        (replay.entry.operation,"SCOPE",replay.entry.SCOPE)):
        assert numeric_wrapper.entry is replay.entry
        assert not hasattr(numeric_wrapper.entry,"numeric")
        assert not hasattr(numeric_wrapper.entry.prior,"sha")
        result=replay.audit_retained_numeric(output,RECEIPT.parent/"outer_exit.json","explicit")
    assert result["native_qualification"] is result["teacher_admission"] is False
    if case=="success":
        assert result["lifecycle_qualified"] is True,result.get("reason")
        assert result["numeric_state_audit"]["numeric_state_unchanged"] is True
        assert len(result["numeric_state_artifact_sha256"])==64
    else:
        assert result["lifecycle_qualified"] is False,result
        assert result["numeric_state_audit"] is None
        assert result.get("reason") or result.get("numeric_sealing_error")
    if case=="failure":
        assert len(result["numeric_state_artifact_sha256"])==64
    if case=="missing":
        assert result["numeric_state_artifact_sha256"] is None

@pytest.mark.parametrize("parent_success,missing_attribute",[(True,"numeric"),(False,"sha")])
def test_consumed_wrapper_two_real_context_errors_remain_reproducible(
        tmp_path,parent_success,missing_attribute):
    receipt=json.loads(RECEIPT.read_bytes())
    original=Path(receipt["output"]);output=tmp_path/"native"
    shutil.copytree(original,output)
    if not parent_success:
        path=output/"result.json";value=json.loads(path.read_bytes())
        value["failure"]="injected parent lifecycle failure"
        path.write_text(json.dumps(value)+"\n")
    numeric_wrapper=legacy.prior
    with replay.producer_profile(ROOT,original),replay.entry.identity_scope(),bindings(
        (numeric_wrapper,"entry",replay.entry),(replay.parent,"entry",replay.entry),
        (replay.entry.operation,"VERSION",replay.entry.VERSION),
        (replay.entry.operation,"SCOPE",replay.entry.SCOPE)):
        with pytest.raises(AttributeError,match="has no attribute '"+missing_attribute+"'"):
            numeric_wrapper.audit_outcome(output,RECEIPT.parent/"outer_exit.json","explicit")
~~~

五类组合：真实父成功+有效numeric；真实父失败+numeric存在；numeric缺失；损坏JSON；合法JSON但实际numeric审计摘要变动。后两类必须进入真实validate_numeric_reset_audit拒绝，不能用exists或producer bool代替。旧wrapper两条错误测试仅用于确证冻结历史行为，新的独立审查才是修复路径。

- [ ] RED先运行新五类测试：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_retained_pinch_v3.py -k real_fast_numeric_context --tb=short
~~~

预期5 failed，原因stable retained numeric audit missing。真实parent的USD依赖仍由scripts/python.sh提供。

### R08-B：新增稳定audit方法并替换独立总入口

- [ ] scripts/verify_retained_dh116_pinch_v3.py新增直接依赖：

~~~python
from go2_pvcnn.control.dh116 import tabletop_mesh_reset_numeric_state as stable_numeric
~~~

- [ ] verify_numerical_reset整体改为直接依赖稳定模块，不再走entry.prior.numeric属性链：

~~~python
def verify_numerical_reset(output):
    artifact=scheduler.native_artifact(output,entry.NUMERIC_ARTIFACT)
    reset=entry.operation._read(scheduler.native_artifact(
        output,entry.operation.evidence.RESET_FILE))
    readback=reset["readback"]
    receipt={**readback["completed_reset_receipt"],
        "mesh_state_readback":{k:v for k,v in readback.items()
                              if k!="completed_reset_receipt"}}
    return stable_numeric.validate_numeric_reset_audit(
        entry.operation._read(artifact),reset_receipt=receipt)
~~~

- [ ] 新增以下完整独立audit方法。entry动态元数据仍正确进入真实父函数，但numeric数学和哈希不从动态entry取：

~~~python
def audit_retained_numeric(output,outer_exit,cleanup):
    output=Path(output)
    with entry.identity_scope(),bindings(
        (entry.operation,"VERSION",entry.VERSION),
        (entry.operation,"SCOPE",entry.SCOPE),(parent,"entry",entry)):
        result=parent.audit_outcome(output,outer_exit,cleanup)
    result=dict(result);numeric=None;digest=None
    try:
        artifact=scheduler.native_artifact(output,entry.NUMERIC_ARTIFACT)
        digest=scheduler.file_hash(artifact)
        if result.get("lifecycle_qualified") is True:
            numeric=verify_numerical_reset(output)
    except (OSError,ValueError,KeyError,TypeError,IndexError,RuntimeError,AttributeError) as error:
        result.update(lifecycle_qualified=False,
            numeric_sealing_error=type(error).__name__+": "+str(error))
    return dict(result,numeric_state_audit=numeric,numeric_state_artifact_sha256=digest,
        qualification_scope=entry.SCOPE,native_qualification=False,
        teacher_admission=False,grasp_qualified=False,contact_success_qualified=False)
~~~

- [ ] 本节完整verify_saved_run取代Task3.2中同名方法。正式独立封存仍先重验原intent/approval/recipe/hash和outer；不能把inner变体测试的副本当成合格原artifact：

~~~python
def verify_saved_run(project,receipt_path):
    project=Path(project)
    receipt,output,outer,generic=verify_original_attempt(project,receipt_path)
    with producer_profile(project,output):
        audit=audit_retained_numeric(output,outer,"explicit")
    numeric=audit["numeric_state_audit"]
    result=assemble_verified_sources(receipt,generic,audit,numeric)
    names=tuple(dict.fromkeys(("scripts/verify_retained_dh116_pinch_v3.py",
        *parent.AUDIT_SOURCES,*entry.FILES)))
    sources={}
    for name in names:
        path=scheduler.native_artifact(project,name)
        sources[name]=scheduler.file_hash(path)
    unchanged=scheduler.file_hash(Path(receipt_path))==RECEIPT_SHA
    if not unchanged:raise ValueError("original failed receipt changed during replay")
    result.update(replay_source_sha256=sources,
        qualification_scope="retained_pinch_v3_saved_physical_source_reaudit_only",
        original_receipt_unchanged=unchanged,
        native_execution_performed=False,new_solver_steps=0)
    return result
~~~

### R08-C：执行新增验收和完整回归

- [ ] 用共同环境重跑完整tests/test_retained_pinch_v3.py；旧29+Task3新增17+本节7=预期53项，实际以收集结果为准。不得报告未经执行的通过数量。
- [ ] 原Task3.3的回归、CLI重审、原receipt前后SHA一致和精确两文件stage保持。旧wrapper/source/recipe/失败receipt仍不改。
- [ ] 报告分列：两处冻结历史属性错误被真实上下文复现；五类新独立audit测试；完整原工件重审；原scheduler仍failed；新物理步0、更新0。


## 投影资格边界补充

中心默认的数值与配置不变；非中心投影修复后的原生结果需要使用新producer源SHA重新核验。旧非中心相机产物不能因同一K摘要自洽就自动资格化为新投影路径。CPU Gf投影正确仍不代替双目原生精度资格。
