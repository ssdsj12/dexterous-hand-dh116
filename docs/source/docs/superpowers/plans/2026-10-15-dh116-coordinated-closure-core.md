# DH116 Coordinated Closure Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. 用户已选Inline Execution，不重新询问、不派子代理。Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已批准D4协同抓取范围内，验证抵消闭指切向拖动的10主动关节速度分配器。

**Architecture:** 每侧使用两指腹3D实际点Jacobian（Panda7+拇指2+食指1），以及掌心角速度Jacobian。两个点只沿各自法向增减夹力，掌心角速度目标为零，最小范数解统一缩放限速；余三指不参与。这里只实现/验证纯计算核心，不接当前pilot，不称为安全路径或真实抓取。

**Tech Stack:** NumPy、现有USD FK/实际点Jacobian合同、pytest。

## Global Constraints

- 继承[批准设计](../specs/2026-10-15-dh116-pad-jacobian-wire-grasp-design.md)：仿真近似、CPU单核、完整碰撞保护、无wire绑定、不沿用O6硬件参数。
- 每侧输出顺序Panda1..7,finger11,finger12,finger21；dt≤100µs；臂控制限速.05rad/s（低于现有.15执行上限）、手.02rad/s不变。
- 夹力目标和摩擦不因本改动改变；非目标1mm、滑移2mm、保持/抬升门不改变。
- 工作树已有大量未提交变更，本计划不提交/推送。持续目标不因纯计算完成而标完成。

### Task 1: Coupled velocity allocation and evidence

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/dh116/coordinated_closure.py`
- Create: `Go2Pvcnn/tests/test_dh116_coordinated_closure.py`

**Interfaces:** consumes `point_jacobians:(2,3,10)`, `normals:(2,3)`, `palm_angular_jacobian:(3,10)`, `loads_n:(2,)`, `dt`; produces finite `delta_q:(10,)`. `target_loads_n`显式两目标，非负有限载荷，单位法向、有限矩阵、正dt/目标，欠秩拒绝。

- [x] Step 1: 写如下最小先失败测试，并追加旋转协变、dt缩放、坏输入和实际左右USD测试。

```python
def test_tangent_and_palm_rotation_cancelled():
    J=np.zeros((2,3,10));J[0,:,3:6]=.03*np.eye(3);J[1,:,6:9]=.03*np.eye(3)
    W=np.zeros((3,10));W[:,:3]=np.eye(3)
    n=np.array([[1.,0,0],[-1.,0,0]])
    dq=coordinated_closure_increment(J,n,W,[.1,1.5],1e-4,target_loads_n=[1.,1.])
    v=np.einsum('ijk,k->ij',J,dq/1e-4)
    assert v[0,0]>0 and v[1,0]>0
    np.testing.assert_allclose(v[:,1:],0,atol=1e-12)
    np.testing.assert_allclose(W@(dq/1e-4),0,atol=1e-12)
    assert np.max(np.abs(dq[:7]/1e-4))<=.05+1e-12
    assert np.max(np.abs(dq[7:]/1e-4))<=.02+1e-12
```

- [x] Step 2: `taskset -c 0 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_dh116_coordinated_closure.py -q`先验证缺少模块/函数失败。
- [x] Step 3: 实现有限值/维度/单位法向/正目标和dt校验后，执行以下分配；不得增加机械臂执行接线。

```python
error=target-load
speed=np.clip(.01*error,-.002,.002)
speed[np.abs(error)<=.03]=0.
A=np.vstack((J.reshape(6,10),.05*W))
b=np.r_[(n*speed[:,None]).reshape(6),np.zeros(3)]
if np.linalg.svd(A,compute_uv=False)[-1]<1e-5:
    raise ValueError('coordinated closure Jacobian is singular')
velocity=np.linalg.lstsq(A,b,rcond=None)[0]
scale=max(1.,float(np.max(np.abs(velocity[:7])))/.05,
             float(np.max(np.abs(velocity[7:])))/.02)
return velocity*(dt/scale)
```

- [x] Step 4: 使用原名义切触姿态，左右独立USD模型点J及掌心祖先轴构造W，核对切向速度/角速度残差≤1e-10、正法向闭指、速率上限。运行新测试及原closure/point Jacobian回归，记录日志与dashboard。

审阅：本计划只覆盖D4控制候选计算，明确不覆盖运行时安全、实际保持/抬升或完整任务。实际接线必须以当前pilot结果为依据，再单独验证完整源保护和负载轨迹，不由本计划测试通过直接晋级。

## Task 2: Explicit runtime pilot (added after pilot4 ended)

pilot4在9807步被左拇指非前区1mm保护拒绝，四点最长.4013s，材料滑移1.684mm，无目标夹力保持。本任务仅改变闭指速度分配，初态、力目标、物理参数与保护不变。已测闭指切向速度约.50–1.60mm/s，验证臂补偿是否能降低该漂移；不预设成功。

- [x] 先写CLI默认关闭、依赖closure-pilot、使用实际掌心角Jacobian与命名臂列、arm/hand目标接线测试，确认RED。
- [x] 添加显式`--coordinated-closure`，累计7臂+3手关节目标；其余手指固定。100ms落定后启用。记录run_config；保留所有每步源碰撞、区域、速度、力矩和反力门。
- [x] 运行新核心与运行接线/已有闭指回归。测试通过后在全新目录运行GPU0/core0物理对照并保存源码快照。
- [ ] 汇总实际载荷、有效接触持续时间、材料滑移、碰撞保护与点Jacobian；只有实际通过才能推进保持/抬升；初始切触pilot不能顶替完整连续接近。

## Task 3: Same-section material evidence contract

批准设计要求两个分离材料抓取区间、四个有效指腹通道，不要求同手两指捏不同轴向材料点。直接同截面候选暴露现有观察器四ID互异的不必要限制；此任务只修正纯观察器，不启用实际抬升。

- [x] 先失败测试：接受(36,36,4,4)及旧(38,34,5,3)；仍要求四有效通道和原保持时长；左右ID集合重叠拒绝；同手同ID的掌心坐标不一致拒绝；同手两行共同滑移>2mm仍abort。
- [x] 构造器改为左右材料ID集合互不重叠，保留4通道/合法ID/实际approach门；update加入同ID同手坐标一致性检查，不改变载荷、非分离、保持/抬升和滑移阈值。
- [x] 运行观察器/材料坐标及DH116回归，记录通过数和未接运行时边界。17专项/347合并通过，见同截面对捏日志；仍未实际保持/抬升。
