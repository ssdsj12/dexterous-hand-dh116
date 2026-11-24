"""Teacher native and Student offline/train/inference entry orchestration; depends on all lower layers and does not define solver/control algorithms."""
from copy import deepcopy as stude_train__deepcopy
import argparse as stere_infer__argparse
import json as tenso_colle__json
import argparse as teach_train__argparse, importlib.util, json as teach_train__json, os, subprocess, sys, uuid
from pathlib import Path as stude_train__Path
from copy import deepcopy as stere_infer__deepcopy
from time import perf_counter as tenso_colle__perf_counter
from functools import partial
import json as stude_train__json
import hashlib
import torch as tenso_colle__torch
from pathlib import Path as teach_train__Path
import torch as stude_train__torch
import io
from dh116.rl.device import DeviceDeformablePPOTrainer as DeviceDeformablePPOTrainer
from time import perf_counter as teach_train__perf_counter
from dh116.rl.student_data import StereoBCTrainer as StereoBCTrainer
import json as stere_infer__json
from dh116.rl.ppo import PPOConfig
from dh116.env.contracts import runti_io__ROOT as teach_train__ROOT
from dh116.rl.student_data import save_actor_package
import math
from dh116.rl.teacher import pack_shared_tensor_inputs
from dh116.env.contracts import runti_io__digest as digest
from dh116.rl.student_data import load_deployment_actor
from pathlib import Path as stere_infer__Path
from dh116.rl.teacher import TensorNativeTeacher
from dh116.env.contracts import runti_io__file_sha256 as teach_train__file_sha256
from dh116.rl.student_data import export_actor
import torch as stere_infer__torch
from dh116.physics.runtime import NativeTensorPhysicsRuntime
from dh116.env.contracts import write_new
from dh116.env.contracts import make_contract
from dh116.physics.vision import StereoCalibration as StereoCalibration
from dh116.physics.runtime import RawTableClearance
from dh116.env.contracts import checked_file
from dh116.env.contracts import stere_contr__file_sha256 as stude_train__file_sha256
from dh116.rl.student_data import DEPLOYMENT_VERSION
from dh116.physics.runtime import SIDES

def load_candidate(profile):
    name = 'ropesim._ropesim'
    binary = teach_train__Path(profile.binary_path).resolve(strict=True)
    loaded = sys.modules.get(name)
    if loaded is not None:
        if teach_train__Path(loaded.__file__).resolve() != binary:
            raise RuntimeError('another native ropesim binary already loaded')
        return loaded
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
from dh116.env.contracts import write_json_new as stude_train__write_json_new
from dh116.rl.student_data import _validate_actor_contract
from dh116.physics.runtime import CLEARANCE_M

def train(sim, owners, trace, project, prepared, archive, lanes, output, gpu, end, *, args, reset, profile, height):
    import torch
    from dh116.physics.solver import make_solver as make_solver
    from dh116.env.contracts import write_new
    from dh116.hand.articulation import prepare_regex_task
    _teach_train_import__NativeTensorCollector = NativeTensorCollector
    from dh116.physics.geometry import geometry_measurement_identity
    from dh116.physics.solver import TensorRodOwner
    (batch, seed, _, bindings, initial) = prepare_regex_task(sim, owners, trace, project, prepared, archive, lanes, seed_height=height)
    names = tuple((row['name'] for row in archive.mesh_plan['entries']))
    backend = load_candidate(profile)
    solver = make_solver(backend, archive.mesh_plan, reset, initial['env_origins'].float().cpu().numpy(), style=archive.style, capacity=41 * len(names) * 12, empty_lane=False)
    owners.keep('shared_rod_solver', solver)
    run_uuid = uuid.uuid4().hex
    runtime = TensorRodOwner(solver, run_uuid=run_uuid, physics_step=sim.current_time_step_index, episodes=tuple((f'{run_uuid}:{i}:1' for i in range(len(lanes)))), mesh_names=names, device='cuda:0')
    torch.manual_seed(args.policy_seed)
    torch.cuda.manual_seed_all(args.policy_seed)
    collector = _teach_train_import__NativeTensorCollector(project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile)
    probe = bool(args.probe_control_windows)
    if probe and len(lanes) == 2:
        collector.arm_probe(output)
    total = 2 if probe else args.updates * 32
    iterations = []
    window_seconds = []
    window_phases = []
    rollout_start = teach_train__perf_counter()
    try:
        for index in range(total):
            started_window = teach_train__perf_counter()
            reward = collector.window(end, probe_reset=probe and len(lanes) == 2 and (index == 0))
            window_seconds.append(teach_train__perf_counter() - started_window)
            window_phases.append(dict(collector.timings))
            if collector.trainer.rollout.full:
                if probe:
                    raise RuntimeError('probe must not update a policy')
                torch.cuda.synchronize()
                start = teach_train__perf_counter()
                metrics = collector.trainer.update()
                torch.cuda.synchronize()
                finish = teach_train__perf_counter()
                if metrics['minibatch_updates'] <= 0:
                    raise RuntimeError('PPO rejected every minibatch')
                update = collector.trainer.counters['updates']
                record = dict(update=update, sdk_step=runtime.physics_step, metrics=metrics, rollout_seconds=start - rollout_start, optimizer_seconds=finish - start, window_seconds=window_seconds[-32:], window_phases=window_phases[-32:])
                path = output / f'tensor_{archive.style}_update_{update:04d}.pt'
                record['checkpoint_sha256'] = save_snapshot(path, collector.trainer, args, profile, seed)
                record['iteration_seconds'] = teach_train__perf_counter() - rollout_start
                write_new(output / f'tensor_update_{update:04d}.json', record)
                iterations.append(record)
                rollout_start = teach_train__perf_counter()
            print('DH116_TENSOR_WINDOW', index + 1, 'sdk', runtime.physics_step, 'reward', reward, flush=True)
    except BaseException as exc:
        collector.ctrl.invalidate()
        write_new(output / 'tensor_failure.json', dict(error=str(exc), type=type(exc).__name__, completed_windows=collector.controls, sdk_step=sim.current_time_step_index, rod_sdk=runtime.physics_step, updates=collector.trainer.counters['updates'], teacher_admission=False))
        raise
    updates = collector.trainer.counters['updates']
    if runtime.physics_step != end or collector.controls != total or updates != (0 if probe else args.updates) or (probe and runtime.resets != ([1, 2] if len(lanes) == 2 else [1] * len(lanes))):
        raise ValueError('incomplete actual bounded run')
    return dict(throughput_probe_only=probe, updates=updates, controls=total, environments=len(lanes), sdk_step=end, iterations=iterations, window_seconds=window_seconds, window_phases=window_phases, sdk_initialization_steps=2, physics_hz=480, control_hz=60, rod_substeps=12, articulations=2, action_size=24, run_uuid=run_uuid, geometry_backend=getattr(collector.geometry, 'backend', None), geometry_measurement=geometry_measurement_identity(collector.geometry), actual_measurement_backend=geometry_measurement_identity(collector.geometry)['actual_measurement_backend'], measurement_scope='bounded_diagnostic', measurement_native_evidence=args.measurement_native_evidence, native_readback_mode=getattr(batch, 'readback_mode', 'raw'), source_cache_metadata=batch.readback_metadata() if hasattr(batch, 'readback_metadata') else None, readback_audit=batch.readback_audit() if hasattr(batch, 'readback_audit') else None, reset_ids=runtime.resets, reset_checks=collector.reset_proofs, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), teacher_admission=False, grasp_qualified=False)
from dh116.env.contracts import require_aligned
from dh116.rl.student_data import _validate_state
from dh116.physics.runtime import collision_clearance

def save_snapshot(path, trainer, args, profile, seed):
    """Use the separately verified v2 optimizer/RNG restart format."""
    from dh116.env.contracts import runti_io__digest as digest
    from dh116.rl.device import save_checkpoint
    recipe = digest(dict(schema='dh116-tensor-native-ppo-recipe-v2', model=trainer.model.model_identity, rollout_steps=trainer.rollout_steps, num_envs=trainer.num_envs, receipt_sha256=args.receipt_sha256, profile_sha256=profile.digest, seed_sha256=seed['sha256'], policy_seed=args.policy_seed, physics_hz=480, control_hz=60, solver_substeps=12, action_size=24, geometry_backend=os.environ.get('DH116_TENSOR_GEOMETRY_BACKEND', 'eager'), readback_backend=os.environ.get('DH116_NATIVE_READBACK_BACKEND', 'raw'), geometry_measurement=__import__('dh116.physics.geometry', fromlist=['CudaGeometryProfile']).CudaGeometryProfile.load().identity() if os.environ.get('DH116_TENSOR_GEOMETRY_BACKEND') == 'cuda_cabi' else None))
    result = save_checkpoint(path, trainer, recipe_sha256=recipe, receipt_sha256=args.receipt_sha256, backend_profile=profile, seed_sha256=seed['sha256'], policy_seed=args.policy_seed)
    return result['sha256']
from dh116.rl.student_data import DAggerDataset
from dh116.rl.student_data import tensor_state_sha256
from dh116.physics.runtime import mesh_reset

def worker(args, end):
    from dh116.env.scene import load_task_scene
    from dh116.env.lifecycle import run_native_session
    from dh116.hand.articulation import OWNED_NAMES
    from dh116.physics.solver import TensorBackendProfile
    from dh116.env.lifecycle import NATIVE_LOCK
    from dh116.env.lifecycle import native_file_lock
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('explicit matching physical GPU required')
    profile = TensorBackendProfile.load()
    (archive, reset) = load_task_scene(args.grasp)
    memory = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if memory > 512:
        raise RuntimeError('physical GPU is occupied')
    task = teach_train__json.loads((teach_train__ROOT / 'configs/tasks' / (args.grasp + '.json')).read_text())
    height = task.get('startup_height')
    if height is not None:
        checked_file(teach_train__ROOT / height['evidence_path'], height['evidence_sha256'])
        height = {k: height[k] for k in ('delta_z_m', 'evidence_sha256')}
    with native_file_lock(NATIVE_LOCK):
        return run_native_session(teach_train__ROOT, archive, args.gpu, args.num_envs, args.output, max_sdk_step=end, author_and_exercise=partial(train, args=args, reset=reset, profile=profile, height=height), owned_names=OWNED_NAMES, runtime_source_audit=True)
from dh116.rl.student_data import StereoDAggerTrainer
from dh116.physics.vision import StereoDepth as StereoDepth
from dh116.physics.runtime import probe_copy

def validate_native_options():
    if os.environ.get('DH116_NATIVE_READBACK_BACKEND', 'raw') not in ('raw', 'official_cache'):
        raise ValueError('readback must be raw or official_cache before App startup')
    if os.environ.get('DH116_TENSOR_GEOMETRY_BACKEND', 'eager') not in ('eager', 'cuda_cabi'):
        raise ValueError('geometry must be eager or cuda_cabi; compilation disabled')
from dh116.rl.student_model import StereoHistoryBatch as StereoHistoryBatch
from dh116.rl.student_model import SensorBatch
from dh116.physics.runtime import require_substep_guard

def train_teacher_main(argv=None):
    parser = teach_train__argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grasp', choices=('pinch', 'tripod', 'power'), required=True)
    parser.add_argument('--num-envs', type=int, choices=(2, 128), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--outer-timeout-s',type=int,default=900)
    parser.add_argument('--output', type=teach_train__Path, required=True)
    parser.add_argument('--policy-seed', type=int, default=20261123)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe-control-windows', type=int, choices=(2,))
    group.add_argument('--updates', type=int)
    parser.add_argument('--native-evidence', type=teach_train__Path, action='append', default=[])
    parser.add_argument('--native-evidence-sha256', action='append', default=[])
    parser.add_argument('--worker', action='store_true', help=teach_train__argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 240<=args.outer_timeout_s<=900:
        parser.error('outer timeout must be 240..900 seconds; physical budget is independent')
    from dh116.env.contracts import validate_runtime_execution
    from dh116.env.contracts import ProjectPaths
    validate_runtime_execution()
    validate_native_options()
    task = teach_train__json.loads((teach_train__ROOT / 'configs/tasks' / (args.grasp + '.json')).read_text())
    args.receipt_sha256 = task['scene_input']['sha256']
    args.measurement_native_evidence = None
    if args.updates is not None:
        if args.updates < 2:
            parser.error('at least two updates required')
        from dh116.physics.geometry import verify_measurement_native_receipts
        if len(args.native_evidence) != len(args.native_evidence_sha256):
            parser.error('paired actual native evidence and SHA required')
        args.measurement_native_evidence = verify_measurement_native_receipts(list(zip(args.native_evidence, args.native_evidence_sha256)), style=args.grasp)
    total = args.probe_control_windows or args.updates * 32
    end = 2 + 8 * total
    if args.worker:
        worker(args, end)
        return 0
    output = ProjectPaths(teach_train__ROOT).output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    native = output / 'native'
    native.mkdir()
    inventory = {str(p.relative_to(teach_train__ROOT)): teach_train__file_sha256(p) for p in (teach_train__ROOT / 'src/dh116').rglob('*') if p.suffix in ('.py', '.cu') and p.is_file()}
    write_new(output / 'attempt.json', dict(schema='dh116-production-native-attempt-v1', source_sha256=inventory, grasp=args.grasp, num_envs=args.num_envs, controls=total, sdk_steps=end, ppo_updates=args.updates or 0, scene_sha256=args.receipt_sha256))
    command = [sys.executable, str(teach_train__ROOT / 'scripts/train_teacher.py'), *(sys.argv[1:] if argv is None else argv), '--output', str(native), '--worker']
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID')
    with (output / 'native.log').open('wb') as log:
        child = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        write_new(output / 'pid.json', dict(pid=child.pid, command=command, gpu=args.gpu))
        try:
            code = child.wait(timeout=args.outer_timeout_s)
        except subprocess.TimeoutExpired:
            write_new(output / 'timeout.json', dict(pid=child.pid, timeout_s=args.outer_timeout_s, natural_exit_verified=False, automatic_signal_sent=False))
            raise RuntimeError('native process exceeded budget; evidence retained, no automatic kill') from None
    write_new(output / 'process_exit.json', dict(returncode=code))
    if code:
        raise RuntimeError(f"native worker exited {code}; see {output / 'native.log'}")
    lifecycle = teach_train__json.loads((native / 'result.json').read_text())
    close = teach_train__json.loads((native / 'app_close_returned.json').read_text())
    actual = lifecycle['acquisition']
    if close.get('app_close_returned') is not True or close.get('close_mode') != 'normal' or lifecycle.get('failure') is not None or (lifecycle.get('close_mode') != 'normal') or (lifecycle.get('lifetime_released_before_close') is not True) or (lifecycle.get('physics_steps') != end) or (actual.get('sdk_step') != end) or (actual.get('controls') != total) or (actual.get('updates') != (args.updates or 0)):
        raise ValueError('complete acquisition and verified natural close required')
    from dh116.physics.geometry import CudaGeometryProfile
    write_new(output / 'native_result.json', dict(actual, grasp=args.grasp, exit_code=code, process_exit_verified=True, scene_sha256=args.receipt_sha256, geometry_manifest_sha256=CudaGeometryProfile.load().manifest_sha256, runtime_graph_compilation=False, close_mode='normal', teacher_admission=False))
    return 0
from dh116.rl.student_model import StereoAsymmetricPolicy
from dh116.rl.student_model import StereoHistory

class NativeTensorCollector(NativeTensorPhysicsRuntime):
    """Keep Teacher sampling/value/rollout separate from shared physical stepping."""

    def __init__(self, project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile):
        self._configure_physics(project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile)
        cfg = tenso_colle__json.loads((project / f'configs/training/teacher_{self.style}.json').read_text())
        self.model = TensorNativeTeacher(cfg, backend_profile=profile, num_envs=self.n, reward_config=self.config).to(self.device)
        pp = self.config.ppo
        self.trainer = DeviceDeformablePPOTrainer(self.model, PPOConfig(lr=pp['learning_rate'], gamma=pp['gamma'], lam=pp['gae_lambda'], clip=pp['clip'], epochs=pp['epochs'], minibatches=pp['minibatches'], value_coef=pp['value'], entropy_coef=pp['entropy'], max_grad_norm=pp['max_grad_norm'], target_kl=pp['target_kl'], kl_backtrack_steps=pp['kl_backtrack_steps']), rollout_steps=32, num_envs=self.n, device=self.device)
        self._initialize_episode()

    def observation(self):
        stamp = self.source._boundary()
        (cv, cm, ci) = self.source.contexts()
        packet = pack_shared_tensor_inputs(self.state['q'], self.state['qd'], joint_names=self.joints, body_link_pose_w=self.state['base_link_pose_w'], body_com_velocity_w=self.state['base_link_com_velocity_w'], root_seed_w=self.initial['root_seed_w'], env_origins=self.origins, rod_positions_w=self.nodes, rod_velocities_w=self.velocity, rest_lengths_m=self.source.rest_lengths_m, task_context=cv, task_context_valid=cm, source_identity=stamp, context_identities=ci, backend_profile_sha256=stamp.profile_sha256, grasp=self.style, scope=self.source.scope)
        return packet.for_teacher(self.model, current_identity=self.source._boundary())

    def window(self, end, *, probe_reset=False):
        try:
            self._assert_usable()
            return self._window_impl(end, probe_reset=probe_reset)
        except BaseException:
            self._physical_failed = True
            raise

    def _window_impl(self, end, *, probe_reset=False):
        self.timings = {}
        phase_start = tenso_colle__perf_counter()
        if self.controls == 0:
            self.first_sdk = self.runtime.physics_step
        obs = self.observation()
        sample = self.trainer.collect_step(obs, obs)
        (actions, token) = (sample['actions'], sample['collection_token'])
        if probe_reset and tenso_colle__torch.equal(actions[0], actions[1]):
            raise ValueError('probe requires independent sampled 24D lane actions')
        self.timings['observation_policy_sample'] = tenso_colle__perf_counter() - phase_start
        phase_start = tenso_colle__perf_counter()
        tokens = (token,) * self.n
        policy = f'{self.trainer.session_id}:{self.trainer.policy_version}'
        self.begin_action_window(actions, token=token, policy_uids=(policy,) * self.n, end=end)
        self.timings['reward_window_begin'] = tenso_colle__perf_counter() - phase_start
        self.advance_action_window(actions, end)
        phase_start = tenso_colle__perf_counter()
        self.record_window_request()
        with tenso_colle__torch.no_grad():
            value = self.model.value(self.observation())
        timeout = tenso_colle__torch.zeros(self.n, device=self.device, dtype=tenso_colle__torch.bool)
        if probe_reset:
            timeout[1] = ~self.reward._terminal[1]
        full = len(self.trainer.rollout) + 1 == self.trainer.rollout_steps
        packet = self.complete_action_window(truncated=timeout, buffer_full=full)
        self.trainer.record_step(packet.reward.float(), packet.done, value, truncated=packet.truncated, bootstrap_mask=~packet.terminated, executed_actions=actions, collection_token=token)
        self.snapshot('window_' + str(self.controls), dict(schemas=tuple(self.reward.schemas), packet=packet, post_value=value, actions=actions))
        self.timings['value_and_reward_settlement'] = tenso_colle__perf_counter() - phase_start
        phase_start = tenso_colle__perf_counter()
        if self._reset_allowed:
            ids = self._reset_allowed
            self.reset(ids)
        self.timings['window_reset'] = tenso_colle__perf_counter() - phase_start
        return float(packet.reward.mean())
from dh116.rl.student_model import make_stereo_actor as stude_train__make_stereo_actor
from dh116.physics.wire import WirePointExtractor
from dh116.rl.student_model import make_student_trainer as _make_student_trainer
from dh116.physics.wire import stere_perce__TrackedWirePointExtractor as TrackedWirePointExtractor
from dh116.rl.student_model import normalize_network_config
from dh116.physics.vision import world_to_palms
from dh116.rl.student_data import validate_sealed_stereo_dataset
from dh116.env.contracts import stere_contr__ROOT as stere_infer__ROOT
from dh116.rl.student_data import batch_dataset as batch_dataset
from dh116.env.contracts import stere_contr__file_sha256 as stere_infer__file_sha256
from dh116.rl.student_data import split_by_initial_condition
from dh116.env.contracts import require_opt_in
from dh116.rl.student_data import save_dataset as save_dataset
from dh116.env.contracts import require_sha
from dh116.rl.student_data import PreActionState
from dh116.env.contracts import write_json_new as stere_infer__write_json_new
from dh116.rl.student_data import StepResult
from dh116.physics.vision import require_raw_processing
from dh116.rl.student_data import collect_step
from dh116.rl.student_model import make_stereo_actor as stere_infer__make_stereo_actor
from dh116.rl.student_data import require_teacher as require_teacher
from dh116.physics.wire import configured_segmentation_depth
from dh116.rl.student_data import bind_frozen_teacher
RAW_FIELDS = {'left_rgb', 'right_rgb', 'proprio', 'left_timestamp_s', 'right_timestamp_s', 'frame_id', 'now_timestamp_s'}

def require_stereo(qualification, camera_config, calibration_sha256):
    """Recheck existing raw stereo proof; calibration alone grants nothing."""
    if qualification is None or camera_config is None:
        raise ValueError('current qualified Stereo evidence and camera processing configuration required')
    from dh116.rl.qualification import verify_stereo_qualification
    from dh116.rl.qualification import require_qualified_stereo_processing
    from dh116.physics.vision import StereoCalibration as StereoCalibration
    verified = verify_stereo_qualification(qualification)
    require_qualified_stereo_processing(verified, camera_config)
    if StereoCalibration.from_dict(camera_config).sha256 != calibration_sha256:
        raise ValueError('qualified Stereo calibration differs from Student data')
    return verified

def _tensor_snapshot(path, *, expected_sha256=None, label):
    data = stere_infer__Path(path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and sha != require_sha(expected_sha256, label + ' SHA'):
        raise ValueError(label + ' SHA mismatch')
    return (stere_infer__torch.load(io.BytesIO(data), map_location='cpu', weights_only=True), sha)

def make_student_trainer(config, num_envs, device='cpu', rollout_steps=None, *, schema_only=False, teacher_qualification=None, source_sha256=None, stereo_qualification=None, camera_config=None, calibration_sha256=None):
    """Construct the existing PPO trainer only after current production admission."""
    if type(schema_only) is not bool:
        raise ValueError('explicit schema_only boolean required')
    if not schema_only:
        require_teacher(teacher_qualification, source_sha256)
        require_stereo(stereo_qualification, camera_config, calibration_sha256)
    trainer = _make_student_trainer(config, num_envs, device=device, rollout_steps=rollout_steps)
    trainer.student_admission = dict(schema_only=schema_only, native_task_qualified=False)
    return trainer

def _load_standalone_actor(path, calibration, *, expected_sha256=None):
    """Use the unchanged v1 contract and validate the same bytes used by inference."""
    (payload, sha) = _tensor_snapshot(path, expected_sha256=expected_sha256, label='Actor package')
    if not isinstance(payload, dict) or payload.get('version') != DEPLOYMENT_VERSION:
        raise ValueError('independent stereo deployment version required; legacy visual3844 forbidden')
    allowed = {'version', 'actor', 'actor_sha256', 'actor_contract', 'network_config', 'calibration_sha256', 'preprocessor'}
    if set(payload) != allowed:
        raise ValueError('actor-only package must contain no critic/Teacher/optimizer/env')
    (contract, network) = (payload['actor_contract'], payload['network_config'])
    if payload['calibration_sha256'] != calibration:
        raise ValueError('deployment calibration SHA mismatch')
    _validate_actor_contract(contract, network, calibration)
    if payload['preprocessor'] != contract['preprocessor']:
        raise ValueError('preprocessor contract mismatch')
    if payload['actor_sha256'] != tensor_state_sha256(payload['actor']):
        raise ValueError('actor tensor SHA mismatch')
    with stere_infer__torch.random.fork_rng(devices=[]):
        actor = stere_infer__make_stereo_actor(network)
    _validate_state(payload['actor'], actor)
    actor.load_state_dict(payload['actor'], strict=True)
    actor.deployment_contract = stere_infer__deepcopy(contract)
    actor.network_config = stere_infer__deepcopy(network)
    actor.calibration_sha256 = calibration
    actor.deployment_package_artifact = dict(path=str(stere_infer__Path(path).resolve()), sha256=sha)
    return actor.eval()

class SharedStudentCollector:
    """Adapt an injected shared runtime without taking Teacher action ownership.

    runtime.snapshot() returns PreActionState; runtime.step(actions) returns
    StepResult and advances the shared environment once. sensor_history(before)
    returns only StereoHistoryBatch. teacher_inputs(before) returns an exact
    {'inputs': six-key mapping, 'identities': list} bound to that same snapshot.
    Both callbacks must be read-only. The collector verifies the existing
    pre-action clocks, execution range and reset boundary around each step.
    """

    def __init__(self, runtime, *, sensor_history, teacher_inputs):
        if not all((callable(value) for value in (sensor_history, teacher_inputs, getattr(runtime, 'snapshot', None), getattr(runtime, 'step', None)))):
            raise TypeError('shared snapshot/step runtime and both explicit callbacks required')
        self.runtime = runtime
        self.sensor_history = sensor_history
        self.teacher_inputs = teacher_inputs

    @property
    def evidence_scope(self):
        return getattr(self.runtime, 'evidence_scope', None)

    def snapshot(self):
        state = self.runtime.snapshot()
        if type(state) is not PreActionState:
            raise TypeError('runtime snapshot must be PreActionState')
        before = state.clone()
        history = self.sensor_history(before.clone())
        if type(history) is not StereoHistoryBatch:
            raise TypeError('sensor_history callback must return StereoHistoryBatch')
        before.sensor_history = history.clone()
        return before.validate()

    def step(self, actions):
        return self.runtime.step(actions)

    def set_action_owners(self, owners):
        if any((owner != 'student' for owner in owners)):
            raise ValueError('shared Student collection cannot transfer action ownership')
        callback = getattr(self.runtime, 'set_action_owners', None)
        if callback is not None:
            callback(owners)

    def collect(self, student, teacher, *, steps, schema_only=False, teacher_qualification=None, source_sha256=None, stereo_qualification=None, camera_config=None, teacher_binding=None):
        if type(steps) is not int or steps < 1:
            raise ValueError('positive bounded collection steps required')
        if student is None:
            raise ValueError('Student-owned collection requires a Student actor')
        if not schema_only:
            require_teacher(teacher_qualification, source_sha256)
            require_stereo(stereo_qualification, camera_config, self.snapshot().identities[0]['calibration_sha256'])
        return [collect_step(self, student, teacher, teacher_probability=0.0, schema_only=schema_only, teacher_qualification=teacher_qualification, source_sha256=source_sha256, teacher_inputs=self.teacher_inputs, teacher_binding=teacher_binding) for _ in range(steps)]

class StereoRGBStudentRunner:

    def __init__(self, actor_package, camera_config, *, num_envs=1, device='cpu', max_image_age_s=1 / 60, expected_actor_package_sha256=None):
        data = stere_infer__Path(camera_config).read_bytes()
        config = stere_infer__json.loads(data.decode('utf-8'))
        self.calibration = StereoCalibration.from_dict(config)
        self.actor = _load_standalone_actor(actor_package, self.calibration.sha256, expected_sha256=expected_actor_package_sha256).to(device)
        self.processing = require_raw_processing(self.actor.deployment_contract, config)
        self.camera_config_sha256 = hashlib.sha256(data).hexdigest()
        self.depth = configured_segmentation_depth(StereoDepth.from_config(config), config, device)
        self.history = StereoHistory(num_envs, self.actor.network_config['history'], device=device)
        self.extractor = WirePointExtractor() if self.processing['perception']['algorithm'] == 'warm_colour_visible_simple_path_ZhangSuen_v1' else TrackedWirePointExtractor(point_sampling=self.processing['perception']['point_sampling'], ribbon_radius_px=self.processing['perception']['ribbon_radius_px'], observed_surface_capacity=self.processing['perception']['observed_surface_capacity'])
        (self.num_envs, self.device) = (num_envs, stere_infer__torch.device(device))
        if type(max_image_age_s) not in (int, float) or not math.isfinite(max_image_age_s) or max_image_age_s <= 0:
            raise ValueError('positive image freshness limit required')
        self.max_image_age_s = float(max_image_age_s)
        (self._frame, self._timestamp) = (-1, None)
        self.last_visible_geometry = None
        self.package_sha256 = self.actor.deployment_package_artifact['sha256']

    def _clear_temporal_state(self):
        self.history.reset(stere_infer__torch.ones(self.num_envs, device=self.device, dtype=stere_infer__torch.bool))
        self.extractor = TrackedWirePointExtractor(point_sampling=self.extractor.point_sampling, ribbon_radius_px=self.extractor.ribbon_radius_px, observed_surface_capacity=self.extractor.observed_surface_capacity) if isinstance(self.extractor, TrackedWirePointExtractor) else type(self.extractor)()
        self.last_visible_geometry = None

    @stere_infer__torch.no_grad()
    def step(self, left_rgb, right_rgb, proprio, *, left_timestamp_s, right_timestamp_s, frame_id, now_timestamp_s, reset_mask=None):
        if type(frame_id) is not int or frame_id <= self._frame or any((type(v) not in (int, float) or not math.isfinite(v) for v in (left_timestamp_s, right_timestamp_s, now_timestamp_s))):
            self._clear_temporal_state()
            raise ValueError('fresh frame ID and finite camera timestamps required')
        if abs(left_timestamp_s - right_timestamp_s) > 1e-09:
            self._clear_temporal_state()
            raise ValueError('left/right RGB must represent the same physical time')
        age = now_timestamp_s - left_timestamp_s
        if age < 0 or age > self.max_image_age_s:
            self._clear_temporal_state()
            raise ValueError('stale/future RGB pair; no truth fallback or silent timing change')
        dt = 1 / 60 if self._timestamp is None else left_timestamp_s - self._timestamp
        if dt <= 0:
            self._clear_temporal_state()
            raise ValueError('camera simulation/physical timestamp must increase')

        def rgb(value):
            if not isinstance(value, stere_infer__torch.Tensor) or value.dtype not in (stere_infer__torch.float32, stere_infer__torch.uint8):
                raise ValueError('RGB must be declared CHW uint8 RGB or float32 RGB; no BGR guessing')
            return value.to(self.device).float() / 255.0 if value.dtype == stere_infer__torch.uint8 else value.to(self.device)
        (left, right) = (rgb(left_rgb), rgb(right_rgb))
        p = proprio.to(self.device)
        if len(left) != self.num_envs:
            raise ValueError('configured batch size mismatch')
        if reset_mask is None:
            reset_mask = stere_infer__torch.zeros(self.num_envs, device=self.device, dtype=stere_infer__torch.bool)
        else:
            reset_mask = reset_mask.to(self.device)
        depth = self.depth.compute(left, right, self.calibration)
        extraction_kwargs = dict(reset_mask=reset_mask)
        if isinstance(self.extractor, TrackedWirePointExtractor):
            extraction_kwargs.update(timestamp=left_timestamp_s, point_confidence=depth['confidence'], wire_mask=depth.get('wire_mask'))
        points = self.extractor.extract(left, depth['depth_m'], depth['valid'], self.calibration.K_left, self.calibration.T_world_left, **extraction_kwargs)
        if 'timestamp_anomaly' in points:
            reset_mask = reset_mask | points['timestamp_anomaly']
        sensor = SensorBatch(left_rgb=left, right_rgb=right, depth_m=depth['depth_m'], depth_valid=depth['valid'], depth_confidence=depth['confidence'], points_world=points['points_world'], point_valid=points['point_valid'], proprio=p, point_confidence=points.get('confidence', points['point_valid'].float()), point_provenance=points.get('point_provenance'), dt=stere_infer__torch.full((self.num_envs,), dt, device=self.device, dtype=stere_infer__torch.float32), image_age=stere_infer__torch.full((self.num_envs,), age, device=self.device, dtype=stere_infer__torch.float32), image_valid=stere_infer__torch.ones(self.num_envs, 2, device=self.device, dtype=stere_infer__torch.bool))
        history = self.history.update(sensor, reset_mask=reset_mask)
        actions = self.actor.deterministic(history)
        if actions.shape != (self.num_envs, 24) or not stere_infer__torch.isfinite(actions).all() or (actions.abs() > 1).any():
            raise RuntimeError('Actor violated canonical normalized 24D action bounds')
        self.last_visible_geometry = dict(points_world_m=points['points_world'].detach().clone(), points_palms_m=world_to_palms(points['points_world'], p[:, 24:38].reshape(self.num_envs, 2, 7)).detach().clone(), point_valid=points['point_valid'].detach().clone(), frame_id=frame_id, simulation_timestamp=float(left_timestamp_s), source='RGB_stereo_and_current_measured_palm_FK', calibration_sha256=self.calibration.sha256)
        self.last_visible_geometry['wire_tracking'] = {key: value.detach().clone() if isinstance(value, stere_infer__torch.Tensor) else value for (key, value) in points.items() if key not in ('points_world', 'point_valid', 'xyz', 'valid')}
        (self._frame, self._timestamp) = (frame_id, float(left_timestamp_s))
        return dict(actions24=actions.detach().clone(), frame_id=frame_id, simulation_timestamp=float(left_timestamp_s), image_age_s=float(age), depth_valid_fraction=depth['valid'].float().flatten(1).mean(1), visible_wire_point_count=points['point_valid'].sum(1), actor_package_sha256=self.package_sha256, calibration_sha256=self.calibration.sha256, camera_config_sha256=self.camera_config_sha256, inference_source_sha256=stere_infer__file_sha256(__file__), processing_sha256=self.processing['processing_sha256'], action_owner='student', teacher_executed=False, mpc_executed=False, critic_executed=False, simulator_input_used=False, depth_source='computed_from_synchronized_left_right_RGB', native_task_qualified=False)

    def from_mapping(self, raw, *, reset_mask=None):
        if not isinstance(raw, dict) or set(raw) != RAW_FIELDS:
            raise ValueError('only RGB pair, measured proprioception and timing are accepted; privileged fields forbidden')
        return self.step(**raw, reset_mask=reset_mask)

def _processing(data, schema_only):
    from dh116.physics.vision import validate_processing_contract
    contracts = [row['identity'].get('observation_preprocessor') for row in data['rows']]
    if schema_only and all((value is None for value in contracts)):
        return None
    if any((value is None for value in contracts)):
        raise ValueError('every sample must bind its actual stereo processing contract')
    first = validate_processing_contract(contracts[0])
    for value in contracts:
        validate_processing_contract(value, expected=first)
    if first['calibration_sha256'] != data['contract']['calibration_sha256']:
        raise ValueError('recorded stereo calibration differs from dataset')
    return stude_train__deepcopy(first)

def run_student_main(argv=None):
    parser = stere_infer__argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor-package', type=stere_infer__Path, required=True)
    parser.add_argument('--actor-package-sha256')
    parser.add_argument('--rgb-frame', type=stere_infer__Path, required=True)
    parser.add_argument('--camera-config', type=stere_infer__Path, default=stere_infer__ROOT / 'configs/cameras/student_stereo.json')
    parser.add_argument('--output', type=stere_infer__Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--enable-experiment', action='store_true')
    args = parser.parse_args(argv)
    require_opt_in(args.enable_experiment)
    raw = stere_infer__torch.load(args.rgb_frame, map_location='cpu', weights_only=True)
    if not isinstance(raw, dict) or set(raw) != RAW_FIELDS:
        raise ValueError('versioned raw RGB/proprio frame required; depth/privileged tensors are not accepted')
    runner = StereoRGBStudentRunner(args.actor_package, args.camera_config, num_envs=len(raw['left_rgb']), device=args.device, expected_actor_package_sha256=args.actor_package_sha256)
    result = runner.from_mapping(raw)
    result = {k: v.detach().cpu().tolist() if isinstance(v, stere_infer__torch.Tensor) else v for (k, v) in result.items()}
    stere_infer__write_json_new(args.output, result)
    return 0

def _validation(actor, data, indices, device, *, state_loss=None):
    if not indices:
        return None
    from dh116.rl.student_data import masked_group_normalized_loss as masked_group_normalized_loss
    total = 0.0
    observed = 0
    with stude_train__torch.no_grad():
        for index in indices:
            batch = batch_dataset(data, [index], device=device)
            if state_loss is None:
                loss = masked_group_normalized_loss(actor.deterministic(batch['history']), batch['teacher_actions'], batch['action_valid'])
            else:
                terms = state_loss(actor.estimator(batch['history']), batch['targets'], batch['validity'], force_qualification=batch['force_qualification'])
                if not any((value > 0 for value in terms['coverage'].values())):
                    continue
                loss = terms['total']
            total += float(loss)
            observed += 1
    return total / observed if observed else None

def train_bc(data, network_config, bc_config, *, output, steps=100, batch_size=32, heldout_initial_sha256=(), seed=116, schema_only=False, device='cpu', stereo_qualification=None, camera_config=None):
    """Execute bounded offline BC on source-bound rows and export only the actor."""
    if type(schema_only) is not bool:
        raise ValueError('explicit schema_only boolean required')
    validate_sealed_stereo_dataset(data, allow_schema_only=schema_only)
    if type(steps) is not int or type(batch_size) is not int or min(steps, batch_size) < 1:
        raise ValueError('positive bounded update and minibatch budgets required')
    if not schema_only:
        if data['purpose'] != 'successful_bc' or not heldout_initial_sha256:
            raise ValueError('complete qualified Teacher success pool and independent physical holdout required')
        require_stereo(stereo_qualification, camera_config, data['contract']['calibration_sha256'])
    processing = _processing(data, schema_only)
    split = split_by_initial_condition(data, heldout_initial_condition_sha256=heldout_initial_sha256)
    network = stude_train__deepcopy(network_config)
    (height, width) = data['rows'][0]['history']['left_rgb'].shape[-2:]
    if any((tuple(row['history']['left_rgb'].shape[-2:]) != (height, width) for row in data['rows'])):
        raise ValueError('one recorded calibrated image resolution required')
    network['image_resolution_wh'] = [width, height]
    network = normalize_network_config(network)
    if any((row['history']['history_valid'].shape[1] != network['history'] for row in data['rows'])):
        raise ValueError('recorded and model history lengths differ')
    stude_train__torch.manual_seed(seed)
    generator = stude_train__torch.Generator(device='cpu').manual_seed(seed)
    actor = stude_train__make_stereo_actor(network).to(device)
    trainer = StereoBCTrainer(actor, bc_config, schema_only=schema_only)
    if trainer.config['lambda_action'] == 0:
        batch = batch_dataset(data, split['train_indices'], device=device)
        with stude_train__torch.no_grad():
            terms = trainer.state_loss(actor.estimator(batch['history']), batch['targets'], batch['validity'], force_qualification=batch['force_qualification'])
        if not any((value > 0 for value in terms['coverage'].values())):
            raise ValueError('state fit requires at least one source-valid observed label')
    output = stude_train__Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset_artifact = save_dataset(data, output / 'dataset.pt')
    provenance = dict(dataset_artifact={key: dataset_artifact[key] for key in ('path', 'sha256')}, teacher_checkpoint_sha256=data['source_sha256'], source_sha256={'student_training.py': stude_train__file_sha256(__file__)}, scope='schema_only_offline_diagnostic' if schema_only else 'source_bound_offline_BC')
    contract = make_contract('bc', dict(network=network, bc=stude_train__deepcopy(bc_config), split=split, schema_only=schema_only, preprocessor=processing, provenance=provenance, state_loss_version=trainer.state_loss.loss_version), calibration_sha256=data['contract']['calibration_sha256'])
    stude_train__write_json_new(output / 'bc_contract.json', contract)
    best = None
    best_state = None
    reports = []
    state_only = trainer.config['lambda_action'] == 0
    validation_metric = 'state_loss' if state_only else 'action_loss'
    with (output / 'updates.jsonl').open('x', encoding='utf-8') as log:
        for update in range(steps):
            pool = split['train_indices']
            chosen = stude_train__torch.randint(len(pool), (min(batch_size, len(pool)),), generator=generator).tolist()
            metrics = trainer.step(batch_dataset(data, [pool[i] for i in chosen], device=device))
            validation = _validation(actor, data, split['heldout_indices'], device, state_loss=trainer.state_loss if state_only else None)
            row = dict(update=update + 1, **{'validation_' + validation_metric: validation}, **metrics)
            log.write(stude_train__json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            reports.append(row)
            if validation is not None and (best is None or validation < best):
                best = validation
                best_state = stude_train__deepcopy(actor.state_dict())
    if best_state is not None:
        actor.load_state_dict(best_state, strict=True)
    package = save_actor_package(output / 'stereo_bc_actor.pt', actor, contract, network_config=network)
    report = dict(steps=steps, samples=len(data['rows']), heldout_samples=len(split['heldout_indices']), actor_package=package, schema_only=schema_only, native_task_qualified=False, selection='lowest_heldout_' + validation_metric if best_state is not None else 'last_update_diagnostic_no_holdout', uncertainty_calibrated=False, validation_scope='offline_error_only', updates=reports)
    stude_train__write_json_new(output / 'report.json', report)
    return report

def fit_state(data, network_config, *, output, steps=100, batch_size=32, **options):
    """Fit observed estimator labels with the existing source-masked state loss."""
    return train_bc(data, network_config, dict(lambda_action=0.0, lambda_state=1.0), output=output, steps=steps, batch_size=batch_size, **options)
