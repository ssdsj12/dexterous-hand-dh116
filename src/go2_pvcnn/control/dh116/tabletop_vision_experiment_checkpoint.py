"""Explicit offline batch experiments and immutable completed PPO snapshots.

No simulator, policy, optimizer, or CUDA context is constructed here. This is
deliberately separate from exact v3 resume, whose full-contract guard remains
unchanged. Inputs must be trusted full Torch checkpoints.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import torch

from .tabletop_surface_contract import semantic_sha256
from .tabletop_vision_action_contract import (
    LEGACY_VERSION, SQUASHED_VERSION, _ADDED, upgrade_action_contract,
    validate_squashed_checkpoint,
)
from .tabletop_vision_resume import _differences

PROJECTION_VERSION = 'dh116-visual-balanced-experiment-projection-v2'
BUNDLE_VERSION = 'dh116-visual-immutable-checkpoint-bundle-v1'
STYLES = ('pinch_thumb_index', 'tripod_thumb_index_middle')
_REFERENCES = ('privileged_critic_contract.success_bottom_m',
               'privileged_critic_contract.surface_force_sharing.lift_reference_bottom_m')
_ALLOWED = frozenset(('num_envs', 'camera_batch_size', 'groups[0].num_envs',
    'groups[1].num_envs', 'ppo.minibatches', 'memory_contract.logical_minibatches',
    'privileged_critic_contract.num_envs', 'privileged_critic_contract.minibatches',
    'privileged_critic_contract.surface_contract_sha256', *_REFERENCES,
    'physics_reward_provenance.privileged_critic_contract_sha256',
    'physics_reward_provenance.critic_authored_configuration_sha256.surface_contract_sha256'))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _same(left, right):
    if type(left) is not type(right): return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_same(a,b) for a,b in zip(left,right))
    return left == right


def _contract_json_view(contract):
    """Normalize the one observed historical tuple in a JSON manifest view.

    The original Torch contract and its typed state hash remain untouched.
    Do not coerce arbitrary tuple fields, numbers, booleans or physical values.
    """
    value = deepcopy(contract)
    critic = value.get('privileged_critic_contract', {}) if isinstance(value, dict) else {}
    phase = critic.get('phase_force_reward') if isinstance(critic, dict) else None
    if isinstance(phase, dict):
        from .tabletop_phase_force_reward import CONTRACT
        labels = phase.get('excluded_sources')
        if type(labels) is tuple and _same(labels, CONTRACT['excluded_sources']):
            phase['excluded_sources'] = list(labels)
    return value


def _state_sha256(value):
    digest = hashlib.sha256()
    def visit(item):
        digest.update((type(item).__name__+':').encode())
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for k in sorted(item, key=lambda k:(type(k).__name__,str(k))):visit(k);visit(item[k])
        elif isinstance(item,(tuple,list)):
            digest.update(str(len(item)).encode())
            for v in item:visit(v)
        else:digest.update(repr(item).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


def _unchanged_source(source, digest):
    if file_sha256(source) != digest:
        raise RuntimeError('source checkpoint changed during offline operation')


def _validate_completed(saved):
    validate_squashed_checkpoint(saved)
    if saved['iteration'] <= 0:
        raise ValueError('snapshot/projection requires a completed PPO update')
    steps = [float(v['step']) for v in saved['optimizer']['state'].values()]
    if not steps or min(steps) <= 0 or len(set(steps)) != 1:
        raise ValueError('completed PPO requires synchronized positive Adam steps')


def _reference(parent, path):
    for key in path.split('.'):
        if not isinstance(parent, dict) or key not in parent:
            raise ValueError('missing source per-environment reference: '+path)
        parent = parent[key]
    return parent


def _audit_batch_lists(value, n, path=''):
    if isinstance(value,dict):
        for k,v in value.items():_audit_batch_lists(v,n,path+'.'+k if path else k)
    elif isinstance(value,(list,tuple)):
        if len(value) == n and path not in _REFERENCES:
            raise ValueError('unknown per-environment contract field: '+path)
        for i,v in enumerate(value):_audit_batch_lists(v,n,path+'['+str(i)+']')


def project_experiment_checkpoint(input_path, output_path, *, num_envs=16,
                                  expected_source_sha256, expected_source_iteration=1439,
                                  target_contract=None):
    """Derive N=8/16/128 from a SHA-pinned, completed N=1024 v3 snapshot.

    The caller supplies the independently recorded source digest and expected
    update. The historical default remains 1439. A supplied target contract is
    an assertion, never a source of new physical defaults.
    Counts and physical reference rows retain the first N/2 of each style;
    global completed iterations/transitions and every model/Adam/RNG byte stay.
    """
    source, output = (Path(p).absolute() for p in (input_path,output_path))
    sidecar = output.with_suffix('.json')
    if output.suffix != '.pt' or source.resolve() in (output.resolve(),sidecar.resolve()):
        raise ValueError('projection requires a new .pt and .json distinct from source')
    for path in (output,sidecar):
        if path.exists() or path.is_symlink():raise FileExistsError(path)
    if type(num_envs) is not int or num_envs not in (8,16,128):
        raise ValueError('explicit balanced experiment requires 8, 16 or 128 environments')
    if type(expected_source_iteration) is not int or expected_source_iteration <= 0:
        raise ValueError('expected source iteration must be a positive integer')
    digest = file_sha256(source)
    if digest != expected_source_sha256:
        raise ValueError('source checkpoint SHA256 differs from independently frozen input')
    saved = torch.load(source,map_location='cpu',weights_only=False)
    _validate_completed(saved)
    old = saved['contract']; n = old['num_envs']
    if n != 1024 or saved['iteration'] != expected_source_iteration:
        raise ValueError('explicit experiment projection requires expected source iteration '
                         +str(expected_source_iteration)+' at num_envs=1024')
    if (old.get('camera_mode') != 'fixed_per_env' or old.get('camera_batch_size') != n
            or not _same(old.get('camera_resolution_wh'),[224,96])
            or old.get('groups') != [dict(label=s,num_envs=512) for s in STYLES]
            or old.get('action_contract',{}).get('size') != 24):
        raise ValueError('projection requires audited fixed camera / balanced 24D source')
    _audit_batch_lists(old,n)
    indices = list(range(num_envs//2))+list(range(n//2,n//2+num_envs//2))
    base = deepcopy(old); base['version'] = LEGACY_VERSION
    for key in _ADDED:base.pop(key)
    base['network'].pop('actor_output',None)
    base['num_envs'] = base['camera_batch_size'] = num_envs
    for g in base['groups']:g['num_envs'] = num_envs//2
    base['ppo']['minibatches'] = base['memory_contract']['logical_minibatches'] = min(8,num_envs)
    critic = base['privileged_critic_contract']
    if type(critic.get('num_envs')) is not int or critic['num_envs'] != n:
        raise ValueError('critic source num_envs must match actor batch')
    if type(critic.get('minibatches')) is not int:
        raise ValueError('critic source minibatches must be an integer')
    critic['num_envs'] = num_envs; critic['minibatches'] = min(8,num_envs)
    for path in _REFERENCES:
        values = _reference(old,path)
        if (type(values) is not list or len(values) != n
                or any(type(v) not in (int,float) for v in values)):
            raise ValueError('source reference requires complete numeric list: '+path)
        parts = path.split('.')[1:]; parent = critic
        for key in parts[:-1]:parent=parent[key]
        parent[parts[-1]] = [values[i] for i in indices]
    if not isinstance(critic.get('surface_contract_sha256'),str):
        raise ValueError('source requires existing surface semantic hash')
    critic['surface_contract_sha256'] = semantic_sha256(critic)
    expected = upgrade_action_contract(base)
    changes = list(_differences(old,expected))
    if any(c['path'] not in _ALLOWED or c['operation'] != 'changed'
           or type(c['from']) is not type(c['to']) for c in changes):
        raise ValueError('unapproved experiment contract change')
    if target_contract is not None and not _same(expected,target_contract):
        raise ValueError('target contract contains unapproved or missing projection changes')
    result = deepcopy(saved); result['contract'] = expected
    for key in ('episodes','successes'):
        result[key] = saved[key].index_select(0,torch.tensor(indices,dtype=torch.long))
    result['experiment_projection'] = dict(version=PROJECTION_VERSION,
        status='explicit_offline_experiment_not_exact_source_resume',
        source_sha256=digest,source_version=saved['version'],source_iteration=saved['iteration'],
        expected_source_iteration=expected_source_iteration,
        source_timesteps=saved['timesteps'],source_num_envs=n,target_num_envs=num_envs,
        retained_indices=indices,changes=changes,
        original_action_derivation=deepcopy(saved.get('action_derivation')),
        global_iterations_transitions_preserved=True,
        rng_coverage='preserved_Torch_CPU_and_saved_CUDA;Python_Numpy_unavailable_in_source')
    _validate_completed(result)
    for key in saved:
        if key not in ('contract','episodes','successes') and not _same(saved[key],result[key]):
            raise RuntimeError('unexpected checkpoint state change: '+key)
    report = dict(version=PROJECTION_VERSION,status='derived_offline',source_path=str(source),
        source_sha256=digest,checkpoint_path=str(output),
        experiment_projection=deepcopy(result['experiment_projection']))
    _json_bytes(report); _unchanged_source(source,digest)
    output.parent.mkdir(parents=True,exist_ok=True); created=[]
    try:
        with output.open('xb') as pt:
            created.append(output)
            with sidecar.open('x',encoding='utf-8') as meta:
                created.append(sidecar); torch.save(result,pt);pt.flush();os.fsync(pt.fileno())
                _unchanged_source(source,digest)
                report['checkpoint_sha256']=file_sha256(output)
                meta.write(_json_bytes(report).decode());meta.flush();os.fsync(meta.fileno())
    except BaseException:
        for path in reversed(created):path.unlink(missing_ok=True)
        raise
    return report


def _copy_bytes(source,destination):
    # copyfileobj rather than copy2: NTFS and network shares need no chmod/stat.
    with Path(source).open('rb') as src:
        for block in iter(lambda:src.read(1024*1024),b''):destination.write(block)
        destination.flush();os.fsync(destination.fileno())


def _declared_hashes(value,path=''):
    result={}
    if isinstance(value,dict):
        for key,item in value.items():
            name=path+'.'+str(key) if path else str(key)
            if 'sha256' in str(key) and isinstance(item,str):result[name]=item
            else:result.update(_declared_hashes(item,name))
    return result


def retain_checkpoint_bundle(source_path, output_dir, *, provenance=None):
    """Retain full rolling checkpoint bytes after a complete PPO update.

    Directory creation claims an iteration once. Same source digest and
    provenance are idempotent; changed bytes at that iteration are a conflict.
    Never evict prior iterations. Return a manifest whose own hash is external.
    """
    source = Path(source_path).absolute(); root = Path(output_dir).absolute()
    digest = file_sha256(source)
    saved = torch.load(source,map_location='cpu',weights_only=False)
    _validate_completed(saved); _unchanged_source(source,digest)
    provenance = deepcopy(provenance or {}); _json_bytes(provenance)
    bundle = root/('iteration_'+format(saved['iteration'],'08d'))
    pt, meta = bundle/'checkpoint.pt',bundle/'manifest.json'
    if source.resolve() in (pt.resolve(),meta.resolve()):
        raise ValueError('snapshot source cannot be its own destination')
    states = {k:_state_sha256(saved[k]) for k in ('policy','optimizer','torch_rng','cuda_rng')}
    states['counts'] = _state_sha256({k:v for k,v in saved.items() if k in
        ('iteration','timesteps','episodes','successes','recovered','reached_counts','transition_counts')})
    states['contract'] = _state_sha256(saved['contract'])
    states['learning_rate'] = _state_sha256(saved['learning_rate'])
    configuration=_declared_hashes(saved['contract']);configuration['contract_exact_sha256']=states['contract']
    declared=_declared_hashes(saved,'checkpoint')|_declared_hashes(provenance,'provenance')
    datasets={k:v for k,v in declared.items() if 'dataset' in k.lower()}
    manifest = dict(version=BUNDLE_VERSION,checkpoint_path=str(pt),manifest_path=str(meta),
        source_path=str(source),checkpoint_sha256=digest,checkpoint_version=saved['version'],
        iteration=saved['iteration'],timesteps=saved['timesteps'],learning_rate=saved['learning_rate'],
        state_sha256=states,contract=saved['contract'],provenance=provenance,
        configuration_sha256=configuration,dataset_sha256=datasets,
        dataset_provenance_status='declared_hashes_preserved' if datasets else 'no_dataset_hash_declared_in_source_or_provenance',
        complete_ppo_update=True)
    payload=_json_bytes(manifest)
    root.mkdir(parents=True,exist_ok=True)
    try:bundle.mkdir()
    except FileExistsError:
        if (bundle.is_symlink() or not pt.is_file() or pt.is_symlink() or not meta.is_file()
                or meta.is_symlink() or file_sha256(pt)!=digest):
            raise FileExistsError('immutable iteration bundle conflict: '+str(bundle)) from None
        try:
            existing=json.loads(meta.read_text(encoding='utf-8'))
        except (OSError,ValueError,UnicodeError):
            raise FileExistsError('immutable iteration bundle conflict: '+str(bundle)) from None
        # Identical bytes may be presented through another source filename.
        # Retain the first provenance path without rewriting the immutable file.
        comparable=dict(manifest,source_path=existing.get('source_path'),
            contract=_contract_json_view(manifest['contract'])) if isinstance(existing,dict) else None
        if (not isinstance(existing,dict) or not isinstance(existing.get('source_path'),str)
                or not existing['source_path'] or not _same(comparable,existing)):
            raise FileExistsError('immutable iteration bundle conflict: '+str(bundle)) from None
        _unchanged_source(source,digest)
        return existing
    created=[]
    try:
        with pt.open('xb') as stream:
            created.append(pt)
            _copy_bytes(source,stream)
        _unchanged_source(source,digest)
        if file_sha256(pt)!=digest:raise RuntimeError('snapshot bytes differ from stable source')
        with meta.open('xb') as stream:
            created.append(meta)
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
    except BaseException:
        # Remove only files created by this owner, never a competing sidecar.
        for path in reversed(created):path.unlink(missing_ok=True)
        try:bundle.rmdir()
        except OSError:pass
        raise
    return manifest
