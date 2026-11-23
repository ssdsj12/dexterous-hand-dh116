"""Bounded orchestration over the original CPU PPO and a source-bound collector.

C06 must supply a separately frozen native worker, lease and resource reservation.
The CLI reports that missing integration; a boolean never launches Isaac/App.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from .tabletop_native_teacher_checkpoint import stop_update_target, _trainer

VERSION="dh116-native-teacher-training-cpu-v1"


class TrainingStopped(RuntimeError):pass


def _finite_numbers(value):
    if isinstance(value,float) and not math.isfinite(value):raise TrainingStopped('nonfinite_optimizer_metrics')
    if isinstance(value,dict):
        for x in value.values():_finite_numbers(x)
    elif isinstance(value,(list,tuple)):
        for x in value:_finite_numbers(x)


def run_updates(collector, *, additional_updates=1, factories=None):
    """CPU optimizer entry for an already constructed, source-bound C04 session.

    Does not create an App, issue a lease, expand budgets, or retry a rejected
    optimizer update. Terminal episodes require explicit original factories.
    """
    from .tabletop_native_teacher_collector import NativeTeacherCollector
    if type(collector) is not NativeTeacherCollector:raise ValueError('original source-bound native collector required')
    tr=collector.trainer;_trainer(tr)
    if collector.quarantined or getattr(tr,'_native_training_failed',False):raise TrainingStopped('failed_session_last_clean_checkpoint_only')
    loaded=tr.counters['updates'];target=stop_update_target(loaded,additional_updates=additional_updates)
    accepted=getattr(tr,'_native_accepted_minibatches',0)
    if loaded and not accepted:raise TrainingStopped('loaded_training_progress_missing')
    delta=0.;metrics=None
    try:
        while tr.counters['updates']<target:
            tr._native_completed_update=None
            while not tr.rollout.full:
                if collector.episode_done:
                    # Session budget survives Reset; the old episode boundary does not.
                    runtime=getattr(getattr(collector.owner,'contract',None),'runtime',None)
                    if runtime is not None and runtime.remaining_steps()<8:
                        raise TrainingStopped('budget_exhausted_before_complete_window')
                    if factories is None:raise TrainingStopped('fresh_verified_reset_factories_required')
                    collector=collector.restart_episode(factories)
                if collector.collect() is None:raise TrainingStopped('budget_exhausted_before_complete_window')
            before=[p.detach().clone() for p in tr.model.parameters()]
            updates=tr.counters['updates'];version=tr.policy_version
            # Original algorithm owns frozen likelihood, GAE, clipping and KL rollback.
            metrics=tr.update()
            count=metrics['minibatch_updates']
            if type(count) is not int or count<=0:raise TrainingStopped('optimizer_update_rejected')
            if tr.counters['updates']!=updates+1 or tr.policy_version!=version+1:
                raise TrainingStopped('original_optimizer_counters_did_not_advance')
            _finite_numbers(metrics)
            change=sum(float((p.detach().double()-old.double()).square().sum()) for p,old in zip(tr.model.parameters(),before))
            if not math.isfinite(change) or change<=0:raise TrainingStopped('optimizer_parameter_change_not_finite_nonzero')
            if any(not bool(torch.isfinite(p).all()) for p in tr.model.parameters()):raise TrainingStopped('nonfinite_model_parameters')
            collector.accept_updated_policy()
            accepted+=count;delta+=change
            tr._native_accepted_minibatches=accepted;tr._native_completed_update=tr.counters['updates']
            collector.journal.emit('optimizer_update_accepted',dict(updates=tr.counters['updates'],
                accepted_minibatch_updates=accepted,parameter_delta_l2=math.sqrt(change),metrics=metrics))
    except BaseException as error:
        tr._native_training_failed=True
        try:collector.journal.emit('training_stopped',dict(reason=str(error),updates=tr.counters['updates'],
            target_updates=target,recovery='last_clean_checkpoint_only'))
        except BaseException:pass
        raise
    return dict(scope=collector.sources.scope,loaded_updates=loaded,target_updates=target,
        updates=tr.counters['updates'],accepted_minibatch_updates=accepted,
        parameter_delta_l2=math.sqrt(delta),metrics=metrics,collector=collector) if factories is not None else dict(
        scope=collector.sources.scope,loaded_updates=loaded,target_updates=target,updates=tr.counters['updates'],
        accepted_minibatch_updates=accepted,parameter_delta_l2=math.sqrt(delta),metrics=metrics)


def candidate_training_config():
    from .tabletop_native_teacher_preflight import candidate_config
    return dict(schema=VERSION,preflight=candidate_config(),model_config=None,episode_cap_s=None,rollout_steps=32)


def preflight_training(config):
    from .tabletop_native_teacher_preflight import static_preflight
    baseline=candidate_training_config()
    if not isinstance(config,dict) or set(config)!=set(baseline) or config['schema']!=VERSION:
        raise ValueError('independent training configuration schema required')
    gate=static_preflight(config['preflight'])
    reasons=list(gate['blocked_reasons'])
    if config['model_config'] is None:reasons.append('independent_native_context_model_config_missing')
    else:
        try:
            from .tabletop_native_teacher_profile import BackendProfile
            from .tabletop_native_teacher_model import model_contract
            identity=model_contract(config['model_config'],BackendProfile.load(),config['preflight']['profile'])
            if identity['grasp']!=config['preflight']['style']:raise ValueError('model/source grasp differs')
        except (ValueError,KeyError,TypeError,OSError) as error:reasons.append('model_contract_invalid: '+str(error))
    if config['preflight']['style'] not in ('pinch','tripod','power'):reasons.append(config['preflight']['style']+'_distance_reader_implementation_pending')
    if (config['preflight']['profile']!='approach_wiring_smoke_v1'
            and (config['preflight']['profile']!='approach_contact_prelift_v1'
                 or config['preflight']['style']!='tripod')):
        reasons.append('full_shared_contact_slip_hold_source_adapter_missing')
    if (type(config['episode_cap_s']) not in (int,float) or not math.isfinite(config['episode_cap_s'])
            or config['episode_cap_s']<=0):reasons.append('explicit_episode_cap_s_missing')
    if type(config['rollout_steps']) is not int or config['rollout_steps']!=32:reasons.append('approved_rollout32_required')
    # No registered worker provider/lease exists in C05. C06 owns its frozen
    # source, fresh budget and reservation; JSON or --execute-native is no lease.
    for field,reason in (('budget','new_native_budget_missing'),
            ('authorization','new_native_authorization_missing'),('resource_lock','kernel_resource_reservation_missing')):
        if config['preflight'][field] is None:reasons.append(reason)
    reasons.append('c06_source_bound_worker_lease_missing')
    return dict(version=VERSION,static=gate,blocked_reasons=reasons,native_execution_ready=False,
        policy_constructed=False,optimizer_constructed=False,native_app_constructed=False,
        worker_provider='C06_pending',native_ready=False,execution_requirements=dict(
            budget=config['preflight']['budget'],authorization=config['preflight']['authorization'],
            resource_reservation=config['preflight']['resource_lock'],fresh_worker_lease_verified=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--additional-updates',type=int,default=1);parser.add_argument('--checkpoint')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute-native',action='store_true')
    args=parser.parse_args(argv);stop_update_target(0,additional_updates=args.additional_updates)
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    report=preflight_training(config)
    report.update(requested_additional_updates=args.additional_updates,checkpoint_requested=args.checkpoint,
        execute_native_requested=args.execute_native,checkpoint_loaded=False)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with (output/'preflight.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 2
