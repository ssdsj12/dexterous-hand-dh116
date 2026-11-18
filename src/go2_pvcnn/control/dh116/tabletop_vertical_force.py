"""Signed hand-to-object vertical force telemetry, independent of task rewards."""
import json
from pathlib import Path
import torch

CONTRACT=dict(version='normal_vertical_force_v3',units='N',positive='world_up_z',
    source='object_filtered_normal_on_hand_negated',friction_excluded=True,
    legacy_total_fields='signed_normal_z_only',friction_field='separate_diagnostic_only',
    excludes='table_and_other_hand_contacts',sampling='live_physics_substeps',
    window='completed_rollout_not_training_lifetime',
    use='telemetry_only_not_reward_or_lift_gate')

CLOSE_EVENT_CONTRACT=dict(version='individual_close_completion_force_v1',
    sampling='valid_close_to_lift_physics_substep',aggregation='none',
    source='same_normal_vertical_force_tensor_as_task_consume',friction_excluded=True,
    tensorboard_step='completion_event_index',missing_event='no_force_point',
    main_namespace='force_N/close完成',rollout_namespace='rollout_force_N',
    journal='close_completion_events.jsonl')


class DiagnosticTensorboardWriter:
    """Disable a failed optional sink; programming and data errors still raise."""
    def __init__(self, writer):
        self.writer = writer
        self.error = None
        self.event_index = None
        self.first_affected_event_index = None

    def _call(self, operation, *args, **kwargs):
        if self.error is not None and operation != 'close':
            return
        try:
            return getattr(self.writer, operation)(*args, **kwargs)
        except OSError as error:
            if self.error is None:
                self.error = f'{operation}: {type(error).__name__}: {error}'
                self.first_affected_event_index = self.event_index

    def add_scalar(self, *args, **kwargs):
        return self._call('add_scalar', *args, **kwargs)

    def add_text(self, *args, **kwargs):
        return self._call('add_text', *args, **kwargs)

    def flush(self):
        return self._call('flush')

    def close(self):
        return self._call('close')


class CloseCompletionForceEvents:
    """Persist each admitted close sample before a reset or later physics fault.

    Scalar steps are event indices, not PPO iterations. The JSONL maps each
    event to its environment, episode and current physical substep. No mean,
    temporal filter, friction or weight subtraction is applied to the gate value.
    """
    def __init__(self,writer,path,event_index=0):
        self.writer=DiagnosticTensorboardWriter(writer)
        self.path=Path(path)
        self.event_index=int(event_index)
        self.observed_events=0
        self.iteration=0
        self.control_step=0
        self.journal_error=None
        self.journal_first_affected_event_index=None

    def diagnostics(self):
        errors={};first_affected={}
        if self.journal_error is not None:
            errors['journal']=self.journal_error
            first_affected['journal']=self.journal_first_affected_event_index
        if self.writer.error is not None:
            errors['tensorboard']=self.writer.error
            first_affected['tensorboard']=self.writer.first_affected_event_index
        return dict(status='degraded' if errors else 'ok',errors=errors,
            first_affected_event_index=first_affected,observed_events=self.observed_events,
            event_index=self.event_index,
            failure_scope='affected_sink_disabled;failing_batch_may_be_partial')

    def record(self,components,gate_force_n,episodes,elapsed,entry,substep):
        ids=entry.nonzero().flatten()
        if not ids.numel():return
        # Read only selected endpoints; values become independent of mutable
        # simulation tensors before any world/task reset.
        values=torch.cat((gate_force_n[ids,None].double(),components[ids,:,0].double(),
                          elapsed[ids,None].double()),dim=1).detach().cpu().tolist()
        identities=torch.stack((ids,episodes[ids]+1),dim=1).cpu().tolist()
        rows=[]
        for (env_index,episode_index),(force,left,right,episode_time) in zip(identities,values):
            self.event_index+=1
            self.observed_events+=1
            rows.append(dict(event_index=self.event_index,env_index=env_index,
                episode_index=episode_index,episode_time_s=episode_time,
                physics_substep=int(substep),ppo_iteration=self.iteration,
                control_step=self.control_step,force_z_n=force,
                hand_normal_z_n=[left,right],friction_included=False,entered_lift=True))
        # Only optional sink I/O is recoverable. Invalid endpoint values or
        # programming errors still raise; existing journal rows are never edited.
        serialized=[json.dumps(row,allow_nan=False)+'\n' for row in rows]
        if self.journal_error is None:
            try:
                with self.path.open('a') as stream:
                    stream.writelines(serialized)
            except OSError as error:
                self.journal_error=f'{type(error).__name__}: {error}'
                self.journal_first_affected_event_index=rows[0]['event_index']
        self.writer.event_index=rows[0]['event_index']
        try:
            for row in rows:
                index=row['event_index'];force=row['force_z_n']
                for label,value in (
                    ('竖直合力',force),('向上力',max(force,0.)),('向下力',max(-force,0.)),
                    ('左手法向Z',row['hand_normal_z_n'][0]),('右手法向Z',row['hand_normal_z_n'][1])):
                    self.writer.add_scalar('force_N/close完成_'+label+'_瞬时',value,index)
                for key in ('env_index','episode_index','episode_time_s','ppo_iteration','control_step','physics_substep'):
                    self.writer.add_scalar('close_completion/'+key,row[key],index)
            self.writer.flush()
        finally:
            if self.writer.error is not None and self.writer.first_affected_event_index is None:
                self.writer.first_affected_event_index=rows[0]['event_index']
            self.writer.event_index=None


def object_vertical_components(normal_on_hands,friction_on_hands):
    """[env,hand,link,xyz] -> [env,hand,(normal_z,friction_z)]."""
    if normal_on_hands.shape!=friction_on_hands.shape or normal_on_hands.ndim!=4:
        raise ValueError('matching [environment,hand,link,xyz] forces required')
    return -torch.stack((normal_on_hands[...,2].sum(2),friction_on_hands[...,2].sum(2)),-1)


class VerticalForceStats:
    def __init__(self,device):
        self.sums=torch.zeros((2,2),device=device,dtype=torch.float64)
        self.totals=torch.zeros(3,device=device,dtype=torch.float64)
        self.peak=torch.zeros((),device=device,dtype=torch.float64)
        self.direction_sums=torch.zeros(2,device=device,dtype=torch.float64)
        self.direction_peaks=torch.zeros(2,device=device,dtype=torch.float64)
        self.clear()

    def clear(self):
        self.sums.zero_();self.totals.zero_();self.peak.fill_(-torch.inf)
        self.direction_sums.zero_();self.direction_peaks.zero_()

    def add(self,components,weight,live):
        finite=torch.isfinite(components).flatten(1).all(1)&torch.isfinite(weight)&(weight>0)
        if bool((live&~finite).any()):raise RuntimeError('nonfinite or invalid vertical contact force telemetry')
        values=torch.where(live[:,None,None],components,0.).double()
        # The second component remains available for separate diagnostics;
        # all directional, peak and weight comparisons use normal Z only.
        total=values[:,:,0].sum(1)
        directions=torch.stack((total.clamp_min(0.),(-total).clamp_min(0.)),dim=1)
        self.direction_sums.add_(directions.sum(0))
        self.direction_peaks.copy_(torch.maximum(self.direction_peaks,directions.amax(0)))
        self.sums.add_(values.sum(0))
        self.totals.add_(torch.stack((live.sum(),((total>weight)&live).sum(),
                                     torch.where(live,weight,0.).sum())))
        self.peak.copy_(torch.maximum(self.peak,torch.where(live,total,-torch.inf).max()))

    def summary(self):
        count,above,weight=self.totals.cpu().tolist()
        values=(self.sums/max(count,1)).cpu()
        total=float(values[:,0].sum())
        upward,downward=(self.direction_sums/max(count,1)).cpu().tolist()
        up_peak,down_peak=self.direction_peaks.cpu().tolist()
        return dict(CONTRACT,sample_count=int(count),
            upward_mean_n=upward,downward_mean_n=downward,
            upward_peak_n=up_peak if count else None,downward_peak_n=down_peak if count else None,
            hand_normal_mean_n=values[:,0].tolist(),hand_friction_mean_n=values[:,1].tolist(),
            hand_total_mean_n=values[:,0].tolist(),hands_total_mean_n=total,
            weight_mean_n=weight/max(count,1),
            force_minus_weight_mean_n=total-weight/max(count,1),
            above_weight_sample_fraction=above/max(count,1),
            hands_total_peak_n=float(self.peak) if count else None)


class LiftEntryForceStats(VerticalForceStats):
    """One force sample per valid close-to-lift transition, not a time mean."""
    def summary(self):
        row=super().summary()
        row.update(version='close_to_lift_force_v1',sampling='close_to_lift_transition_only',
                   use='telemetry_of_same_sample_as_close_downward_event',
                   aggregation='mean_over_transition_events_in_completed_rollout')
        if not row['sample_count']:
            for key in ('upward_mean_n','downward_mean_n','hand_normal_mean_n','hand_friction_mean_n',
                        'hand_total_mean_n','hands_total_mean_n','weight_mean_n',
                        'force_minus_weight_mean_n','above_weight_sample_fraction'):
                row[key]=None
        return row


def write_lift_entry_force_scalars(writer,row,iteration):
    writer.add_scalar('lift_entry_force/事件数',row['sample_count'],iteration)
    if not row['sample_count']:return  # No transition is not a zero-force measurement.
    labels={'upward_mean_n':'向上力_平均','downward_mean_n':'向下力_平均',
            'upward_peak_n':'向上力_峰值','downward_peak_n':'向下力_峰值',
            'hands_total_mean_n':'竖直合力_向上为正','weight_mean_n':'物体重量_参考'}
    for key,label in labels.items():writer.add_scalar('lift_entry_event_summary_N/'+label,row[key],iteration)
    for hi,side in enumerate(('左手','右手')):
        writer.add_scalar('lift_entry_event_summary_N/'+side+'_竖直合力',row['hand_total_mean_n'][hi],iteration)


REWARD_CONTRACT = dict(version='signed_normal_vertical_contact_v2',
    source='object_filtered_normal_on_hand_negated',friction_excluded=True,positive='world_up_z',
    reduction='signed_sum_both_hands_before_positive_negative_split',
    upward_window_start='close_start',upward_window_s=1.,
    upward_full_scale_n=10.,upward_points_per_s=5.,upward_episode_max_points=5.,
    replaces_unsigned_contact=True,downward_points_per_n_s=.5,
    downward_scope='all_live_physics_substeps_including_terminal',downward_cap=None,
    gravity_and_table_excluded=True,observation_unchanged=True,
    grasp_ready_bonus='once_after_lift_admission_while_ready_and_positive_vertical_force')


def vertical_reward_contract(mode='window',start='episode_start',*,skip_hold=False):
    if skip_hold and start!='episode_start':
        return dict(vertical_reward_contract(mode,start),
            force_reward_activation=('upward_immediate_downward_first_substep_after_close_to_lift'
                if start=='downward_after_close' else 'first_substep_after_close_to_lift'))
    if start=='downward_after_close':
        return dict(vertical_reward_contract(mode,'after_close'),version='signed_normal_close_upward_v5',
            force_reward_start='upward_from_episode_start_downward_after_first_close_completion',
            force_reward_activation='upward_immediate_downward_first_substep_after_hold_entry',
            force_reward_latch='downward_until_selected_reset_no_reexemption_on_grip_loss',
            upward_scope='all_finite_live_nonfailure_substeps',
            observation_flag_meaning='downward_penalty_enabled')
    if start=='after_close':
        if mode!='continuous':raise ValueError('after_close requires continuous force reward')
        return dict(vertical_reward_contract(mode),version='signed_normal_after_close_v4',
            force_reward_start='after_first_close_completion',
            force_reward_activation='first_substep_whose_previous_phase_is_hold_or_lift',
            force_reward_latch='until_selected_episode_reset_no_reexemption_on_grip_loss',
            upward_scope='scoring_started_and_finite_live_nonfailure_substeps',
            downward_scope='scoring_started_and_finite_live_substeps_including_terminal',
            observation_unchanged=False,observation_append='force_scoring_started')
    if start!='episode_start':raise ValueError('unsupported force reward start')
    if mode=='window':return dict(REWARD_CONTRACT)
    if mode!='continuous':raise ValueError('unsupported vertical force reward mode')
    return dict(REWARD_CONTRACT,version='signed_normal_continuous_v3',
        upward_window_start=None,upward_window_s=None,upward_episode_max_points=None,
        upward_scope='all_finite_live_nonfailure_substeps',
        legacy_window_clock='observation_only_no_reward_gate')


def write_directional_force_scalars(writer,vertical,iteration):
    """Normal-only N curves; upward/downward magnitudes split before averaging."""
    labels={'upward_mean_n':'向上力_平均','downward_mean_n':'向下力_平均',
            'upward_peak_n':'向上力_瞬时峰值','downward_peak_n':'向下力_瞬时峰值',
            'hands_total_mean_n':'竖直合力_向上为正','weight_mean_n':'物体重量_参考'}
    for key,label in labels.items():
        value=vertical[key]
        if value is not None:
            writer.add_scalar('rollout_force_N/'+label,value,iteration)
            if key in ('upward_mean_n','downward_mean_n','upward_peak_n','downward_peak_n'):
                writer.add_scalar('vertical_force/'+key,value,iteration)
