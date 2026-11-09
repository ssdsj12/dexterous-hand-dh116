"""Explicit first-cohort metadata history, separate from strict F0 v1.

Fresh standalone F0 snapshots keep their original clocks, raw and schema. This
new history owner retires each first episode once, never re-enables that slot,
and requires an explicit generation-readback receipt for a retired slot's later
reset generation. An inactive slot is merely a carrier in full-view metadata;
it receives no first-cohort pair credit. No contact/force/planning qualification
is issued. There is no SDK, capture hook, reset, target or controller operation.
F0 v1's own previous-terminal refusal remains unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType

from .tabletop_bimanual_mpc_state import SampleClock
from .tabletop_bimanual_mpc_filtered_contact_provenance import (
    FilteredPairSnapshot, _plain, resolve_filtered_sensor_pairs,
)

VERSION='dh116-filtered-contact-first-cohort-v2'
_QUALIFICATION=MappingProxyType(dict(planning_eligible=False,contact_values_qualified=False,
    native_attested=False,support_qualified=False,force_sign_frame_qualified=False,
    native_success_qualification_issued=False,new_expert_qualification_issued=False))


def _require(ok,message):
    if not ok:raise ValueError(message)


def _same(a,b):
    return json.dumps(_plain(a),sort_keys=True,allow_nan=False)==json.dumps(_plain(b),sort_keys=True,allow_nan=False)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _mask(value,n,name):
    _require(isinstance(value,(list,tuple)) and len(value)==n and all(type(v) is bool for v in value),name+' requires one actual boolean per env')
    return tuple(value)


def _clock_step(before,after):
    _require(before.run_uuid==after.run_uuid and before.env_id==after.env_id,'clock run/environment changed')
    _require(after.world_step-before.world_step==after.sdk_step-before.sdk_step==1
        and 0<=after.policy_step-before.policy_step<=1
        and math.isclose(after.world_time_s-before.world_time_s,1/480,rel_tol=0,abs_tol=1e-7)
        and math.isclose(after.sdk_time_s-before.sdk_time_s,1/480,rel_tol=0,abs_tol=1e-7),
        'clock must advance by one actual 480Hz frame')


@dataclass(frozen=True,slots=True)
class RetiredGenerationReceipt:
    before: SampleClock
    after: SampleClock
    source: str
    eligibility_granted: bool = False

    def __post_init__(self):
        _require(type(self.before) is SampleClock and type(self.after) is SampleClock,'actual generation clocks required')
        _clock_step(self.before,self.after)
        _require(self.after.reset_generation==self.before.reset_generation+1
            and self.after.episode_uid!=self.before.episode_uid
            and self.after.policy_step==self.before.policy_step+1,
            'generation receipt requires a new diagnostic generation at a policy boundary')
        _require(isinstance(self.source,str) and bool(self.source.strip()),'actual readback source required')
        _require(self.eligibility_granted is False,'a retired generation receipt cannot grant eligibility')


@dataclass(frozen=True,slots=True)
class ContactEpochState:
    metadata: FilteredPairSnapshot
    initial_clocks: tuple[SampleClock,...]
    retired: tuple[bool,...]
    first_terminals: tuple[SampleClock | None,...]
    actual_live: tuple[bool,...]
    actual_terminal: tuple[bool,...]
    cohort_live: tuple[bool,...]
    active_pairs: tuple
    active_pairs_per_env: tuple[int,...]
    generation_receipts: tuple[RetiredGenerationReceipt,...]
    liveness_source: str
    previous_receipt_sha256: str | None
    receipt_sha256: str
    version: str = VERSION

    def __post_init__(self):
        for name in ('initial_clocks','retired','first_terminals','actual_live','actual_terminal',
                     'cohort_live','active_pairs','active_pairs_per_env','generation_receipts'):
            object.__setattr__(self,name,tuple(getattr(self,name)))

    @property
    def qualification(self):return _QUALIFICATION

    def to_dict(self):
        # Full metadata is archived once in its existing F0 field, never copied
        # N times or rewritten to replace first-terminal clocks.
        return dict(version=self.version,metadata_binding_fingerprint=self.metadata.binding_fingerprint,
            current_clocks=[asdict(c) for c in self.metadata.clocks],
            initial_clocks=[asdict(c) for c in self.initial_clocks],retired=list(self.retired),
            first_terminals=[None if c is None else asdict(c) for c in self.first_terminals],
            actual_live=list(self.actual_live),actual_terminal=list(self.actual_terminal),cohort_live=list(self.cohort_live),
            active_pairs_per_env=list(self.active_pairs_per_env),
            generation_receipts=[asdict(r) for r in self.generation_receipts],liveness_source=self.liveness_source,
            previous_receipt_sha256=self.previous_receipt_sha256,receipt_sha256=self.receipt_sha256,
            qualification=dict(self.qualification),
            scope='first-cohort archival metadata eligibility only; no contact, force, Native or planning authority')


def _metadata(value):
    _require(type(value) is FilteredPairSnapshot and value.metadata_binding_valid is True
        and value.errors==() and value.version=='dh116-filtered-pair-metadata-f0-v1',
        'valid fresh standalone F0 metadata required')
    raw=value.to_dict()['raw']
    replay=resolve_filtered_sensor_pairs(raw.get('sensor_sources'),wire_body_paths=raw.get('wire_body_paths'),
        clocks_before=[SampleClock(**c) for c in raw.get('clocks_before',[])],
        clocks_after=[SampleClock(**c) for c in raw.get('clocks_after',[])],
        source_identity=raw.get('source_identity'))
    expected=replay.to_dict();expected['raw']=raw
    _require(replay.metadata_binding_valid is True and _same(value.to_dict(),expected),
        'fresh F0 DTO/raw binding differs; old metadata cannot be retimestamped')
    n=len(value.clocks);by_env={c.env_id:c for c in value.clocks}
    _require(set(by_env)==set(range(n)),'explicit contiguous env IDs required for liveness masks')
    return tuple(by_env[i] for i in range(n))


def _history_payload(state):
    return dict(version=state.version,binding=state.metadata.binding_fingerprint,
        metadata_sha256=_digest(state.metadata.to_dict()),
        clocks=[asdict(c) for c in state.metadata.clocks],initial=[asdict(c) for c in state.initial_clocks],
        retired=state.retired,first_terminals=[None if c is None else asdict(c) for c in state.first_terminals],
        actual_live=state.actual_live,actual_terminal=state.actual_terminal,cohort_live=state.cohort_live,
        generation_receipts=[asdict(r) for r in state.generation_receipts],source=state.liveness_source,
        previous_receipt_sha256=state.previous_receipt_sha256)


def _validate_previous(state,n):
    _require(type(state) is ContactEpochState and state.version==VERSION,'typed previous contact epoch required')
    _metadata(state.metadata)
    _require(state.receipt_sha256==_digest(_history_payload(state)),'previous history receipt changed')
    _require(len(state.initial_clocks)==len(state.first_terminals)==n,'first cohort cannot change size')
    for name in ('retired','actual_live','actual_terminal','cohort_live'):
        _mask(getattr(state,name),n,'previous '+name)
    for i,(retired,terminal) in enumerate(zip(state.retired,state.first_terminals)):
        _require((terminal is not None)==retired,'previous history retirement differs from pinned terminal')
        if terminal is not None:
            _require(type(terminal) is SampleClock and terminal.terminal and not terminal.reset
                and terminal.env_id==i and (terminal.run_uuid,terminal.episode_uid,terminal.reset_generation)==
                    (state.initial_clocks[i].run_uuid,state.initial_clocks[i].episode_uid,state.initial_clocks[i].reset_generation),
                'previous history terminal must be from its original first episode')
    pairs=tuple(p for p in state.metadata.pairs if state.cohort_live[p.clock.env_id])
    _require(state.active_pairs==pairs and state.active_pairs_per_env==tuple(34 if v else 0 for v in state.cohort_live),
        'previous history active pair credit changed')


def advance_contact_epoch(metadata,*,previous=None,actual_live,actual_terminal,cohort_live,
                          retired_generation_receipts=(),liveness_source):
    """Advance a distinct history using fresh F0 plus explicit observer masks.

    The mask/readback sources are caller receipts, not Native attestation. A
    first-terminal sample remains archival credit; every later row of that slot
    is inactive even if the original simulator starts another physical episode.
    Retired generation changes require explicit matching readback receipts;
    they never clear the ledger or grant another first episode.
    """
    clocks=_metadata(metadata);n=len(clocks)
    live=_mask(actual_live,n,'actual_live');terminal=_mask(actual_terminal,n,'actual_terminal')
    cohort=_mask(cohort_live,n,'cohort_live')
    _require(all(not t or l for l,t in zip(live,terminal)),'new terminal must have been physically live')
    _require(isinstance(liveness_source,str) and bool(liveness_source.strip()),'liveness source required')
    _require(isinstance(retired_generation_receipts,(list,tuple))
        and all(type(r) is RetiredGenerationReceipt for r in retired_generation_receipts),'typed generation receipts required')
    receipts=tuple(retired_generation_receipts)
    receipt_map={r.after.env_id:r for r in receipts}
    _require(len(receipt_map)==len(receipts),'duplicate generation receipt')
    if previous is None:
        _require(all(live),'initial open cohort must all have an actual first physical sample')
        _require(not receipts,'initial cohort cannot consume retired generation receipts')
        initial=clocks;retired=[False]*n;pinned=[None]*n;old_clocks=None;prior_sha=None
    else:
        _validate_previous(previous,n)
        _require(metadata.binding_fingerprint==previous.metadata.binding_fingerprint,'stage/API/view/path binding changed')
        initial=previous.initial_clocks;retired=list(previous.retired);pinned=list(previous.first_terminals)
        old_by_env={c.env_id:c for c in previous.metadata.clocks};old_clocks=tuple(old_by_env[i] for i in range(n))
        prior_sha=previous.receipt_sha256
    expected=tuple(live[i] and not retired[i] for i in range(n))
    _require(cohort==expected,'cohort mask cannot revive a retired first episode')
    _require(any(cohort),'no live first-cohort sample; retain previous terminal ledger unchanged')
    used=set()
    for i,c in enumerate(clocks):
        if old_clocks is not None:_clock_step(old_clocks[i],c)
        if not retired[i]:
            _require(live[i],'open first episode cannot disappear without a terminal')
            _require((c.run_uuid,c.episode_uid,c.reset_generation)==
                (initial[i].run_uuid,initial[i].episode_uid,initial[i].reset_generation),'open first episode identity changed')
            _require(c.terminal==terminal[i] and not c.reset,'actual first terminal differs from its clock')
            if terminal[i]:pinned[i]=c;retired[i]=True
        else:
            old=old_clocks[i]
            changed=(c.episode_uid,c.reset_generation)!=(old.episode_uid,old.reset_generation)
            if changed:
                receipt=receipt_map.get(i)
                _require(receipt is not None and receipt.before==old and receipt.after==c,
                    'retired generation change requires an exact explicit readback receipt')
                used.add(i)
            elif c.episode_uid==initial[i].episode_uid:
                _require(c.terminal,'original retired episode cannot clear its terminal')
    _require(used==set(receipt_map),'unused or inapplicable retired generation receipt')
    pairs=tuple(p for p in metadata.pairs if cohort[p.clock.env_id])
    counts=tuple(sum(p.clock.env_id==i for p in pairs) for i in range(n))
    _require(counts==tuple(34 if v else 0 for v in cohort),'active metadata pair count mismatch')
    state=ContactEpochState(metadata,tuple(initial),tuple(retired),tuple(pinned),live,terminal,cohort,
        pairs,counts,receipts,liveness_source,prior_sha,'')
    return replace(state,receipt_sha256=_digest(_history_payload(state)))
