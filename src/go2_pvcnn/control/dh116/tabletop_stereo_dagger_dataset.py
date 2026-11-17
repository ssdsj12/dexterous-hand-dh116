"""Append-only visited-state aggregation, separate from successful BC and PPO."""
import copy

from .tabletop_stereo_contract import make_contract
from .tabletop_stereo_supervision_dataset import (
    VERSION,rows_from_records,validate_dataset,save_dataset,load_dataset,batch_dataset,
)


class DAggerDataset:
    def __init__(self,*,source_sha256,contract=None,schema_only=False,qualification=None):
        self.source_sha256=source_sha256;self.contract=copy.deepcopy(contract)
        self.schema_only=schema_only;self.qualification=copy.deepcopy(qualification)
        self._rows=[]

    def __len__(self):return len(self._rows)

    def _payload(self,rows,contract):
        return dict(version=VERSION,contract=copy.deepcopy(contract),source_sha256=self.source_sha256,
            schema_only=self.schema_only,native_qualified=False,validation_scope='schema_only_not_native_success',
            purpose='dagger_visited_states',successful_demonstrations=False,qualification=copy.deepcopy(self.qualification),
            rows=copy.deepcopy(rows))

    def append(self,record):
        records=record if isinstance(record,list) else [record]
        if any(row.get('mode')=='teacher_reference' for row in records):
            raise ValueError('D0 Teacher-only reference belongs in a reference pool, not Student DAgger')
        incoming=rows_from_records(records)
        if not incoming:raise ValueError('nonempty executed visited-state rows required')
        contract=self.contract
        if contract is None:contract=make_contract('dagger',{},calibration_sha256=incoming[0]['identity']['calibration_sha256'])
        candidate=self._payload(self._rows+incoming,contract)
        validate_dataset(candidate,allow_schema_only=self.schema_only)
        # Transactional append: a duplicate or bad reset never mutates the pool.
        self.contract=copy.deepcopy(contract);self._rows=candidate['rows']
        return len(incoming)

    def payload(self):
        value=self._payload(self._rows,self.contract)
        return validate_dataset(value,allow_schema_only=self.schema_only)

    def batch(self,indices,*,device='cpu'):
        return batch_dataset(self.payload(),indices,device=device)

    def save(self,path):return save_dataset(self.payload(),path)

    @classmethod
    def load(cls,path,*,allow_schema_only=False,expected_sha256=None):
        payload=load_dataset(path,allow_schema_only=allow_schema_only,expected_sha256=expected_sha256)
        if payload.get('purpose')!='dagger_visited_states':raise ValueError('DAgger visited-state pool required')
        result=cls(source_sha256=payload['source_sha256'],contract=payload['contract'],
            schema_only=payload['schema_only'],qualification=payload['qualification'])
        result._rows=copy.deepcopy(payload['rows'])
        return result
