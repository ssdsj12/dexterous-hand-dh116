"""Explicit fresh F0 metadata acquisition from existing views, never physics.

No SDK imports, sensor/view/World construction, numeric contact getter, refresh,
step, setting or target writes. Every metadata getter is copied before another
getter can reuse its borrowed storage. Source and actual batch clocks bracket
acquisition. The reviewed pure resolver alone supplies the pair mapping; all
numeric IDs and physical/Native/planning/support qualifications remain NA/False.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

from .tabletop_bimanual_mpc_filtered_contact_provenance import (
    FilteredPairSnapshot, _clocks, _freeze, resolve_filtered_sensor_pairs,
)


def _type_name(value):
    return type(value).__module__+'.'+type(value).__qualname__


def _view_identity(value):
    if value is None:
        raise ValueError('actual existing view is unavailable')
    return dict(python_runtime_id=id(value),python_type=_type_name(value))


def _invalid(raw,errors):
    return FilteredPairSnapshot(False,tuple(dict.fromkeys(errors)),(),(),raw,None)


class FilteredPairMetadataProvider:
    """A stateless acquisition: failed reads never reuse an earlier snapshot.

    ``clock_reader()`` returns actual batch SampleClocks, not a caller flag.
    ``source_reader()`` returns actual current stage/API identity. Injected CPU
    sources cannot attest Native execution. The caller explicitly supplies any
    valid ``previous`` to the pure resolver; resets never silently clear it.
    """
    def __init__(self,*,sensors,wire_root_view,clock_reader,source_reader,enabled=False):
        if type(enabled) is not bool:
            raise ValueError('enabled must be an actual boolean')
        self.enabled=enabled
        self.sensors,self.wire_root_view=sensors,wire_root_view
        self.clock_reader,self.source_reader=clock_reader,source_reader

    def capture(self,clocks,previous=None):
        acquisition=dict(enabled=self.enabled,read_brackets=[],getter_errors=[],
            scope='fresh global sensor/filter metadata only; no numeric contacts or force ownership',
            clock_authority='actual caller-installed batch clock reader, not Native attestation')
        if not self.enabled:
            return _invalid(dict(acquisition=acquisition),('filtered metadata provider disabled',))
        errors=[]
        try:
            acquisition['expected_clocks']=_freeze(clocks)
            expected,expected_by_env=_clocks(clocks,clocks)
            acquisition['expected_clocks']=expected
        except Exception as exc:
            acquisition['input_error']=type(exc).__name__+': '+str(exc)
            return _invalid(dict(acquisition=acquisition),(acquisition['input_error'],))

        def checked_clock(name):
            try:
                current=self.clock_reader()
                owned,by_env=_clocks(current,current)
                if by_env!=expected_by_env:
                    errors.append(name+': actual batch clock differs from requested sample')
                return owned
            except Exception as exc:
                errors.append(name+': '+type(exc).__name__+': '+str(exc))
                return None

        def read(name,getter,*,required=True,object_reference=False):
            before=checked_clock(name+'.before')
            receipt=dict(name=name,before=before,after=None,error=None,value_type=None)
            value=None
            try:
                borrowed=getter()
                receipt['value_type']=_type_name(borrowed)
                # A view object is held only as an existing getter source. All
                # returned metadata values use the reviewed recursive copier.
                value=borrowed if object_reference else _freeze(borrowed)
            except Exception as exc:
                receipt['error']=type(exc).__name__+': '+str(exc)
                if receipt['value_type'] is not None:
                    try:receipt['unowned_value_repr']=repr(borrowed)
                    except Exception:receipt['unowned_value_repr']='repr unavailable'
                message=name+': '+receipt['error']
                acquisition['getter_errors'].append(message)
                if required:errors.append(message)
            finally:
                receipt['after']=checked_clock(name+'.after')
                acquisition['read_brackets'].append(receipt)
            return value

        before=checked_clock('capture.before')
        source_before=read('source_identity.before',self.source_reader)
        wire_paths=read('wire_root_view.prim_paths',lambda:self.wire_root_view.prim_paths)
        rows=[];held=[]
        if not isinstance(self.sensors,(list,tuple)) or len(self.sensors)!=34:
            errors.append('exactly 34 existing sensor objects in an explicit collection required')
            sensors=()
        else:sensors=tuple(self.sensors)
        acquisition['sensor_object_identities_before']=[id(sensor) for sensor in sensors]
        for index,sensor in enumerate(sensors):
            prefix=f'sensors[{index}]'
            contact=read(prefix+'.contact_physx_view',lambda s=sensor:s.contact_physx_view,object_reference=True)
            body=read(prefix+'.body_physx_view',lambda s=sensor:s.body_physx_view,object_reference=True)
            held.append((sensor,contact,body))
            row=dict(sensor_index=index,source=_type_name(sensor),
                contact_view_identity=_view_identity(contact) if contact is not None else None,
                body_view_identity=_view_identity(body) if body is not None else None)
            row['sensor_paths']=read(prefix+'.sensor_paths',lambda v=contact:v.sensor_paths)
            row['body_prim_paths']=read(prefix+'.body_prim_paths',lambda v=body:v.prim_paths)
            filters=read(prefix+'.filter_paths',lambda v=contact:v.filter_paths)
            row['filter_paths']=dict(available=filters is not None,value=filters,
                source='actual contact_physx_view.filter_paths getter',resolved_semantics='unknown')
            row['filter_count']=read(prefix+'.filter_count',lambda v=contact:v.filter_count)
            row['sensor_count']=read(prefix+'.sensor_count',lambda v=contact:v.sensor_count)
            # Configuration is independent optional raw evidence. Its absence
            # cannot fill a failed actual getter or determine a resolved path.
            row['configured_filter_expressions']=read(prefix+'.configured_filter_expressions',
                lambda s=sensor:s.cfg.filter_prim_paths_expr,required=False)
            rows.append(row)

        source_after=read('source_identity.after',self.source_reader)
        # Validate references after the final source receipt as well: a source
        # reader cannot hide a changed view behind an unchanged API label.
        for index,(sensor,contact,body) in enumerate(held):
            contact_after=read(f'sensors[{index}].contact_physx_view.after',
                lambda s=sensor:s.contact_physx_view,object_reference=True)
            body_after=read(f'sensors[{index}].body_physx_view.after',
                lambda s=sensor:s.body_physx_view,object_reference=True)
            if contact_after is not contact or body_after is not body:
                errors.append(f'sensors[{index}]: actual view identity changed during acquisition')
        if isinstance(self.sensors,(list,tuple)):
            acquisition['sensor_object_identities_after']=[id(sensor) for sensor in self.sensors]
            if len(self.sensors)!=len(sensors) or any(a is not b for a,b in zip(sensors,self.sensors)):
                errors.append('actual sensor object collection changed during acquisition')
        after=checked_clock('capture.after')
        result=resolve_filtered_sensor_pairs(dict(available=True,sensors=rows,
            per_contact_partner_id_recorded=False),wire_body_paths=wire_paths,
            clocks_before=before or (),clocks_after=after or (),
            source_identity=dict(before=source_before,after=source_after),previous=previous)
        raw=dict(result.raw,acquisition=acquisition)
        if errors or not result.metadata_binding_valid:
            return _invalid(raw,errors+list(result.errors))
        return replace(result,raw=raw)


def make_source_reader(*,stage,sensors,stage_reader):
    """Read existing stage/view Python source identity, without importing SDKs.

    Construction is lazy. Every call rechecks the actual stage and all 34 view
    classes. File hashes are cached only while current stat identity is equal;
    a file changed during hashing is rejected, without committing that cache.
    File paths remain actual absolute host paths. The reviewed resolver's
    installed-source path contract is Unix; Windows paths are not disguised as
    Unix paths to manufacture a successful F0 binding in a CPU test.
    """
    files={}
    def stat_identity(path):
        stat=path.stat()
        return (str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

    def reader():
        if stage is None or not callable(stage_reader) or stage_reader() is not stage:
            raise ValueError('actual active stage changed or is unavailable; rebuild independently')
        if not isinstance(sensors,(list,tuple)) or len(sensors)!=34:
            raise ValueError('exactly 34 existing sensors required for source identity')
        views=tuple(sensor.contact_physx_view for sensor in sensors)
        if any(view is None for view in views) or len({type(view) for view in views})!=1:
            raise ValueError('all actual contact views must share one current API class')
        kind=type(views[0])
        path=Path(inspect.getfile(kind)).resolve()
        before=stat_identity(path)
        cached=files.get(str(path))
        if cached is None or cached[0]!=before:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if stat_identity(path)!=before:
                raise RuntimeError('actual view API file changed during source hashing')
            cached=(before,dict(path=str(path),sha256=digest))
            files[str(path)]=cached
        root,session=stage.GetRootLayer(),stage.GetSessionLayer()
        layers={}
        for label,layer in (('root',root),('session',session)):
            identifier,real_path=layer.identifier,layer.realPath
            if not isinstance(identifier,str) or not identifier or not isinstance(real_path,str):
                raise ValueError('actual stage root/session layer identity is unavailable')
            layers[label]=dict(identifier=identifier,real_path=real_path)
        stage_evidence=dict(stage_python_runtime_id=id(stage),stage_python_type=_type_name(stage),
            layers=layers)
        if stage_reader() is not stage:
            raise ValueError('actual active stage changed during source identity read')
        if stat_identity(path)!=before:
            raise RuntimeError('actual view API file changed during source identity read')
        if (len(sensors)!=len(views)
                or any(sensor.contact_physx_view is not view for sensor,view in zip(sensors,views))):
            raise RuntimeError('actual contact view lifecycle changed during source identity read')
        return dict(stage_identity=json.dumps(stage_evidence,sort_keys=True,separators=(',',':')),
            stage_python_runtime_id=id(stage),stage_layers=layers,
            contact_view_runtime_ids=[id(view) for view in views],
            contact_view_api=dict(cached[1],python_type=kind.__module__+'.'+kind.__qualname__),
            source_scope='actual existing stage and contact-view Python wrapper; binary semantics unqualified')
    return reader
