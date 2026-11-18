"""Relocate USD dependency closure without changing the source layers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from pxr import Sdf,Usd,UsdUtils
from tools.migration_copy import copy_file,sha256

BUILTINS={'OmniPBR.mdl'}


def relocate(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(source))
    missing=set(unresolved)-BUILTINS
    if missing:raise ValueError(f'unresolved assets: {sorted(missing)}')
    paths={Path(l.realPath).resolve() for l in layers}|{Path(a).resolve() for a in assets}
    mapping={p:output/('dependencies/'+hashlib.sha256(str(p).encode()).hexdigest()[:12]+'_'+p.name) for p in paths}
    mapping[source]=output/source.name
    output.mkdir(parents=True)
    records=[]
    layer_paths={Path(l.realPath).resolve() for l in layers}
    for old in sorted(paths-layer_paths):
        records.append(copy_file(old,output,mapping[old].relative_to(output),'usd_dependency'))
    for layer in layers:
        old=Path(layer.realPath).resolve();new=mapping[old];new.parent.mkdir(parents=True,exist_ok=True)
        clone=Sdf.Layer.CreateAnonymous();clone.TransferContent(layer)
        def rewrite(value):
            if not value or value in BUILTINS:return value
            resolved=Path(Sdf.ComputeAssetPathRelativeToLayer(layer,value)).resolve()
            if resolved not in mapping:raise ValueError(f'undeclared dependency: {value}')
            return os.path.relpath(mapping[resolved],new.parent)
        UsdUtils.ModifyAssetPaths(clone,rewrite)
        if not clone.Export(str(new)):raise RuntimeError('USD export failed')
        records.append(dict(source=str(old),destination=str(new.relative_to(output)),
            source_sha256=sha256(old),sha256=sha256(new),category='relocated_usd'))
    stage=Usd.Stage.Open(str(mapping[source]))
    if stage is None:raise ValueError('relocated stage failed to open')
    if any(l.realPath and not Path(l.realPath).resolve().is_relative_to(output) for l in stage.GetUsedLayers()):
        raise ValueError('relocated USD still uses external layers')
    report=dict(root=str(mapping[source]),layers=records,external_software_assets=sorted(set(unresolved)&BUILTINS))
    (output/'relocation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('output')
    args=parser.parse_args();print(json.dumps(relocate(args.source,args.output),indent=2))
