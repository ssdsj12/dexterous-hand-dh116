import json
import numpy as np
import pytest
from go2_pvcnn.control.dh116.evidence_json import dumps


def test_numpy_endpoint_comparison_is_serialized_as_native_boolean():
    value={'approach':{'final_errors':[{'passed':np.float64(.001)<=np.deg2rad(5)}]}}
    restored=json.loads(dumps(value))
    assert restored=={'approach':{'final_errors':[{'passed':True}]}}


def test_arrays_and_scalars_convert_without_accepting_nonfinite_evidence():
    assert json.loads(dumps({'a':np.array([1.,2.]),'n':np.int64(2)}))=={'a':[1.,2.],'n':2}
    with pytest.raises(ValueError):dumps({'a':np.array([np.nan])})
    with pytest.raises(TypeError):dumps({'a':object()})
