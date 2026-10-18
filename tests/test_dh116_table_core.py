import hashlib
import os
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import pytest

from go2_pvcnn.control.dh116.table_core import AngleTable
from go2_pvcnn.control.dh116.table_source import read_samples


def test_values_and_derivative_share_interpolant():
    x = np.linspace(0., 1., 11)
    y = x*x + x
    table = AngleTable(x, y)
    np.testing.assert_allclose(table.evaluate(x)[0], y, atol=1e-10)
    q = np.array([[.15, .35], [.55, .75]])
    value, derivative = table.evaluate(q)
    numeric = (table.evaluate(q+1e-6)[0]-table.evaluate(q-1e-6)[0])/2e-6
    np.testing.assert_allclose(derivative, numeric, atol=1e-5, rtol=0)
    assert value.shape == derivative.shape == q.shape


@pytest.mark.parametrize('q', [-.001, 1.001, np.nan, np.inf, -np.inf])
def test_no_extrapolation_or_nonfinite(q):
    table = AngleTable([0., 1.], [0., -1.])
    before = table.evaluate(.5)
    with pytest.raises(ValueError):
        table.evaluate(q)
    np.testing.assert_equal(table.evaluate(.5), before)


@pytest.mark.parametrize('x,y', [([], []), ([0.], [0.]),
    ([0., 0.], [0., 1.]), ([1., 0.], [0., 1.]),
    ([0., 1.], [0.]), ([0., 1.], [0., np.nan]),
    ([0., np.inf], [0., 1.]), ([[0., 1.]], [[0., 1.]])])
def test_bad_samples_rejected(x, y):
    with pytest.raises(ValueError):
        AngleTable(x, y)


def test_input_mutation_does_not_change_table():
    x = np.array([0., 1.]); y = np.array([0., -2.])
    table = AngleTable(x, y)
    x[:] = 4.; y[:] = 10.
    assert table.evaluate(.5)[0] == pytest.approx(-1.)


def test_monotone_values_and_continuous_derivative():
    x = np.array([0., .2, .5, 1.])
    y = np.array([0., -.1, -.8, -1.])
    table = AngleTable(x, y)
    for i in range(len(x)-1):
        value, derivative = table.evaluate(np.linspace(x[i], x[i+1], 50))
        assert np.all(np.diff(value) <= 1e-14)
        assert value.min() >= y[i+1]-1e-14
        assert value.max() <= y[i]+1e-14
        assert np.all(derivative <= 1e-14)
    np.testing.assert_allclose(table.evaluate(x[1:-1]-1e-9)[1],
                               table.evaluate(x[1:-1]+1e-9)[1], atol=1e-6)


def test_endpoint_one_sided_derivative_converges():
    table = AngleTable([0., .2, .5, 1.], [0., .3, .9, 1.4])
    for endpoint, sign in [(0., 1.), (1., -1.)]:
        value, derivative = table.evaluate(endpoint)
        errors = []
        for h in [1e-4, 1e-5, 1e-6]:
            numeric = (table.evaluate(endpoint+sign*h)[0]-value)/(sign*h)
            errors.append(abs(numeric-derivative))
        assert errors[-1] < errors[0]
        assert errors[-1] < 1e-5


def workbook(tmp_path, rows=None, headers=('x', 'y'), missing=False,
             formula=False, cell_type='n'):
    rows = [(0., 0.), (30., -45.)] if rows is None else rows
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    data = ['<row r="4"><c r="B4" t="s"><v>0</v></c>'
            '<c r="C4" t="s"><v>1</v></c></row>']
    for r, (x, y) in enumerate(rows, 5):
        c = '' if missing else f'<c r="C{r}" t="{cell_type}">' + (
            '<f>1+1</f>' if formula else '') + f'<v>{y}</v></c>'
        data.append(f'<row r="{r}"><c r="B{r}"><v>{x}</v></c>{c}</row>')
    path = tmp_path/'sample.xlsx'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}">' +
                   ''.join(f'<si><t>{escape(h)}</t></si>' for h in headers)+'</sst>')
        z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData>'
                   + ''.join(data) + '</sheetData></worksheet>')
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_samples_preserves_sign_and_converts_units(tmp_path):
    p, sha = workbook(tmp_path)
    x, y = read_samples(p, sha, ('x', 'y'), 2)
    np.testing.assert_allclose(x, np.deg2rad([0., 30.]))
    np.testing.assert_allclose(y, np.deg2rad([0., -45.]))


def test_wrong_sha_rejected_before_parsing(tmp_path):
    p = tmp_path/'source.xlsx'; p.write_bytes(b'not a workbook')
    with pytest.raises(ValueError, match='SHA'):
        read_samples(p, '0'*64, ('x', 'y'), 2)


@pytest.mark.parametrize('kwargs', [dict(headers=('wrong', 'y')),
    dict(missing=True), dict(formula=True), dict(cell_type='s'),
    dict(rows=[(0., 0.), (0., 1.)]), dict(rows=[(1., 0.), (0., 1.)]),
    dict(rows=[(0., 0.), (1., 'nan')]), dict(rows=[(0., 0.), (1., 'text')]),
    dict(rows=[(0., 0.)])])
def test_malformed_sources_rejected(tmp_path, kwargs):
    p, sha = workbook(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        read_samples(p, sha, ('x', 'y'), 2)


def test_invalid_zip_has_readable_error(tmp_path):
    p = tmp_path/'bad.xlsx'; p.write_bytes(b'not a workbook')
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='DH116 table'):
        read_samples(p, sha, ('x', 'y'), 2)


@pytest.mark.parametrize('name,sha,count,header', [
    ('大拇指', '74def5596dced4c5eb5851cb8f96c931897bad2b79abe6fe1fa1c19e7bfc5aaf',
     501, '指尖角位置 (deg)'),
    ('四指', 'd0f65ea0b951b982645fa9c9562e627549a068b738673ab8a20536ac8263c926',
     801, '指尖角位移13 (deg)'),
])
def test_pinned_manufacturer_tables(name, sha, count, header):
    root = os.environ.get('DH116_TABLE_SOURCE_DIR')
    if not root:
        pytest.skip('set DH116_TABLE_SOURCE_DIR for real XLSX verification')
    path = Path(root)/f'{name}被动角与驱动角的关系-20260112.xlsx'
    x, y = read_samples(path, sha, ('指节角位移 (deg)', header), count)
    table = AngleTable(x, y)
    node_error = float(np.max(abs(table.evaluate(x)[0]-y)))
    assert node_error <= 1e-10
    midpoint = (x[:-1]+x[1:])/2
    h0 = np.minimum(np.diff(x)/20, 1e-6)
    errors = []
    for scale in (1., .1, .01):
        h = h0*scale
        numeric = (table.evaluate(midpoint+h)[0]-table.evaluate(midpoint-h)[0])/(2*h)
        errors.append(float(np.max(abs(table.evaluate(midpoint)[1]-numeric))))
    assert errors[-1] <= 1e-5
    # A second-order one-sided stencil avoids O(h) endpoint truncation error.
    endpoint_errors = []
    for index, sign, width in [(0, 1, x[1]-x[0]), (-1, -1, x[-1]-x[-2])]:
        value, derivative = table.evaluate(x[index])
        sequence = []
        for scale in (1., .1, .01):
            h = min(width/20, 1e-5)*scale
            numeric = (-3*value + 4*table.evaluate(x[index]+sign*h)[0]
                       - table.evaluate(x[index]+sign*2*h)[0])/(2*sign*h)
            sequence.append(float(abs(numeric-derivative)))
        assert sequence[-1] <= 1e-5
        endpoint_errors.append(sequence)
    print({'table': name, 'count': count, 'node_error_rad': node_error,
           'mid_derivative_errors': errors, 'endpoint_derivative_errors': endpoint_errors})
