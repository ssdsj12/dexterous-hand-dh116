"""Strict, offline reader for the pinned DH116 20260112 XLSX templates.

This extracts measured angle pairs only. Returned samples are not USD joint
angles until a separately evidenced coordinate alignment has been applied.
"""

import hashlib
import io
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .table_core import AngleTable


NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_samples(path, expected_sha256, expected_headers, expected_count):
    """Read literal B/C samples after row 4; validate identity before parsing.

    Do not sort, repair, discard, or extrapolate source samples. Caller pins
    the file SHA, two exact header strings and expected sample count.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('source SHA mismatch')
    try:
        return _parse(raw, expected_headers, expected_count)
    except (ValueError, KeyError, IndexError, TypeError, ET.ParseError,
            zipfile.BadZipFile) as exc:
        raise ValueError(f'invalid DH116 table: {exc}') from exc


def _parse(raw, expected_headers, expected_count):
    if (isinstance(expected_count, bool) or not isinstance(expected_count, int)
            or expected_count < 2 or len(expected_headers) != 2):
        raise ValueError('expected two headers and at least two samples')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            strings = [''.join(t.text or '' for t in n.findall('.//m:t', NS))
                       for n in root.findall('m:si', NS)]
        root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = root.findall('./m:sheetData/m:row', NS)
        values, header_seen = [], False
        previous_row = 0
        for row in rows:
            r = int(row.attrib['r'])
            if r <= previous_row:
                raise ValueError('duplicate or out-of-order rows')
            previous_row = r
            if r < 4:
                continue
            if r != 4 + len(values) + int(header_seen):
                raise ValueError('nonconsecutive sample rows')
            cells = {}
            for c in row.findall('m:c', NS):
                ref = c.attrib['r']
                if ref in cells:
                    raise ValueError('duplicate cell reference')
                cells[ref] = c
            pair = []
            for col in ('B', 'C'):
                c = cells.get(f'{col}{r}')
                if c is None:
                    raise ValueError('missing B/C cell')
                v = c.find('m:v', NS)
                if v is None or v.text is None or c.find('m:f', NS) is not None:
                    raise ValueError('expected literal cell value')
                if r == 4:
                    if c.get('t') != 's':
                        raise ValueError('expected shared-string header')
                    index = int(v.text)
                    if not 0 <= index < len(strings):
                        raise ValueError('invalid shared-string index')
                    pair.append(strings[index])
                else:
                    if c.get('t', 'n') != 'n':
                        raise ValueError('expected literal numeric samples')
                    pair.append(float(v.text))
            if r == 4:
                if tuple(pair) != tuple(expected_headers):
                    raise ValueError('table header mismatch')
                header_seen = True
            else:
                values.append(pair)
        if not header_seen or len(values) != expected_count:
            raise ValueError('sample count/header mismatch')
    samples = np.deg2rad(np.asarray(values, dtype=np.float64))
    AngleTable(samples[:, 0], samples[:, 1])
    return samples[:, 0].copy(), samples[:, 1].copy()
