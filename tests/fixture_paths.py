"""Explicit independent fixtures; archived evidence is read-only."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'archive/dh116_v2'
ASSET=ROOT/'assets/fixed_dual_panda_dh116_v1/fixed_dual_panda_dh116.usd'
BACKEND=ROOT/'vendor/DeformX'
CONTACT=ROOT/'configs/table_margin_raise1mm/contact/manifest.json'
CONTACT_SHA=hashlib.sha256(CONTACT.read_bytes()).hexdigest()
SURFACE_REFS=ROOT/'configs/surface_references.json'
