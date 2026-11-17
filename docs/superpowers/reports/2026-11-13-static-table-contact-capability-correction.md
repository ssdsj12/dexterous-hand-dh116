# Static GPU table contact filter capability correction

Read-only evidence from the independently executed single-left v4 log:
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f4-pinch-left-v4-native/d7f87b161b3283a31ffe8a7befbbce0050f7d6bcc4b1473020e2e117067847cc/attempt-wezd3fyw/native.log

At lines 119 onward, omni.physx.tensors.plugin reports:
GPU contact filter for collider '/World/PinchTable' is not supported.

The existing dual spawn creates its Table with CuboidCfg collision_props only,
without rigid_props; original USD inspection also shows no RigidBodyAPI.
It therefore has the same static GPU filter capability gap. This is a concrete
reason that the earlier CPU-only 5a6223c reader cannot treat its empty buffers
as measured contact absence. That package was not native-qualified.

The corrected reader detects an actual static Table before creating views.
It returns available=False, total_contacts=None and the explicit unsupported
static GPU filter reason. No table conversion, shape/material/physics change,
new reporter, readback setting, contact view or native attempt is introduced.
The old source remains in Git history; this new source SHA is a successor,
not a reinterpretation of an earlier native receipt.

The synthetic API parsing tests now explicitly use a dynamic USD partner;
that fixture is not the production table and does not claim native capability.
A separate actual static-prim negative test first failed on the previous
reader, then passed after the correction. All 21 contact-reader CPU tests
passed in 3.26 seconds using the installed API class with a fake CPU backend.

Command:
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_bimanual_table_contacts.py

Future dual CPU/native diagnostic composition must preserve this NA and cannot
grant no-contact, grasp or Teacher eligibility. Alternative full contact report
readback is a separate unqualified experiment and is not enabled here.
