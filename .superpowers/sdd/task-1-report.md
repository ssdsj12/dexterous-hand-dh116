# Task 1 report — DH116 middle-hook contract

Status: DONE_WITH_CONCERNS

Implemented the six-channel DH116 contact contract in the runtime evidence,
restore, stage-state, material tracking, contact manifest, and staged touch
paths. Channel order is explicitly:

`left_thumb, left_index, left_middle, right_thumb, right_index, right_middle`.

The evidence gate now requires thumb/index pinch opposition plus measured
upward middle-finger support on both hands before entering hold; hold/lift
continues to require all six valid contacts and rejects slip or missing
reaction evidence. The staged speed path emits a `(2, 3)` pad-speed matrix and
keeps middle fingers parked until measured thumb/index opposition, then allows
the middle hook during preload. The low-level touch/governor helpers retain a
two-column compatibility path for unrelated legacy unit tests; the tabletop
sequence, gate, restore state, and production contract are six-channel only.

Verification:

```text
PYTHONPATH=src bash scripts/python.sh -m pytest -q \
  tests/test_dh116_middle_hook_contract.py \
  tests/test_dh116_touch_first.py \
  tests/test_dh116_speed_profile.py
39 passed
```

`python -m py_compile` passed for all modified DH116 modules and `git diff
--check` passed.

Concern for the next task: several older four-channel integration fixtures
still need migration to the six-channel contract before the broad suite can
be green. They are intentionally not used as evidence for the new contract.
No Isaac or physical lift claim is made by this task.
