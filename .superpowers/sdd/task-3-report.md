# Task 3 report — six-channel runtime/Jacobian/reaction integration

Status: DONE_WITH_CONCERNS

The runner now accepts the middle-hook reference schema, validates six ordered
references, builds six moving mesh states, submits all three digits per hand
to the reaction sink, and allocates closure Jacobians over thumb/index/middle
channels. The middle channels are represented in the active mask and can be
held inactive until preload; collision source delegation includes
`finger33_link`. Per-step records now expose `middle_hook_load_n`,
`middle_hook_valid`, `middle_hook_direction_w`, and
`middle_hook_contact_duration_s`.

`coordinated_closure` supports both the new `(2,3,...)` contract and the
legacy two-pad helper shape, while all production tabletop paths use six.

Verification:

```text
PYTHONPATH=src bash scripts/python.sh -m pytest -q \
  tests/test_dh116_middle_hook_runtime.py tests/test_dh116_coordinated_closure.py
38 passed
python -m py_compile scripts/verify_fixed_dual_panda_dh116.py \
  src/go2_pvcnn/control/dh116/coordinated_closure.py
```

Concern for Task 4: the broad legacy integration suite still contains old
four-channel fixtures and needs migration or exclusion from the new six-channel
run. No Isaac physical close/hold/lift success is claimed here.
