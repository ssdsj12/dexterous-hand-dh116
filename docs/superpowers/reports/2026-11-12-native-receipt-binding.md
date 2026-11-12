# DH116 native thumb prerequisite receipt

Base commit `eb31c793d60cb50eb8d6dd2e06bae2f15c5bd8d5`.

## Receipt and launch contract

- The right recipe declares `dependency_receipt.package` as `flex.f4-left-contact-native` and `dependency_receipt.qualification_scope` as `dh116_left_thumb_deformx_contact_lifecycle`. The scheduler requires the same existing native prerequisite and scope in the registry.
- The right recipe uses the exact pairs `--left-scheduler-receipt {left_scheduler_receipt}` and `--left-receipt-sha256 {left_receipt_sha256}`. The scheduler locates the unique private producer receipt whose SHA-256 matches the published result for the current command and code/config identities.
- The scheduler verifies the approved intent, outer exit zero, shared native lock evidence, log digest and required physical output artifacts. It records the resolved dependency in the private intent and checks it again before launch. The public receipt keeps the dependency digest and launch record digest. A restarted scheduler reuses the valid left receipt and retains failed or invalidated evidence for inspection.
- Optional `debugger: gdb_on_segv` binds the installed `/usr/bin/gdb` digest. The same authorized attempt supplies the physical arguments to the inferior through `scripts/python.sh`. GDB records the original child signal, current and all-thread backtraces, loaded libraries and process mappings in `native.log`. The scheduler records actual outer and inferior statuses and independently verifies both when a debugger-produced receipt is later consumed.

## CPU validation

```sh
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_work_packages.py tests/test_dh116_deformx_thumb_contact.py
```

The combined CPU regression passed 51 tests. The protection inventory matched 528 source, script and config files and two historical checkpoints. The regressions exercise recovery without repeating the predecessor, changed receipt or physical ledger, an outer exit of 139, missing outer evidence, and withdrawal of a dependent qualification after evidence changes. Linux CPU fixture processes under the installed GDB exercise preserved quoted arguments, actual shared-lock ownership, a normal exit, a real SIGSEGV with a recorded stack, and a handled SIGBUS followed by a separately captured SIGSEGV. GDB returned outer 255 for the signalled fixture and reported inferior signal 11. The run records each value as observed.

The CPU fixtures confer software test coverage. Native thumb qualification requires the approved single-instance Isaac executions and independent validation of their physical artifacts.

## Budget categories for future recipes

Keep the approved left and conditional right contact recipes at 240 physics ticks, one environment and zero updates. A lifecycle smoke recipe can declare a zero or short physics budget. A reset smoke recipe can declare an explicit number of resets and physics ticks. A future Teacher diagnostic should declare environments, rollout steps, control rates, physics ticks and ledger rows separately. At 60 Hz control and 480 Hz physics, 32 policy steps require 256 ticks per environment before any declared warm-up. A full Hold evaluation needs at least 2,400 physics ticks for the Hold interval alone, plus approach and Lift. Those purposes need separately frozen recipes and their own authorization. This change leaves the current budget limits intact.

## Own-receipt revalidation on recovery

The scheduler reuses the same independent validation for its own successful receipts before planning dependent execution or publication. It compares the receipt with the scheduled intent, actual outer status, elapsed budget, shared lock, log and physical artifacts. A deferred native publication journal checks the same qualification when resumed. Rejected evidence stays at the original location and the attempted fingerprint stays consumed.

CPU regressions first reproduced three acceptance errors: claiming qualification after an independently recorded outer exit of 139, changing the physical ledger after a genuine success, and changing the ledger after a publication interruption. All three are now rejected. The full scheduler and thumb CPU regression passed 54 tests in 15.37 seconds.
