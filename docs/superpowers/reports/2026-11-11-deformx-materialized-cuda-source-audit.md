# DeformX CUDA source availability audit

The Ubuntu worktree started from DH116 commit
`5a7eeb41c3d2f4efed98541ad76c375f9a1ee1d2`. The required DeformX CUDA
solver commit is `96b8514c7c55c724f1abba1fbf6ed6642b7db72b`.

## Checkout inventory

`scripts/verify_dh116_deformx_cuda_source.py` enumerates the committed
tree and verifies each initialized submodule against the gitlink in its
parent commit before descending. It excludes missing files, empty required
files, Git LFS pointers and symbolic links from source evidence. Both Git
symlink modes and symlinks in the physical path are checked. The JSON
records each rejected path and reason as well as the expected and actual
submodule commits. Materialized inventory is the only positive claim the
verifier can make. Build verification and solver execution require
separate evidence.

On Ubuntu, the verifier exited with code 2 for both
`/mnt/mydisk/xk/dh116/vendor/DeformX` and
`/mnt/mydisk/xk/integrations/deformx-gpu-20261110/hf-assets-9ef440fa`. Both
reports recorded `not a standalone Git checkout`, with
`source_inventory_available`, `build_verified` and `solver_executed`
all false. The two JSON results have SHA-256
`af4cca075a39d774b4f6d46f09ddb163c428fbe01c52502b5eabaa00be807beb`.

The existing `/usr/local/cuda*/bin/nvcc` executables resolve to Toolkit
12.2.91 and 12.4.99. This inspection found no 12.8 compiler in those
directories. The PyTorch cu128 build reports a CUDA runtime version.

## Verification

The initial Ubuntu CPU test run reproduced five failures against the
original verifier. After the change, all seven tests passed. The cases
include LFS pointers for the required files, a missing CUDA header
payload alongside a valid kernel, tracked symlinks including a plain
checkout stub, physical parent links and a real recursive Git inventory
with an exact pinned submodule and a wrong-pinned checkout. The passing
test log SHA-256 is
`8d0c0cdca84c00aefcec15705f88c5b8c048f448af649c97f1af89ebbb5bc5d0`.

```bash
PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /mnt/mydisk/xk/envs/dh116/bin/python \
  -m pytest -q -p no:cacheprovider tests/test_dh116_deformx_cuda_source.py
/mnt/mydisk/xk/envs/dh116/bin/python scripts/verify_dh116_deformx_cuda_source.py \
  /mnt/mydisk/xk/dh116/vendor/DeformX
/mnt/mydisk/xk/envs/dh116/bin/python scripts/verify_dh116_deformx_cuda_source.py \
  /mnt/mydisk/xk/integrations/deformx-gpu-20261110/hf-assets-9ef440fa
```

WP5 remains blocked at source availability. The next build step requires
the solver checkout at the fixed commit, materialized recursive
submodules and payloads, license review, and a separately approved
CUDA Toolkit 12.8 setup for SM89. This work package used CPU fixtures and
read-only source inspection. Solver build, CUDA execution, Isaac
integration and two-way physical coupling remain unverified.
