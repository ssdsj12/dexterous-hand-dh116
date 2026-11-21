# Pinch native checkpoint resume B v2

The second independently launched camera-free process reverified A and loaded its exact checkpoint into a fresh physical episode. It restored the A RNG state and completed one accepted PPO update on eight new on-policy controls. Cumulative evidence is 16 controls and two accepted updates on policy versions 1 and 2. The physical ledger covers 64 measured SDK steps after two setup steps, 768 DeformX solver substeps and 41 material nodes. Outer and GDB inferior exit were both zero within the original 180/240 second limits. The original geometry and physical parameter contract was used.

| Evidence | Value |
|---|---:|
| Read-only A recheck before launch | 23.940 s |
| B native child wall time | 176.585 s |
| Time remaining after PPO | 18.852 s |
| B policy parameter change L2 | 0.003452933 |
| Guarded full-rate KL | 0.08048153 |
| Accepted post-step KL | 0.00080486 |
| Accepted rate | 1e-05 |

A scheduler receipt `475f58af9f1a704a3c1a42d68b139f768397f6409a8305d56afe0f6f5fc29b91`. B scheduler receipt `ee36b494ae2032476540c9ef9cb88257c3b0eae15efd8ab087e902966d89fcca`. A checkpoint `addf12d42ffd3e03f0e896e5b8b3a25a99f410f1c2440048e03478c72891f0df`. B checkpoint `9007431f17d39e714b7d33ded879ac2ba2c2d2e667726039aaf8796035431025`. Source closure `7d7b385d0d576bc5352e584cdd2cf83e6d842ec4a1251dfdf268927388fb8f91`. The original v1 wall-budget failure remains in its diagnostic report.

Revalidation used `Repository.verify_actual_resume_diagnostic` against the stored native receipt and private run directory. CPU verification uses `python -m pytest -q tests/test_dh116_work_packages.py tests/test_native_teacher_actual_resume_launch.py tests/test_native_teacher_actual_launch_process.py tests/test_native_teacher_actual_resume_artifacts.py`. Native source model provenance remains `owner_source_unqualified`. The diagnostic reward exercised transport and optimizer restore. Grasp, complete task reward and Teacher admission remain unqualified. The next physical work is fixed-wrist, measured opposing contact, then release from supports before lift progress can count.
