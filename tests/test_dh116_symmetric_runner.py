from pathlib import Path


def test_explicit_symmetric_mode_is_wired_and_fingerprinted():
    root=Path(__file__).parents[1]
    code=(root/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert "parser.add_argument('--symmetric-lift'" in code
    assert 'symmetric_lift=args.symmetric_lift' in code
    assert "'touch-first-symmetric-lift-v1'" in code
    launcher=(root/'scripts/run_current.py').read_text()
    assert '--config' in launcher and 'symmetric_grasp_v1' in launcher
    assert "cmd.append('--symmetric-lift')" in launcher
