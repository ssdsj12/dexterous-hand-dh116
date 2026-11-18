import pytest

from go2_pvcnn.control.dh116.tabletop_solver_order import verify_solver_order


def test_old_engine_default_is_explicitly_unverified():
    result = verify_solver_order(object(), False)
    assert result['actual'] is None
    assert result['verified'] is False


def test_old_engine_cannot_claim_contact_last_enabled():
    with pytest.raises(RuntimeError, match='cannot enable or verify'):
        verify_solver_order(object(), True)


def test_new_engine_must_match_requested_order():
    class Context:
        def get_solve_articulation_contact_last(self):
            return False
    assert verify_solver_order(Context(), False)['verified'] is True
    with pytest.raises(RuntimeError, match='mismatch'):
        verify_solver_order(Context(), True)


def test_requested_order_requires_boolean():
    with pytest.raises(RuntimeError, match='must be bool'):
        verify_solver_order(object(), 0)
