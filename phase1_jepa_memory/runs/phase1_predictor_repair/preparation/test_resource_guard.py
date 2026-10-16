"""Synthetic accounting fixtures only; no scientific measurements or model calls."""
import pytest
from resource_guard import reconcile_prior, elapsed_from_allocation, full_sequence_projection


def native_rows():
    return [{"job_id": j, "state": s, "elapsed_raw": n,
             "account": "csci_ga_2271-2026fa", "allocated_gpus": 1}
            for j, s, n in [(7477, "FAILED", 100), (7480, "FAILED", 126), (7485, "COMPLETED", 6021)]]


def proposal():
    return dict(new_gpu_seconds_charged=100,
                candidate_epoch_seconds={"gate": 2., "motion": 2.},
                reference_epoch_seconds={"gate": 2., "motion": 2.},
                development_epochs_done={t: {17: 1, 29: 0, 43: 0} for t in ("gate", "motion")},
                encoder_frames_per_second=200., confirmation_frame_bound=72928,
                encoder_equivalence_passed=True)


def test_failed_partial_is_charged_and_missing_native_record_rejected():
    assert reconcile_prior(native_rows()) == {"used_seconds": 6247, "remaining_seconds": 11753, "cap_seconds": 18000}
    with pytest.raises(ValueError):
        reconcile_prior(native_rows()[1:])


def test_cpu_or_duplicate_allocation_does_not_fake_gpu_remaining_balance():
    rows = native_rows()
    rows[0]["allocated_gpus"] = 0
    with pytest.raises(ValueError):
        reconcile_prior(rows)
    with pytest.raises(ValueError):
        reconcile_prior(native_rows() + native_rows()[:1])


def test_startup_wall_time_is_included_and_rounded_up():
    assert elapsed_from_allocation("2026-10-15T17:44:22Z", "2026-10-15T17:46:22.01Z") == 121
    with pytest.raises(ValueError):
        elapsed_from_allocation("2026-10-15T17:44:22", "2026-10-15T17:46:22Z")


def test_complete_confirmation_is_reserved_even_when_development_fits_alone():
    p = proposal()
    p["reference_epoch_seconds"] = {"gate": 15., "motion": 15.}
    result = full_sequence_projection(**p)
    assert result["fits_shared_cap"] is False
    assert result["confirmation_models"] == 12
    assert result["phase1_go"] is False
    assert full_sequence_projection(**proposal())["fits_shared_cap"] is True


def test_unmeasured_or_nonfinite_rate_and_smaller_scope_fail_closed():
    for key, replacement in [("encoder_frames_per_second", float("nan")),
                             ("encoder_equivalence_passed", False),
                             ("confirmation_frame_bound", 1000)]:
        p = proposal(); p[key] = replacement
        with pytest.raises(ValueError):
            full_sequence_projection(**p)
    p = proposal(); del p["development_epochs_done"]["gate"][43]
    with pytest.raises(ValueError):
        full_sequence_projection(**p)


def test_spent_preflight_time_can_turn_a_fit_into_a_real_budget_stop():
    p = proposal(); p["new_gpu_seconds_charged"] = 8000
    assert full_sequence_projection(**p)["fits_shared_cap"] is False
