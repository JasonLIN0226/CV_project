from watchdog import inspect_allocation, epoch
import pytest


def native(state='CONFIGURING', **changes):
    return dict(Account='csci_ga_2271-2026fa', Partition='c12m85-a100-1',
        JobState=state, AllocTRES='cpu=12,mem=60G,gres/gpu=1',
        StartTime='2026-10-19T05:00:00', EndTime='Unknown', Restarts='0', **changes)


def test_stalled_startup_is_cancelled_before_batch_entrypoint():
    start = epoch('2026-10-19T05:00:00')
    record, reason = inspect_allocation({}, native(), start+601, startup_limit=600, allocation_limit=900)
    assert record['charged_seconds'] == 601 and reason.startswith('Cloud startup')


def test_queue_never_masquerades_as_gpu_time():
    record, reason = inspect_allocation({}, native('PENDING', **{} ) | {'AllocTRES':'(null)', 'StartTime':'Unknown'},
        9999999, startup_limit=600, allocation_limit=900)
    assert record['charged_seconds'] == 0 and reason is None


def test_restart_cannot_erase_prior_configuration_charge():
    start = epoch('2026-10-19T05:00:00')
    record, _ = inspect_allocation({}, native(), start+300, startup_limit=600, allocation_limit=900)
    record, reason = inspect_allocation(record, native('PENDING') | {'AllocTRES':'(null)', 'StartTime':'Unknown', 'Restarts':'1'},
        start+700, startup_limit=600, allocation_limit=900)
    assert record['charged_seconds'] == 700 and reason.startswith('Scheduler restarted')


def test_running_start_reset_does_not_hide_boot_time():
    start = epoch('2026-10-19T05:00:00')
    record, _ = inspect_allocation({}, native(), start+200, startup_limit=600, allocation_limit=900)
    record, reason = inspect_allocation(record, native('RUNNING') | {'StartTime':'2026-10-19T05:03:00'},
        start+871, startup_limit=600, allocation_limit=900)
    assert record['charged_seconds'] == 871 and reason.startswith('Whole allocation deadline')


def test_other_users_account_is_never_a_cancellation_target():
    with pytest.raises(ValueError, match='unrelated account'):
        inspect_allocation({}, native() | {'Account':'other'}, 999, startup_limit=600, allocation_limit=900)


def test_running_allocation_without_batch_entry_is_still_startup():
    start = epoch('2026-10-19T05:00:00')
    _, reason = inspect_allocation({}, native('RUNNING'), start+181,
        startup_limit=180, allocation_limit=900, batch_started=False)
    assert reason.startswith('Cloud startup')


def test_real_batch_entry_allows_science_beyond_startup_limit():
    start = epoch('2026-10-19T05:00:00')
    _, reason = inspect_allocation({}, native('RUNNING'), start+181,
        startup_limit=180, allocation_limit=900, batch_started=True)
    assert reason is None


def test_enabled_requeue_stops_even_before_first_restart():
    start = epoch('2026-10-19T05:00:00')
    _, reason = inspect_allocation({}, native('RUNNING') | {'Requeue':'1'}, start+1,
        startup_limit=180, allocation_limit=900, batch_started=True)
    assert reason.startswith('Scheduler requeue is enabled')


def test_terminal_result_is_reconciled_without_spurious_cancellation():
    start = epoch('2026-10-19T05:00:00')
    record, reason = inspect_allocation({}, native('CANCELLED') | {
        'EndTime':'2026-10-19T05:02:00', 'Restarts':'56'}, start+1000,
        startup_limit=180, allocation_limit=900, batch_started=False)
    assert record['charged_seconds'] == 120 and reason is None
