from cleanup_receipts import cleanup_all


def row(job, state='COMPLETED'):
    return {'job': job, 'state': state, 'account': 'csci_ga_2271-2026fa',
            'tres': {'gres/gpu': '1'}}


def test_purged_completed_job_does_not_block_next_active_receipt():
    cancelled, saved = [], []
    result = cleanup_all([8951, 9030],
        accounting=lambda ids: [row(ids[0], 'COMPLETED' if ids[0] == 8951 else 'RUNNING')],
        strict_cancel=cancelled.append, persist=saved.append)
    assert cancelled == [9030]
    assert result[0]['verified_terminal']
    assert not result[1]['verified_terminal']  # scancel success is not terminal proof.
    assert len(saved) == 2


def test_missing_accounting_is_unresolved_but_other_jobs_still_attempted():
    cancelled = []
    def accounting(ids):
        if ids == [8951]:
            raise RuntimeError('Accounting unavailable')
        return [row(ids[0], 'RUNNING')]
    result = cleanup_all([8951, 9030], accounting=accounting,
                         strict_cancel=cancelled.append, persist=lambda _: None)
    assert 'error' in result[0] and not result[0]['verified_terminal']
    assert cancelled == [9030]


def test_wrong_account_cannot_be_cancelled():
    bad = row(8951, 'RUNNING'); bad['account'] = 'another_account'
    cancelled = []
    result = cleanup_all([8951], accounting=lambda _: [bad],
                         strict_cancel=cancelled.append, persist=lambda _: None)
    assert not cancelled and 'error' in result[0]


def test_pending_receipt_still_uses_strict_native_identity_cancellation():
    waiting = row(9030, 'PENDING'); waiting['tres'] = {}
    cancelled = []
    result = cleanup_all([9030], accounting=lambda _: [waiting],
                         strict_cancel=cancelled.append, persist=lambda _: None)
    assert cancelled == [9030] and not result[0]['verified_terminal']


def test_persistence_failure_does_not_skip_other_owned_receipts():
    cancelled = []
    def broken(_): raise OSError('Evidence storage unavailable')
    result = cleanup_all([8951, 9030], accounting=lambda ids: [row(ids[0], 'RUNNING')],
                         strict_cancel=cancelled.append, persist=broken)
    assert cancelled == [8951, 9030]
    assert all('persist_error' in r for r in result)
