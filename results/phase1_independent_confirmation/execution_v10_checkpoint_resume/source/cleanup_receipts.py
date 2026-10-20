"""Prepared failure cleanup: isolate each owned receipt and retain all errors.

Not installed into V9. The caller must supply its frozen, source-checked GPU
receipt IDs. Active cancellation still uses the original strict identity check.
"""
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'PREEMPTED',
            'NODE_FAIL', 'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def cleanup_all(jobs, *, accounting, strict_cancel, persist):
    outcomes = []
    for job in jobs:
        result = {'job': job, 'verified_terminal': False}
        try:
            if not isinstance(job, int) or isinstance(job, bool) or job <= 0:
                raise ValueError('Actual integer receipt ID required')
            # native_allocations restricts queries to the current user/course/date,
            # reconciles duplicate attempts, and rejects incomplete evidence.
            rows = accounting([job])
            if len(rows) != 1 or rows[0]['job'] != job:
                raise ValueError('Native receipt identity mismatch')
            row = rows[0]
            gpu = row['tres'].get('gres/gpu')
            waiting = row['state'].split('+')[0] in ('PENDING', 'CONFIGURING') and gpu is None
            if row['account'] != 'csci_ga_2271-2026fa' or not (gpu == '1' or waiting):
                raise ValueError('Refuse an unrelated/non-single-GPU receipt')
            result['observed_state'] = row['state']
            if row['state'].split('+')[0] in TERMINAL:
                result['verified_terminal'] = True
                result['action'] = 'already_terminal'
            else:
                # Fresh original cancel() checks current user/account/partition.
                strict_cancel(job)
                result['action'] = 'cancel_requested_requires_terminal_recheck'
        except Exception as exc:
            result['error'] = type(exc).__name__ + ': ' + str(exc)
        outcomes.append(result)
        try:
            persist(list(outcomes))
        except Exception as exc:
            # Evidence filesystem failure must not prevent the next owned cancellation.
            result['persist_error'] = type(exc).__name__ + ': ' + str(exc)
    return outcomes
