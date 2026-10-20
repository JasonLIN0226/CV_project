"""Compact final evidence receiver; no model/encoder work and no new submission."""
from pathlib import Path
import csv
import time
from phase1.utils import read_json,write_json,write_csv,file_hash
from budget_guard import SequenceGuard,native_allocations,CAP


def collect(execution,c):
    execution=Path(execution);data=Path(c['output']);out=execution/'compact';out.mkdir(exist_ok=False)
    final=read_json(data/'final_outcome.json');audit=read_json(data/'independent_final_audit.json')
    if audit['status']!='PASS_INDEPENDENT_FRESH_PROTOCOL_AUDIT':raise ValueError('Independent final audit absent')
    budget=SequenceGuard(data).check()
    jobs=read_json(data/'sequence_budget.json')['GPU_job_ids']
    native=native_allocations(jobs)
    if any(r['state'] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY') for r in native):
        raise ValueError('No active/unknown GPU allocation may remain at final transfer')
    if budget['actual_GPU_seconds']>CAP:raise ValueError('Actual full cumulative budget exceeded')
    # Pair/scalar summaries only. Full raw rows, vectors, maps, weights and RGB stay HPC.
    for task in ('gate','motion'):
        ref=data/'evaluation/reference'/task/'raw_records.csv'
        cand=data/'evaluation/candidate'/task/'raw_records.csv'
        with ref.open() as f:rr=list(csv.DictReader(f))
        idx={tuple(r[k] for k in ('seed','split','condition','horizon','episode')):r for r in rr}
        grouped={seed:[] for seed in (17,29,43)}
        with cand.open() as f:
            for r in csv.DictReader(f):
                if r['condition']!='privileged_small_budget' or not r['split'].startswith('test_'):continue
                k=tuple(r[n] for n in ('seed','split','condition','horizon','episode'));rk=list(k);rk[2]='recent_only'
                grouped[int(r['seed'])].append(dict(seed=r['seed'],split=r['split'],episode=r['episode'],pair=r['pair'],
                    horizon=r['horizon'],candidate=r['normalized_mse'],reference=idx[k]['normalized_mse'],
                    reference_recent=idx[tuple(rk)]['normalized_mse']))
        for seed,rows in grouped.items():
            if len(rows)!=3072:raise ValueError('Complete compact fresh fixed-reference axes required')
            write_csv(out/f'fixed_reference_{task}_{seed}.csv',rows)
    with (data/'deletion/paired_policy_losses.csv').open() as f:rows=list(csv.DictReader(f))
    grouped={}
    for r in rows:grouped.setdefault((r['task'],r['seed'],r['pool_group']),[]).append(r)
    for (task,seed,group),rows in grouped.items():write_csv(out/f'deletion_{task}_{seed}_{group}.csv',rows)
    files=[execution/'source_lock.json',execution/'protocol.json',execution/'config.json',execution/'permit.json',
        execution/'preflight_cpu.json',execution/'preflight_gpu.json',execution/'resource_release8523.json',
        data/'final_outcome.json',data/'independent_final_audit.json',data/'sequence_budget.json',
        data/'last_budget_forecast.json',data/'observation_maps/training_crossfit.json',
        data/'observation_maps/independent_fit_audit.json',data/'observation_maps/fitted_manifest.json',
        data/'probes/frozen_readout_final_verification.json',data/'dataset/split_audit.json',
        data/'probes/metrics.json',data/'all_models_lock.json',data/'encoder_loaded_train.json',data/'encoder_loaded_test.json',
        data/'features/pair_feature_audit.json',data/'deletion/summary.json',data/'deletion/independent_raw_audit.json']
    for task in ('gate','motion'):
        files.append(data/'evaluation'/task/'comparison.json')
        for role in ('reference','candidate'):files.append(data/'evaluation'/role/task/'metrics.json')
    files.extend(sorted((execution/'receipts').glob('*.json')))
    files.extend(sorted((execution/'stage_completed').glob('*.json')))
    files.extend(sorted(out.glob('*.csv')))
    ledger={str(p.relative_to(execution)):dict(sha256=file_hash(p),bytes=p.stat().st_size) for p in files}
    if any(v['bytes']>1500000 for v in ledger.values()):
        raise ValueError('A declared compact artifact exceeds the native transfer cap; do not pull large raw data')
    write_json(execution/'final_receipt.json',dict(status='COMPLETE_FRESH_CONFIRMATION_TERMINAL_EVIDENCE',
        final_outcome=final,independent_audit_status=audit['status'],GPU_native_terminal=native,
        cumulative_GPU_seconds=budget['actual_GPU_seconds'],remaining_GPU_seconds=CAP-budget['actual_GPU_seconds'],
        compact_files=ledger,large_weights_features_vectors_and_raw_stay_HPC=True))
    print('FINAL_COMPACT_READY',flush=True)
    deadline=time.monotonic()+720
    while time.monotonic()<deadline and not (execution/'transfer_done.json').exists():time.sleep(5)
    write_json(execution/'evidence_receiver_release.json',dict(transfer_flag_received=(execution/'transfer_done.json').exists(),
        released_by_normal_exit=True,no_GPU=True))
