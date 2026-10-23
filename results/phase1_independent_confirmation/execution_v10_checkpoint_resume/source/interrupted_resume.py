"""Native CPU-only inheritance of accepted V9 data, fits and exact partial state."""
from pathlib import Path
from phase1.utils import read_json, write_json, file_hash
from budget_guard import native_allocations
import shutil


def initialize(root, protocol, c):
    root = Path(root)
    lineage = protocol['interruption_resume']
    old = Path(lineage['origin_root'])
    audit = read_json(root/'interruption_native_audit.json')
    assert file_hash(root/'interruption_native_audit.json') == lineage['native_audit_sha256']
    assert audit['conservative_ledger']['settled'] == lineage['execution_settled_GPU_seconds'] == 42185
    assert audit['conservative_ledger']['cap'] == 69299
    assert file_hash(old/'source_lock.json') == lineage['origin_source_lock_sha256']
    for name, sha in read_json(old/'source_lock.json')['files'].items():
        assert file_hash(old/name) == sha, name
    actual = native_allocations([8949, 8950, 8951, 9029, 9030])
    expected = {row['job']: row for row in audit['terminal_rows']}
    for row in actual:
        for field in ('state', 'exitcode', 'elapsed', 'start', 'end', 'attempt_count'):
            assert row[field] == expected[row['job']][field], (row['job'], field)
    assert read_json(old/'config.json') == c
    data = Path(c['output'])
    assert data == old/'fresh'
    assert not (data/'all_models_lock.json').exists()
    assert not (data/'confirmation_test_generation_attempt.json').exists()
    m = read_json(data/'dataset/manifest.json')
    assert set(r['split'] for r in m['records']) == {'train', 'validation'}
    budget = read_json(data/'sequence_budget.json')
    assert budget['epochs'] == {'reference/gate/17': 80}
    assert budget['encoded_frames'] == 62418 and budget['GPU_job_ids'] == [8951, 9030]
    assert file_hash(data/'models/reference/gate/seed_17/latest.pt') == audit['checkpoint']['latest_sha256']
    assert file_hash(data/'models/reference/gate/seed_17/best.pt') == audit['checkpoint']['best_sha256']
    # Recheck cache bytes against accepted native index, without forwarding or encoding.
    index = read_json(data/'features/index.json')
    count = 0
    for entry in index['episodes'].values():
        assert file_hash(data/'features'/entry['path']) == entry['sha256']
        count += entry['shape'][0]
    assert count == 62418
    for name in ('permit.json', 'preflight_cpu.json', 'preflight_gpu.json',
                 'resource_release8523.json', 'encoder_wire_metadata_CPU_check.json'):
        if (old/name).exists():
            shutil.copy2(old/name, root/name)
            assert file_hash(old/name) == file_hash(root/name)
    from staged_data import verify_permit
    verify_permit(c, root/'permit.json')
    write_json(root/'interruption_resume_CPU_proof.json', dict(
        origin_root=str(old), origin_source_lock_sha256=lineage['origin_source_lock_sha256'],
        original_permit_sha256=file_hash(root/'permit.json'), reused_committed_frames=count,
        original_CPU_fit_job=9029, original_partial_GPU_job=9030,
        original_Gate_reference17_epochs=80, actual_terminals=actual,
        all_original_sources_and_cached_feature_SHA_exact=True,
        no_model_forward_no_encoding_no_training=True, tests_unopened=True,
        no_scientific_threshold_or_sample_change=True))
