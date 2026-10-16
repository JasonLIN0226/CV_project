import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from phase1.v5_real_training_cpu import (HORIZONS, POLICIES, QUERY_STEPS, balanced_rows,
                                    full_batch_loss, make_batch, prepared_episodes,
                                    query_rows, training_fit, validate_cache_report)
from phase1.v5_real_reader import CausalReader


def episodes():
    torch.manual_seed(4)
    return [{'recording': s, 'participant': s, 'episode': 0, 'split': s,
             'features': torch.randn(256, 8)} for s in ('train', 'validation')]


def test_exact_query_grid_all_horizons_and_balanced_epoch_coverage():
    es = episodes()
    rows = query_rows(es, 'train')
    assert len(rows) == 38 and len(QUERY_STEPS) == 38
    assert all(t + max(HORIZONS) < 256 for _, t in rows)
    toy = tuple((0, i) for i in range(32))
    batches = list(balanced_rows(list(reversed(range(32))), toy))
    assert len(batches) == 2 and all(len(b) == 64 for b in batches)
    for p in POLICIES:
        assert sorted((ei, t) for b in batches for ei, t, policy in b if policy == p) == list(toy)
    with pytest.raises(ValueError):
        list(balanced_rows([0] * 32, toy))


def test_validation_poisoning_cannot_affect_train_fit_or_fit_ids():
    es = episodes()
    train = query_rows(es, 'train')
    a, pa = training_fit(es, train, rank=4)
    changed = copy.deepcopy(es)
    changed[1]['features'][:] = 1e8
    b, pb = training_fit(changed, train, rank=4)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert pa == pb
    assert pa['normalization_and_input_rows'] == 256
    assert pa['future_residual_rows'] == 38 * 4
    with pytest.raises(ValueError):
        training_fit(es, query_rows(es, 'validation'), rank=4)


def test_dropout_preserves_recent_current_clock_and_ignores_targets():
    es = episodes()
    fit, _ = training_fit(es, query_rows(es, 'train'), rank=4)
    prepared = prepared_episodes(es, fit)
    rows = ((0, 40, 'full'), (0, 40, 'recent'))
    a = make_batch(prepared, rows, True, np.random.default_rng(9))
    # 8 recent samples survive history dropout; current has zero age and role1.
    assert (~a[2][1]).sum() == 8
    for i in range(2):
        last = (~a[2][i]).sum().item() - 1
        torch.testing.assert_close(a[1][i, last], torch.tensor([0., 1.]))
    prepared[0]['normalized'][41:] += 10000
    b = make_batch(prepared, rows, True, np.random.default_rng(9))
    for x, y in zip(a[:4], b[:4]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert not torch.equal(a[4], b[4])


def test_literal_full_feature_four_horizon_loss_and_gradients():
    es = episodes()
    fit, _ = training_fit(es, query_rows(es, 'train'), rank=4)
    prepared = prepared_episodes(es, fit)
    model = CausalReader(rank=4, hidden=16, layers=2, heads=4, ffn=32, dropout=0)
    rows = tuple((0, 40, p) for p in POLICIES)
    batch = make_batch(prepared, rows, False, None)
    loss, slices = full_batch_loss(model, batch, fit)
    assert slices.shape == (4, 4)
    torch.testing.assert_close(loss, slices.mean())
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_partial_scientific_cache_cannot_become_training_input():
    with pytest.raises(ValueError, match='incomplete'):
        validate_cache_report({'status': 'RUNNING_FROZEN_SINGLE_FRAME_ENCODING'}, {}, 7004)
    with pytest.raises(ValueError, match='incomplete'):
        validate_cache_report({'status': 'ALL78_FROZEN_CAUSAL_FEATURES_COMPLETE_NOT_PHASE1_GO',
                               'job_id': '7004', 'frames_encoded': 512}, {}, 7004)



def test_missing_actual_cache_audit_never_authorizes_fit(tmp_path):
    from phase1.v5_real_training_cpu import verify_external_cache_audit
    with pytest.raises(ValueError):verify_external_cache_audit({},tmp_path,7004)
    path=tmp_path/'full_cache_audit.json';path.write_text('{}')
    with pytest.raises(ValueError):verify_external_cache_audit({},tmp_path,7004)


@pytest.mark.parametrize('wrong',('encoding_job_id','complete','labels_read','source_lock_sha256'))
def test_wrong_external_attestation_refused_even_with_matching_file_hash(tmp_path,wrong):
    import json,hashlib
    from phase1.v5_real_training_cpu import verify_external_cache_audit
    proof={'status':'ACTUAL_FULL_CACHE_VERIFIED_NOT_READER_TRAINING_NOT_PHASE1_GO','complete':True,'encoding_job_id':7004,
        'source_lock_sha256':'1'*64,
        'encoding_source_lock_sha256':'2'*64,'encoding_report_sha256':'3'*64,
        'encoding_device':'cpu','allocated_gpu_count':0,'encoding_partition':'n2c48m24',
        'reader_fit_or_training':False,'labels_read':False,'sealed_data_read':False,'frames':50688,'recordings':78,'episodes':198}
    path=tmp_path/'full_cache_audit.json';path.write_text(json.dumps(proof))
    p={'actual_cache_audit_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'cache_audit_source_lock_sha256':'1'*64,'encoding_source_lock_sha256':'2'*64,
       'actual_encoding_report_sha256':'3'*64}
    assert verify_external_cache_audit(p,tmp_path,7004)==proof
    proof[wrong]='wrong';path.write_text(json.dumps(proof));p['actual_cache_audit_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):verify_external_cache_audit(p,tmp_path,7004)

def test_cuda_cache_or_changed_gpu_allocation_cannot_authorize_cpu_fit(tmp_path):
    import json, hashlib
    from phase1.v5_real_training_cpu import verify_external_cache_audit
    proof = {'status':'ACTUAL_FULL_CACHE_VERIFIED_NOT_READER_TRAINING_NOT_PHASE1_GO',
             'complete':True,'encoding_job_id':910001,'source_lock_sha256':'1'*64,
             'encoding_source_lock_sha256':'2'*64,'encoding_report_sha256':'3'*64,
             'encoding_device':'NVIDIA L4','allocated_gpu_count':1,
             'encoding_partition':'g2-standard-12',
             'reader_fit_or_training':False,'labels_read':False,'sealed_data_read':False,
             'frames':50688,'recordings':78,'episodes':198}
    path = tmp_path/'full_cache_audit.json'
    path.write_text(json.dumps(proof))
    p = {'actual_cache_audit_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'cache_audit_source_lock_sha256':'1'*64,'encoding_source_lock_sha256':'2'*64,
         'actual_encoding_report_sha256':'3'*64}
    with pytest.raises(ValueError, match='incompatible'):
        verify_external_cache_audit(p,tmp_path,910001)


def test_cpu_reader_scientific_functions_unchanged():
    import inspect
    from phase1 import v5_real_training_cpu as cpu, v5_real_training_l4 as frozen
    for name in ('query_rows','training_fit','prepared_episodes','balanced_rows','make_batch',
                 'full_batch_loss','validation_scores','schedule_lr','validate_cache_report'):
        assert inspect.getsource(getattr(cpu,name)) == inspect.getsource(getattr(frozen,name))


def test_slurm_cpu_source_attestation_rejects_gpu_or_incomplete_parent():
    from v5_audit_cpu_cache import validate_cpu_slurm
    okay = '910001|csci_ga_2271-2026fa|n2c48m24|COMPLETED|0:0|cpu=24,mem=16G,node=1|cpu=4,mem=16G,node=1|\n'
    assert validate_cpu_slurm(okay,910001)[0]=='910001'
    bad = (
        okay.replace('COMPLETED','FAILED'),
        okay.replace('0:0','1:0'),
        okay.replace('n2c48m24','g2-standard-12'),
        okay.replace('cpu=24,','gres/gpu=1,cpu=24,'),
        okay.replace('cpu=4,','gres/gpu:l4=1,cpu=4,'),
        okay.replace('cpu=4,','cpu=8,'),
        okay.replace('16G','32G'),
        okay+okay)
    for row in bad:
        with pytest.raises(ValueError):
            validate_cpu_slurm(row,910001)

