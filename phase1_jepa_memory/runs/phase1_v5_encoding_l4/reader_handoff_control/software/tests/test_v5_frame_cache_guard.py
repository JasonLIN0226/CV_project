"""Artificial metadata tests; no actual cache, fitting, or scientific training."""
import copy
import json
from pathlib import Path

import pytest
from phase1 import v5_real_training as training

ACCOUNT = ('7187|COMPLETED|0:0|csci_ga_2271-2026fa|c12m85-a100-1|cpu=12,gres/gpu=1,mem=60G\n'
           '7188|COMPLETED|0:0|csci_ga_2271-2026fa|n2c48m24|cpu=24,mem=16G\n')


@pytest.fixture
def binding(tmp_path, monkeypatch):
    root=tmp_path/'job_7187';root.mkdir()
    (root/'encoding_report.json').write_text('{"artificial":true}\n')
    (root/'encoder_metadata.json').write_text('{"artificial":true}\n')
    proof={
        'audit_job_id':'7188','encoding_job_id':7187,
        'status':'ACTUAL_FULL_CACHE_VERIFIED_NOT_READER_TRAINING_NOT_PHASE1_GO','complete':True,
        'source_lock_sha256':'fc02564863f8af84a79eed2bf0861b210252667c2f70411518a32a9eeb219d00',
        'protocol_sha256':'51c0b25cffb63a9f7ea2e8f3a059c4ab99cc020b976d1a400d6ac167ecf78b26',
        'encoding_source_lock_sha256':'cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e',
        'encoding_protocol_sha256':'c456f06f572df47e66e7863ca752e22214c44846ab9be9934955f31ffa79ac7c',
        'encoding_report_sha256':training.sha_file(root/'encoding_report.json'),
        'encoder_metadata_sha256':training.sha_file(root/'encoder_metadata.json'),
        'reader_fit_or_training':False,'labels_read':False,'sealed_data_read':False,
        'frames':50688,'recordings':78,'episodes':198,'actual_feature_files':[{} for _ in range(78)]}
    path=root/'full_cache_audit.json';path.write_text(json.dumps(proof))
    protocol={'actual_cache_audit_job_id':7188,'actual_cache_audit_sha256':training.sha_file(path),
              'actual_encoding_report_sha256':proof['encoding_report_sha256'],
              'actual_encoder_metadata_sha256':proof['encoder_metadata_sha256']}
    monkeypatch.setattr(training.subprocess,'check_output',lambda *args,**kwargs:ACCOUNT)
    return root,path,proof,protocol


def test_positive_actual_metadata_contract(binding):
    root,_,proof,protocol=binding
    assert training.verify_external_cache_audit(protocol,root,7187)==proof


@pytest.mark.parametrize('key,value',[
    ('complete',False),('encoding_job_id',7179),('audit_job_id','7005'),
    ('encoding_source_lock_sha256','0'*64),('source_lock_sha256','0'*64),
    ('labels_read',True),('sealed_data_read',True),('reader_fit_or_training',True),
    ('frames',512),('recordings',77),('actual_feature_files',[{}]*77)])
def test_hashed_but_incompatible_proof_rejected(binding,key,value):
    root,path,proof,protocol=binding
    altered=copy.deepcopy(proof);altered[key]=value
    path.write_text(json.dumps(altered));protocol['actual_cache_audit_sha256']=training.sha_file(path)
    with pytest.raises(ValueError,match='incomplete or incompatible'):
        training.verify_external_cache_audit(protocol,root,7187)


@pytest.mark.parametrize('name',['encoding_report.json','encoder_metadata.json','full_cache_audit.json'])
def test_changed_payload_bytes_rejected(binding,name):
    root,_,_,protocol=binding
    with (root/name).open('a') as file:file.write(' ')
    with pytest.raises(ValueError):training.verify_external_cache_audit(protocol,root,7187)


@pytest.mark.parametrize('account',[
    ACCOUNT.replace('COMPLETED','RUNNING',1),ACCOUNT.replace('0:0','2:0',1),
    ACCOUNT.replace('csci_ga_2271-2026fa','wrong-course',1),
    ACCOUNT.replace('gres/gpu=1','gres/gpu=2'),
    ACCOUNT.replace('cpu=24,mem=16G','cpu=24,mem=16G,gres/gpu=1')])
def test_actual_slurm_completion_and_resources_required(binding,monkeypatch,account):
    root,_,_,protocol=binding
    monkeypatch.setattr(training.subprocess,'check_output',lambda *args,**kwargs:account)
    with pytest.raises(ValueError):training.verify_external_cache_audit(protocol,root,7187)


def test_audit_symlink_and_unbound_protocol_rejected(binding):
    root,path,_,protocol=binding
    saved=root/'saved.json';path.rename(saved);path.symlink_to(saved)
    with pytest.raises(ValueError,match='symlink'):training.verify_external_cache_audit(protocol,root,7187)
    path.unlink();saved.rename(path)
    del protocol['actual_encoding_report_sha256']
    with pytest.raises(ValueError):training.verify_external_cache_audit(protocol,root,7187)
