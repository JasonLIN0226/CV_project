import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_path=Path(__file__).resolve().parents[1]/'scripts/v5_audit_l4_cache.py'
_spec=importlib.util.spec_from_file_location('v5_cache_audit',_path)
audit=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(audit)


def test_episode_clock_digest_matches_encoder_serialization():
    h=hashlib.sha256()
    for i in range(512):
        n,d=(i,2) if i%2 else (i//2,1)
        h.update(f'{i//256}:{i%256}:{n}/{d}\n'.encode())
    episodes=[{'start_seconds':[0,1]},{'start_seconds':[128,1]}]
    assert audit.selected_digest(episodes)==h.hexdigest()
    changed=copy.deepcopy(episodes);changed[1]['start_seconds']=[129,1]
    assert audit.selected_digest(changed)!=h.hexdigest()


def artificial_cache(tmp_path,monkeypatch,failure=None):
    control=tmp_path/'audit';control.mkdir()
    enc=tmp_path/'encoding_source';enc.mkdir()
    cache=tmp_path/'job_7004';cache.mkdir();(cache/'features').mkdir()
    weights={'artificial_weight':{'sha256':'1'*64,'bytes':8}}
    manifest={'development_rgb':[]};records=[];pts={'videos':[]}
    for index in range(78):
        split='train' if index<58 else 'validation'
        local=index if index<58 else index-58
        episodes=3 if index<20 else 2 if index<58 else 4 if local<2 else 3
        descriptors=[{'start_seconds':[128*i,1]} for i in range(episodes)]
        person=str(local//2) if split=='train' else str(100+local//2)
        row={'recording':str(index),'participant':person,'split':split,'path':f'rgb/{index}.mp4','sha256':'2'*64}
        manifest['development_rgb'].append(row)
        payload=cache/'features'/f'recording_{index:03d}.npy';payload.write_bytes(b'artificial bytes')
        timestamp=audit.selected_digest(descriptors)
        f={'path':payload.name,'shape':[episodes*256,5120],'dtype':'float32','forwards':episodes*128,
           'sha256':audit.sha(payload),'timestamp_sha256':timestamp,'bytes':payload.stat().st_size}
        records.append({**{k:row[k] for k in ('recording','participant','split')},'rgb_path':row['path'],
                        'rgb_sha256':row['sha256'],'episodes':descriptors,'feature':f})
        pts['videos'].append({'path':row['path'],'episodes':descriptors,'selected_timestamps_sha256':timestamp})
    ep={'encoder':{'pooling':'grid2','dtype':'float32','revision':'fixed-revision','repository':'artificial-repo','cache_dir':'not accessed'}}
    for filename,value in [('protocol.json',ep),('manifest.json',manifest),('pts_report.json',pts)]:
        (enc/filename).write_text(json.dumps(value))
    (enc/'source_lock.json').write_text(json.dumps({'files':{name:audit.sha(enc/name) for name in ('protocol.json','manifest.json','pts_report.json')}}))
    metadata={'pooling':'grid2','dtype':'float32','resolved_revision':'fixed-revision','name':'artificial-repo'}
    (cache/'encoder_metadata.json').write_text(json.dumps(metadata))
    report={'status':'ALL78_FROZEN_CAUSAL_FEATURES_COMPLETE_NOT_PHASE1_GO','job_id':'7004',
        'frames_encoded':50688,'forwards':25344,'planned_frames':50688,'preflight_passed':True,
        'course':'csci_ga_2271-2026fa','device':'NVIDIA L4','runtime':{'cpu_threads':4},
        'new_weights_trained':False,'predictor_trained':False,'task_labels_read':False,
        'sealed_rgb_or_numeric_labels_read':False,'completed_recordings':records,
        'actual_official_checkpoint_files':weights,'encoder_metadata_sha256':audit.sha(cache/'encoder_metadata.json')}
    for key,file in [('source_lock_sha256','source_lock.json'),('protocol_sha256','protocol.json'),('manifest_sha256','manifest.json'),('pts_report_sha256','pts_report.json')]:
        report[key]=audit.sha(enc/file)
    if failure=='partial':report['frames_encoded']=512
    if failure=='future_phase':report['completed_recordings'][0]['feature']['timestamp_sha256']='3'*64
    if failure=='label_leak':report['task_labels_read']=True
    if failure=='feature_bytes':(cache/'features'/'recording_000.npy').write_bytes(b'corrupt')
    (cache/'encoding_report.json').write_text(json.dumps(report))
    p={'encoding_job_id':7004,'encoding_control':str(enc),'encoding_root':str(cache),
       'encoding_source_lock_sha256':audit.sha(enc/'source_lock.json'),'encoding_protocol_sha256':audit.sha(enc/'protocol.json'),
       'output':str(tmp_path/'result.json')}
    (control/'protocol.json').write_text(json.dumps(p))
    (control/'source_lock.json').write_text(json.dumps({'files':{'protocol.json':audit.sha(control/'protocol.json')}}))
    monkeypatch.setenv('SLURM_JOB_ACCOUNT','csci_ga_2271-2026fa');monkeypatch.setenv('SLURM_JOB_ID','99999')
    monkeypatch.setattr(audit.socket,'gethostname',lambda:'b-artificial')
    state='CANCELLED' if failure=='slurm' else 'COMPLETED';gpu=2 if failure=='two_gpu' else 1
    monkeypatch.setattr(audit.subprocess,'check_output',lambda *a,**kw:f'7004|csci_ga_2271-2026fa|g2-standard-12|{state}|0:0|cpu=12,mem=32G,gres/gpu={gpu}|\n')
    import phase1.v5_encoding_l4 as encoding
    monkeypatch.setattr(encoding,'checkpoint_files',lambda *a:{'changed':True} if failure=='weights' else weights)
    def artificial_array(path,*,mmap_mode,allow_pickle):
        assert mmap_mode=='r' and allow_pickle is False
        index=int(Path(path).stem.split('_')[1]);n=records[index]['feature']['shape'][0]
        if failure=='nonfinite':return np.broadcast_to(np.float32(np.nan),(n,5120))
        return np.broadcast_to(np.float32(0),(n,5120))
    monkeypatch.setattr(np,'load',artificial_array)
    return control,audit.sha(control/'source_lock.json'),Path(p['output'])


def test_complete_artificial_cache_attestation_reads_no_labels_or_fits(tmp_path,monkeypatch):
    control,sha,out=artificial_cache(tmp_path,monkeypatch)
    audit.run(control,sha)
    proof=json.loads(out.read_text())
    assert proof['complete'] and len(proof['actual_feature_files'])==78
    assert not proof['reader_fit_or_training'] and not proof['phase1_go']


@pytest.mark.parametrize('failure',('partial','future_phase','label_leak','feature_bytes','slurm','two_gpu','weights','nonfinite'))
def test_false_complete_or_wrong_bindings_reject_without_publishing_proof(tmp_path,monkeypatch,failure):
    control,sha,out=artificial_cache(tmp_path,monkeypatch,failure)
    with pytest.raises(ValueError):audit.run(control,sha)
    assert not out.exists()
