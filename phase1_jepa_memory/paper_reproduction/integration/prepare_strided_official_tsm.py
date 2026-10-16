"""Bitwise-equivalent author conversion using the verified original archive."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from strided_tsm_reader import read_recording

COURSE='csci_ga_2271-2026fa'
PYTHON='/scratch/xl4358/phase1_env/bin/python'
DATA=Path('/scratch/xl4358/phase1_space_official_data')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    job=os.environ.get('SLURM_JOB_ID','');started=time.monotonic()
    if not job.isdigit() or sys.executable!=PYTHON:raise RuntimeError('course batch and approved Python required')
    raw=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',job],text=True)
    fields=dict(x.split('=',1) for x in raw.split() if '=' in x)
    if fields.get('Account')!=COURSE or fields.get('JobState')!='RUNNING' or 'gres/gpu' in fields.get('AllocTRES',''):
        raise RuntimeError('course zero-GPU data allocation required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    lock=json.loads((args.root/'source_lock.json').read_text())
    for name,value in lock['files'].items():
        if (args.root/name).is_symlink() or sha(args.root/name)!=value:raise RuntimeError('source changed')
    protocol=json.loads((args.root/'data_protocol.json').read_text())
    old=DATA/'job_7305/archive_integrity.json'
    if sha(old)!=protocol['original_archive_audit_sha256']:raise RuntimeError('initial verified acquisition proof changed')
    ledger=json.loads(old.read_text());archive=DATA/'download/TSM_features/C10095_rgb.zip'
    if archive.stat().st_size!=33956379008 or sha(archive)!='de9e50b081e2dd7798d76356b0828375e1a995de9715babebcf3b06d3155b821':
        raise RuntimeError('official archive changed')
    lmdb_root=DATA/'lmdb/C10095_rgb';lmdb_file=lmdb_root/'data.mdb'
    member=next(x for x in ledger['members'] if x['name'].endswith('/data.mdb'))
    if lmdb_file.is_symlink() or lmdb_file.stat().st_size!=member['bytes']:raise RuntimeError('extracted LMDB boundary differs')
    lmdb_stat=lmdb_file.stat();lmdb_hash=sha(lmdb_file)
    if lmdb_file.stat().st_mtime_ns!=lmdb_stat.st_mtime_ns:raise RuntimeError('LMDB changed while hashing')
    write(args.output/'reused_acquisition_integrity.json',{
        'original_data_job':7305,'original_archive_audit_sha256':sha(old),'archive_sha256':ledger['sha256'],
        'lmdb_bytes':lmdb_stat.st_size,'lmdb_sha256':lmdb_hash,
        'no_archive_redownload':True,'original_upstream_source_unchanged':True,
        'equivalence_proof_sha256':protocol['equivalence_proof_sha256']})
    spec=importlib.util.spec_from_file_location('author_prepare',args.root/'upstream/scripts/prepare_assembly101_tsm.py')
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    timings=[]
    def guarded_read(env,recording_id,frame_stride):
        if time.monotonic()-started>3300:raise RuntimeError('one-hour preparation soft budget reached')
        tick=time.monotonic();result=read_recording(env,recording_id,frame_stride)
        timings.append({'recording':recording_id,'seconds':time.monotonic()-tick,'frames':len(result[0])})
        if len(timings)==3:
            projection=(time.monotonic()-started)+max(x['seconds'] for x in timings)*298
            write(args.output/'first_three_budget.json',{'timings':timings,'remaining_records_upper_projection_seconds':projection,
                    'budget_seconds':3300,'passed':projection<=3300,'warm_IO_may_affect_projection':True})
            if projection>3300:raise RuntimeError('fixed full-data projection exceeds one-hour budget')
        return result
    author.read_recording=guarded_read
    target=DATA/f'latents_strided_job_{job}'
    if target.exists():raise RuntimeError('refuse ambiguous preexisting prepared outputs')
    sys.argv=['prepare_assembly101_tsm.py','--lmdb',str(lmdb_root),'--output-dir',str(target),'--splits','train','val']
    author.main()
    manifest=json.loads((target/'manifest.json').read_text())
    if manifest['test_features_accessed'] or set(manifest['requested_splits'])!={'train','val'}:
        raise RuntimeError('test isolation failed')
    if len(manifest['videos'])+len(manifest['dropped'])!=298:raise RuntimeError('full author-selected record set required')
    participants={'train':set(),'val':set()};proofs=[]
    for row in manifest['videos']:
        file=target/row['path'];before=sha(file);payload=torch.load(file,map_location='cpu',weights_only=True)
        features,indices=payload['latents'],payload['clip_indices']
        if features.dtype!=torch.float32 or tuple(features.shape)!=(row['valid_length'],2048) or not torch.isfinite(features).all():
            raise RuntimeError('feature shape/dtype/finite mismatch')
        if not torch.allclose(features.norm(dim=1),torch.ones(len(features)),atol=1e-5):raise RuntimeError('normalization mismatch')
        if indices.dtype!=torch.int64 or not bool((indices[1:]-indices[:-1]==15).all()):raise RuntimeError('index cadence mismatch')
        if not torch.equal(indices.float()/30.,payload['timestamps_sec']):raise RuntimeError('author timestamp mismatch')
        pid=payload['metadata']['participant_id'];participants[row['split']].add(pid)
        if sha(file)!=before:raise RuntimeError('prepared feature changed')
        proofs.append({'video_id':row['video_id'],'split':row['split'],'participant':pid,'frames':len(features),
                       'first_frame_index':int(indices[0]),'last_frame_index':int(indices[-1]),
                       'bytes':file.stat().st_size,'sha256':before})
    if len(participants['train'])!=29 or len(participants['val'])!=10 or participants['train']&participants['val']:
        raise RuntimeError('participant split mismatch')
    if lmdb_file.stat().st_size!=lmdb_stat.st_size or lmdb_file.stat().st_mtime_ns!=lmdb_stat.st_mtime_ns:
        raise RuntimeError('source LMDB changed during conversion')
    report={'job_id':int(job),'account':COURSE,'partition':fields['Partition'],'allocated_tres':fields.get('AllocTRES'),
            'status':'OFFICIAL_TSM_TRAIN_VAL_PREPARED_AND_AUDITED','model_inference_fit_training':False,
            'test_payload_deserialized':False,'manifest':str(target/'manifest.json'),
            'manifest_sha256':sha(target/'manifest.json'),'counts':dict(Counter(x['split'] for x in proofs)),
            'participants':{k:sorted(v) for k,v in participants.items()},'features':proofs,
            'data_source_lock_sha256':sha(args.root/'source_lock.json'),'data_protocol_sha256':sha(args.root/'data_protocol.json'),
            'original_archive_audit_sha256':sha(old),'lmdb_sha256':lmdb_hash,
            'conversion_equivalence_proof_sha256':protocol['equivalence_proof_sha256'],
            'conversion_only_changed_value_IO':True,'annotation_join_proved':False,'elapsed_seconds':time.monotonic()-started}
    write(args.output/'data_report.json',report)
    write(args.output/'conversion_timings.json',{'timings':timings,'total_seconds':sum(x['seconds'] for x in timings)})
    print(json.dumps({k:report[k] for k in ('status','counts','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
