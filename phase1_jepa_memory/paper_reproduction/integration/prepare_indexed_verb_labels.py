"""Audit and materialize verb labels on official TSM keys, data only."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from indexed_verb_alignment import selected_segments,indexed_multihot

COURSE='csci_ga_2271-2026fa'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def git_blob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():raise RuntimeError('actual course data allocation required')
    s=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o',job],text=True)
    fields=dict(x.split('=',1) for x in s.split() if '=' in x)
    if fields.get('Account')!=COURSE or fields.get('JobState')!='RUNNING' or 'gres/gpu' in fields.get('AllocTRES',''):
        raise RuntimeError('course zero-GPU data stage required')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    lock=json.loads((args.root/'source_lock.json').read_text())
    for name,value in lock['files'].items():
        if (args.root/name).is_symlink() or sha(args.root/name)!=value:raise RuntimeError('label preparation source changed')
    protocol=json.loads(args.protocol.read_text());report_path=Path(protocol['data_report'])
    if sha(report_path)!=protocol['data_report_sha256']:raise RuntimeError('complete official data binding changed')
    data=json.loads(report_path.read_text());manifest_path=Path(data['manifest'])
    actual=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j',str(data['job_id']),'-n','-P','--format=JobIDRaw,State,ExitCode,Account'],text=True)
    if not any(row.split('|')[:4]==[str(data['job_id']),'COMPLETED','0:0',COURSE] for row in actual.splitlines()):
        raise RuntimeError('official indexed data parent not completed')
    if data['test_payload_deserialized'] or sha(manifest_path)!=data['manifest_sha256']:raise RuntimeError('data scope changed')
    manifest=json.loads(manifest_path.read_text())
    selected={row['video_id']+'/C10095_rgb.mp4':row for row in manifest['videos']}
    metadata=json.loads((args.root/'annotation_files_metadata.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False);quarantine=args.output/'quarantine';quarantine.mkdir()
    from huggingface_hub import hf_hub_download
    paths={};source_proofs=[]
    for file in metadata['files']:
        if not file['rfilename'].endswith('.csv'):continue
        path=Path(hf_hub_download('cvml-nus/assembly101',filename=file['rfilename'],repo_type='dataset',
                revision=metadata['revision'],local_dir=str(quarantine)))
        if path.is_symlink() or path.stat().st_size!=file['size']:raise RuntimeError('annotation file boundary differs')
        if file.get('lfs'):
            if sha(path)!=file['lfs']['sha256']:raise RuntimeError('annotation checksum mismatch')
        elif git_blob(path)!=file['blobId']:raise RuntimeError('annotation Git blob mismatch')
        paths[file['rfilename']]=path;source_proofs.append({'path':file['rfilename'],'bytes':path.stat().st_size,'sha256':sha(path)})
    with paths['annotations/fine-grained-annotations/actions.csv'].open(newline='') as f:
        catalog={int(row['action_id']):(int(row['verb_id']),int(row['noun_id'])) for row in csv.DictReader(f)}
    if set(catalog)!=set(range(1380)) or {v[0] for v in catalog.values()}!=set(range(24)):raise RuntimeError('catalog incomplete')
    intervals=defaultdict(set);source_counts=Counter()
    for name,path in paths.items():
        if Path(name).name not in ('train.csv','validation.csv','test.csv'):continue
        with path.open(newline='') as f:
            for video,start,end,action,verb,noun in selected_segments(csv.DictReader(f),selected,catalog):
                intervals[video].add((start,end,verb));source_counts[name]+=1
    if set(intervals)!=set(selected):raise RuntimeError('some author-selected recordings lack available released numeric labels')
    proofs=[];proof_by_video={row['video_id']:row for row in data['features']}
    for video,row in selected.items():
        path=manifest_path.parent/row['path'];before=sha(path);expected=proof_by_video[row['video_id']]
        if path.is_symlink() or before!=expected['sha256'] or path.stat().st_size!=expected['bytes']:raise RuntimeError('indexed features changed')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        frames=payload['clip_indices'].numpy();labels=indexed_multihot(frames,sorted(intervals[video]))
        output=args.output/(row['video_id']+'_verb24.npz')
        np.savez(output,frame_indices=frames,verb_labels=labels)
        if sha(path)!=before:raise RuntimeError('source changed during alignment')
        proofs.append({'video_id':row['video_id'],'split':row['split'],'path':str(output),
                       'bytes':output.stat().st_size,'sha256':sha(output),'samples':len(frames),
                       'foreground_samples':int(labels.any(1).sum()),'multilabel_samples':int((labels.sum(1)>1).sum()),
                       'intervals':len(intervals[video]),'feature_sha256':before})
    report={'status':'OFFICIAL_INDEXED_TSM_VERB_AXIS_AUDITED','job_id':int(job),'account':COURSE,
            'data_parent':data['job_id'],'data_report_sha256':sha(report_path),'manifest_sha256':sha(manifest_path),
            'mapping':'annotation index = official TSM JPEG index minus one; CSV intervals inclusive',
            'official_mapping_source_sha256':protocol['official_mapping_source_sha256'],
            'raw_60fps_IJEPA_alignment_proved':False,'sealed_numeric_labels_interpreted':False,
            'fit_training_inference':False,'source_annotation_files':source_proofs,'selected_source_rows':dict(source_counts),
            'recordings':len(proofs),'label_artifacts':proofs,'elapsed_seconds':time.monotonic()-started}
    (args.output/'alignment_report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','job_id','recordings','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
