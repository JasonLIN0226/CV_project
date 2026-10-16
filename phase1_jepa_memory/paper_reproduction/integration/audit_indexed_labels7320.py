"""Independent inclusive-interval replay on course Cloud; no model inputs."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import os
import subprocess
import time
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    current=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job','-o','7313'],text=True)
    if 'JobState=RUNNING' not in current or 'Account=csci_ga_2271-2026fa' not in current:
        raise RuntimeError('current course allocation required')
    actual=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7320','-n','-P','--format=JobIDRaw,State,ExitCode,Account'],text=True)
    if not any(row.split('|')[:4]==['7320','COMPLETED','0:0','csci_ga_2271-2026fa'] for row in actual.splitlines()):
        raise RuntimeError('actual label data parent not completed')
    started=time.monotonic();root=Path('/scratch/xl4358/phase1_space_indexed_labels_masked/job_7320')
    report=json.loads((root/'alignment_report.json').read_text())
    data_path=Path('/scratch/xl4358/phase1_space_official_data/job_7310/data_report.json')
    if sha(data_path)!=report['data_report_sha256']:raise RuntimeError('feature provenance differs')
    data=json.loads(data_path.read_text());feature={row['video_id']:row for row in data['features']}
    targets={row['video_id']+'/C10095_rgb.mp4':row for row in report['label_artifacts']}
    if len(targets)!=294:raise RuntimeError('all feature records must remain represented')
    intervals=defaultdict(list)
    for source in report['source_annotation_files']:
        path=root/'quarantine'/source['path']
        if path.stat().st_size!=source['bytes'] or sha(path)!=source['sha256']:raise RuntimeError('annotation source changed')
        if path.name not in ('train.csv','validation.csv','test.csv'):continue
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                name=row['video']
                if name not in targets:continue
                intervals[name].append((int(row['start_frame']),int(row['end_frame']),int(row['verb_id'])))
    summary={'train':{'records':0,'labelled_records':0,'samples':0,'known_samples':0},
             'val':{'records':0,'labelled_records':0,'samples':0,'known_samples':0}}
    unknown=[]
    for name,row in targets.items():
        path=Path(row['path']);before=sha(path)
        if path.is_symlink() or before!=row['sha256'] or path.stat().st_size!=row['bytes']:raise RuntimeError('label artifact differs')
        with np.load(path,allow_pickle=False) as z:
            frames=z['frame_indices'];labels=z['verb_labels'];valid=z['label_valid']
            expected_frames=feature[row['video_id']]['first_frame_index']+15*np.arange(row['samples'],dtype=np.int64)
            if not np.array_equal(frames,expected_frames) or labels.dtype!=np.uint8 or valid.dtype!=np.bool_ or labels.shape!=(len(frames),24):
                raise RuntimeError('actual indexed label shape, mask, or axis differs')
            expected=np.zeros_like(labels)
            known=name in intervals
            if known:
                for start,end,verb in intervals[name]:
                    lo=np.searchsorted(frames,start+1,side='left')
                    hi=np.searchsorted(frames,end+1,side='right')
                    expected[lo:hi,verb]=1
                if not valid.all():raise RuntimeError('known recording was masked')
            else:
                if row['split']!='train' or valid.any():raise RuntimeError('unknown validation labels or false supervision')
                unknown.append(row['video_id'])
            if not np.array_equal(labels,expected):raise RuntimeError('independent interval assignment disagrees')
            s=summary[row['split']];s['records']+=1;s['labelled_records']+=int(known)
            s['samples']+=len(frames);s['known_samples']+=int(valid.sum())
        if sha(path)!=before:raise RuntimeError('labels changed during audit')
    if summary['train']['records']!=219 or summary['train']['labelled_records']!=218 or summary['val']['records']!=75 or summary['val']['labelled_records']!=75:
        raise RuntimeError('coverage differs from predeclared data-only policy')
    if unknown!=['nusar-2021_action_both_9025-c06b_9025_user_id_2021-02-18_103624']:
        raise RuntimeError('unknown recording changed')
    output={'status':'ALL294_INDEXED_LABEL_ARTIFACTS_AND_MASKS_REPLAYED_EXACTLY','job_id':7320,
            'alignment_report_sha256':sha(root/'alignment_report.json'),'data_report_sha256':sha(data_path),
            'counts':summary,'unknown_training_recordings':unknown,'all75_validation_recordings_have_labels':True,
            'independent_method':'searchsorted assignment of official JPEG [start+1,end+1], not producer event sweep',
            'model_fit_training_inference':False,'sealed_numeric_labels_read':False,
            'raw60fps_IJEPA_alignment_claim':False,'elapsed_seconds':time.monotonic()-started}
    path=root/'root_independent_label_axis_audit.json';path.write_text(json.dumps(output,sort_keys=True,indent=2)+'\n')
    print(json.dumps(output));print('sha256',sha(path))


if __name__=='__main__':main()
