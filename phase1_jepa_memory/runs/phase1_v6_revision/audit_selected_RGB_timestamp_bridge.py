"""Approved RGB container/timecode vs official timestamp metadata; no pixels/labels."""
import argparse, datetime, hashlib, json, socket, subprocess, sys
from pathlib import Path
from fractions import Fraction

p=argparse.ArgumentParser()
p.add_argument('--live-job',type=int,required=True)
p.add_argument('--live-host',required=True)
p.add_argument('--live-partition',choices=['n2c48m24','interactive'],required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
assert socket.gethostname()==args.live_host and not args.output.exists()
actual=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-j',str(args.live_job),'-o','%i|%T|%a|%P|%N'],text=True).strip()
assert actual==f'{args.live_job}|RUNNING|csci_ga_2271-2026fa|{args.live_partition}|{args.live_host}'
record='nusar-2021_action_both_9011-c13b_9011_user_id_2021-02-01_160915'
source=Path('/scratch/xl4358/phase1_jepa_memory_v5_encoding_frame')
lock=source/'source_lock.json'
def sha(path):
    path=Path(path);assert path.is_file() and not any(x.is_symlink() for x in (path,*path.parents))
    h=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
assert sha(lock)=='cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e'
for name,digest in json.loads(lock.read_text())['files'].items():assert sha(source/name)==digest
manifest=json.loads((source/'manifest.json').read_text())
spec=next(x for x in manifest['development_rgb'] if x['recording']==record)
assert spec['split']=='train' and spec['camera']=='C10095_rgb'
video=Path('/scratch/xl4358/phase1_v5_data/rgb')/spec['path']
before=sha(video);assert before==spec['sha256'] and video.stat().st_size==spec['bytes']
protocol=json.loads((source/'protocol.json').read_text())
sys.path.insert(0,protocol['decoder']['vendor'])
import av
assert av.__version__=='18.0.0'
with av.open(str(video)) as container:
    streams=[{'index':s.index,'type':s.type,'metadata':dict(s.metadata),'frames':s.frames,'time_base':str(s.time_base),'start_time':s.start_time,'duration':s.duration,'average_rate':str(getattr(s,'average_rate',None))} for s in container.streams]
    metadata=dict(container.metadata)
after=sha(video);assert after==before
timestamp=Path('/scratch/xl4358/phase1_v6_readonly_audit/selected_timestamp_member7268.json')
timestamp_sha=sha(timestamp);summary=json.loads(Path('/scratch/xl4358/phase1_v6_readonly_audit/selected_timestamp_structure7268.json').read_text());assert summary['payload_sha256']==timestamp_sha
values=json.loads(timestamp.read_text());keys=sorted(map(int,values));assert keys==list(range(21276))
first=values['0'];residuals=[values[str(k)]-first-k/60 for k in keys]
timecodes=[{'scope':'container','value':metadata['timecode']}] if 'timecode' in metadata else []
timecodes += [{'scope':f'stream_{x["index"]}_{x["type"]}','value':x['metadata']['timecode']} for x in streams if 'timecode' in x['metadata']]
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'READ_ONLY_APPROVED_RAW_RGB_TIMESTAMP_CLOCK_BRIDGE_AUDIT','job_guard':actual,'recording':record,'video_bytes':spec['bytes'],'video_before_sha256':before,'video_after_sha256':after,'timestamp_sha256':timestamp_sha,'timestamp_keys':len(keys),'timestamp_first_seconds':first,'timestamp_last_seconds':values[str(keys[-1])],'relative_60fps_residual_min_seconds':min(residuals),'relative_60fps_residual_max_seconds':max(residuals),'container_metadata':metadata,'streams':streams,'timecode_candidates':timecodes,'frame_pixels_models_labels_or_sealed_data_read':False,'exact_30fps_annotation_origin_proved':False,'label_join_selected':False,'limitation':'Capture timestamps/MP4 timecode alone do not attest which raw frame generated annotation-frame0. Need authoritative annotation extraction producer or indexed original annotated frames.'}
args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'output':str(args.output),'sha256':sha(args.output),'timecode_candidates':timecodes,'annotation_origin_proved':False}))
