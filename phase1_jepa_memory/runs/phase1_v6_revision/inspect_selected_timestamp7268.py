"""Read one approved-development timestamp metadata member, no labels/poses."""
import datetime, hashlib, json, socket, subprocess, zipfile
from pathlib import Path

assert socket.gethostname()=='b-10-102'
active=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-j','7268','-o','%i|%T|%a|%P|%N'],text=True).strip()
assert active=='7268|RUNNING|csci_ga_2271-2026fa|n2c48m24|b-10-102'
record='nusar-2021_action_both_9011-c13b_9011_user_id_2021-02-01_160915'
manifest=json.loads(Path('/scratch/xl4358/phase1_jepa_memory_v5_encoding_frame/manifest.json').read_text())
assert record+'/C10095_rgb.mp4' in json.dumps(manifest['development_rgb'])
root=Path('/scratch/xl4358/phase1_v6_readonly_audit')
out=root/'selected_timestamp_member7268.json';summary=root/'selected_timestamp_structure7268.json'
assert not out.exists() and not summary.exists()
from huggingface_hub import HfFileSystem
file=HfFileSystem().open('datasets/cvml-nus/assembly101@bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967/AssemblyPoses.zip','rb',block_size=65536,cache_type='none')
class Bounded:
    def __init__(self,f):self.f=f;self.bytes=0
    def seek(self,*args):return self.f.seek(*args)
    def tell(self):return self.f.tell()
    def seekable(self):return True
    def read(self,n=-1):
        if n<0:n=self.f.size-self.f.tell()
        assert n<=16*1024*1024-self.bytes
        data=self.f.read(n);self.bytes+=len(data);return data
stream=Bounded(file)
name='assembly101_camera_and_hand_poses/timestamp/'+record+'.json'
with zipfile.ZipFile(stream) as archive:
    item=archive.getinfo(name);assert item.file_size<1000000 and item.compress_size<300000
    payload=archive.read(item)
file.close();value=json.loads(payload);out.write_bytes(payload)
def describe(v,depth=0):
    if isinstance(v,dict):return {'type':'dict','length':len(v),'first_keys':list(v)[:12],'first_values':{k:describe(x,depth+1) if depth<2 else {'type':type(x).__name__} for k,x in list(v.items())[:5]}}
    if isinstance(v,list):return {'type':'list','length':len(v),'first_values':[describe(x,depth+1) if isinstance(x,(dict,list)) and depth<2 else x if not isinstance(x,(dict,list)) else {'type':type(x).__name__} for x in v[:8]]}
    return v
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'APPROVED_DEVELOPMENT_TIMESTAMP_MEMBER_STRUCTURE_ONLY','actual_route_guard':active,'recording':record,'official_revision':manifest['revision'],'member_name':name,'payload_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'member_payload_stays_hpc':True,'native_read_bytes':stream.bytes,'structure':describe(value),'poses_pixels_labels_or_sealed_data_read':False,'label_join_selected':False,'exact_annotation_origin_proved':False}
summary.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'summary':str(summary),'sha256':hashlib.sha256(summary.read_bytes()).hexdigest(),'structure':result['structure']}))
