"""Bounded ZIP directory metadata audit; never read archive member payloads."""
import datetime, hashlib, io, json, socket, subprocess, zipfile
from pathlib import Path

assert socket.gethostname()=='b-10-102'
active=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-j','7268','-o','%i|%T|%a|%P|%N'],text=True).strip()
assert active=='7268|RUNNING|csci_ga_2271-2026fa|n2c48m24|b-10-102'
OUT=Path('/scratch/xl4358/phase1_v6_readonly_audit/archive_directory_audit7268.json')
assert not OUT.exists()
from huggingface_hub import HfFileSystem
fs=HfFileSystem()
file=fs.open('datasets/cvml-nus/assembly101@bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967/AssemblyPoses.zip','rb',block_size=65536,cache_type='none')

class BoundedDirectoryReader:
    def __init__(self,stream):self.stream=stream;self.bytes_read=0
    def seek(self,*args):return self.stream.seek(*args)
    def tell(self):return self.stream.tell()
    def seekable(self):return True
    def read(self,n=-1):
        if n<0: n=self.stream.size-self.stream.tell()
        if n>16*1024*1024-self.bytes_read:raise RuntimeError('Directory metadata exceeds fixed16MiB audit cap')
        data=self.stream.read(n);self.bytes_read+=len(data);return data

stream=BoundedDirectoryReader(file)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'BOUNDED_OFFICIAL_ZIP_DIRECTORY_METADATA_ONLY','job_guard':active,'repo_id':'cvml-nus/assembly101','revision':'bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967','archive':'AssemblyPoses.zip','archive_size':file.size,'archive_member_payloads_read':False,'model_inference_fit_training':False,'numeric_labels_or_sealed_observations_read':False,'max_directory_read_bytes':16*1024*1024}
try:
    with zipfile.ZipFile(stream) as archive:
        entries=[{'name':item.filename,'bytes':item.file_size,'compressed_bytes':item.compress_size} for item in archive.infolist()]
    directory=Path('/scratch/xl4358/phase1_v6_readonly_audit/archive_directory_entries7268.json')
    directory.write_text(json.dumps(entries,indent=2)+'\n')
    result.update(entries=len(entries),directory_metadata_sha256=hashlib.sha256(directory.read_bytes()).hexdigest(),directory_metadata_path=str(directory),interesting_metadata_names=[item for item in entries if any(k in item['name'].lower() for k in ('readme','timestamp','sync','frame','camera','calib','meta'))][:100])
except Exception as error:
    result.update(status='BOUNDED_DIRECTORY_AUDIT_UNAVAILABLE',error_type=type(error).__name__,error=str(error)[:250])
finally:
    result['actual_directory_bytes_read']=stream.bytes_read;file.close()
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'summary':str(OUT),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'metadata_bytes':stream.bytes_read,'archive_member_payloads_read':False}))
