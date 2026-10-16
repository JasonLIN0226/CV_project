import os,socket,json,hashlib,subprocess
from pathlib import Path
actual=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7243','-n','-P','--format=JobIDRaw,State,Account,Partition,NodeList,AllocTRES'],text=True).strip().split('|')
assert actual[:5]==['7243','RUNNING','csci_ga_2271-2026fa','interactive',socket.gethostname()]
assert socket.gethostname()=='b-9-9' and 'gres/gpu' not in actual[5]
root=Path('/scratch/xl4358/phase1_v6_revision_metadata_7243')
assert not root.exists();root.mkdir(mode=0o700)
manifest=json.loads(Path('/scratch/xl4358/phase1_jepa_memory_v5_encoding_frame/manifest.json').read_text())
from huggingface_hub import hf_hub_download
out={'status':'PRIMARY_RELEASE_README_ONLY_NO_LABEL_JOIN','revision':manifest['revision'],'sources':[],'numeric_labels_read':False,'model_inference_fit_training':False}
for name in ['README.md','annotations/README.md']:
 p=Path(hf_hub_download(manifest['repo_id'],name,repo_type='dataset',revision=manifest['revision'],local_dir=root/'docs'))
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<30000
 payload=p.read_bytes()
 out['sources'].append({'path':name,'sha256':hashlib.sha256(payload).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest(),'text':payload.decode()})
(root/'official_readme_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'output':str(root/'official_readme_audit.json'),'sha256':hashlib.sha256((root/'official_readme_audit.json').read_bytes()).hexdigest()}))
