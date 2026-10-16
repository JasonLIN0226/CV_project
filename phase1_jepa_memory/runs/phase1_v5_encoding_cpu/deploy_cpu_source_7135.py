from pathlib import Path
import hashlib,json,socket,subprocess,zipfile,stat
BASE=Path('/scratch/xl4358')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-j','7135','-o','%i|%a|%P|%T|%N'],text=True)
if '7135|csci_ga_2271-2026fa|n2c48m24|RUNNING|'+socket.gethostname() not in q: raise RuntimeError('Fresh control allocation mismatch')
archive=BASE/'phase1_v5_native_control7135/cpu_source.zip'
if sha(archive)!='5d49f012412020e24e21da5f12f436f3521522d78490eea27deb2c9d0b3f1cea': raise RuntimeError('CPU source archive SHA mismatch')
target=BASE/'phase1_jepa_memory_v5_encoding_cpu'
if target.exists(): raise RuntimeError('CPU source namespace already exists; inspect rather than replace')
with zipfile.ZipFile(archive) as z:
 names=z.namelist()
 if len(names)!=len(set(names)) or z.testzip() is not None: raise RuntimeError('Corrupt/duplicate ZIP entries')
 for i in z.infolist():
  n=Path(i.filename)
  if n.is_absolute() or '..' in n.parts or stat.S_ISLNK(i.external_attr>>16) or i.file_size>1000000: raise RuntimeError('Unsafe source ZIP entry')
 lockbytes=z.read('source_lock.json')
 if hashlib.sha256(lockbytes).hexdigest()!='01f0adc454082c050331fc7d30911814f46c8cbf4740b56697b12796dc9797df': raise RuntimeError('Source lock changed')
 lock=json.loads(lockbytes)
 if set(names)!=set(lock['files'])|{'source_lock.json','submit_v5_encoding_cpu.sh'}: raise RuntimeError('Unexpected ZIP payload')
 for rel,h in lock['files'].items():
  if hashlib.sha256(z.read(rel)).hexdigest()!=h: raise RuntimeError('Locked payload mismatch')
 if hashlib.sha256(z.read('submit_v5_encoding_cpu.sh')).hexdigest()!='a0ecae2481d237e925c9db326bee1aea3c7b84a5510d50c2042137bac71eaa57': raise RuntimeError('Batch mismatch')
 target.mkdir(mode=0o700)
 z.extractall(target)
for rel,h in lock['files'].items():
 f=target/rel
 if not f.is_file() or any(x.is_symlink() for x in (f,*f.parents)) or sha(f)!=h: raise RuntimeError('Native deployed byte mismatch')
(target/'logs').mkdir(mode=0o700)
proof={'source_lock_sha256':sha(target/'source_lock.json'),'source_files':len(lock['files']),'protocol_sha256':sha(target/'protocol.json'),'batch_sha256':sha(target/'submit_v5_encoding_cpu.sh'),'source_directory':str(target),'all_native_bytes_exact':True,'no_science_submission':True,'control_job':7135,'host':socket.gethostname()}
proofpath=BASE/'phase1_v5_native_control7135/cpu_source_deployment.json'
if proofpath.exists(): raise RuntimeError('Deployment proof already exists')
proofpath.write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
