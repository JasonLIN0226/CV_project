"""CPU-only audit of new controls and immutable old outputs. Never launches science."""
from pathlib import Path
import csv, datetime, hashlib, itertools, json, os, random, subprocess, sys
HERE=Path(__file__).resolve().parent
ROOT=Path('/scratch/xl4358/phase1_far_reference');UP=ROOT/'upstream'
OLD=Path('/scratch/xl4358/phase1_far_phase1');OLDA=Path('/scratch/xl4358/phase1_far_phase1_analysis')
sys.path[:0]=[str(ROOT/'vendor'),str(HERE)]
import numpy as np
from baseline_analysis import audit_trace, paired_stats, window

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def locked(base,lock,expected):
 assert sha(lock)==expected
 entries=json.loads(lock.read_text())['files']
 if isinstance(entries,list):entries={x['path']:x['sha256'] for x in entries}
 assert all(sha(base/n)==h for n,h in entries.items())
 return {'lock_sha256':expected,'files':len(entries),'all_match':True}
def native(j):
 return subprocess.check_output(['/opt/slurm/bin/sacct','-j',str(j),'-n','-P','-o','JobID,State,ExitCode,Account,Start,End,Elapsed'],text=True)
def complete(s,j):return any(x.split('|')[:3]==[str(j),'COMPLETED','0:0'] for x in s.splitlines())
def controls_trace(trace,reads,policy,targets):
 arrivals=[];previous=[];deleted=set();seen=set()
 assert len(reads)==len(targets)
 generated=0
 for row in trace:
  if not arrivals:arrivals=[row['offered']];seen.add(row['offered'])
  assert row['offered']==arrivals[-1] and row['arrived'] not in seen
  cand=previous+[row['offered']]
  expected=[] if policy=='Current_only' else cand[-16:]
  assert row['kept']==expected
  assert set(row['deleted'])==set(cand)-set(expected)
  assert not set(expected)&deleted
  if row['arrived'].startswith('g:'):
   q=reads[generated];target=int(row['arrived'][2:])
   expected_read=([row['offered']] if policy=='Current_only' else previous[-11:]+[row['offered']])
   assert q['target']==target==int(targets[generated]) and q['read']==expected_read
   assert all(int(x[2:])<target for x in q['read'])
   assert len(q['read'])==(1 if policy=='Current_only' else 12)
   generated+=1
  else:assert not generated and row['arrived'].startswith('o:')
  deleted.update(row['deleted']);seen.add(row['arrived']);arrivals.append(row['arrived']);previous=expected
 assert generated==len(targets)
 obs=[int(x[2:]) for x in arrivals if x.startswith('o:')]
 assert obs==sorted(set(obs)) and max(obs)<int(targets[0])
 return {'logical_capacity':0 if policy=='Current_only' else 16,'generated_frames':generated,'all_actual_read_IDs_causal_and_exact':True,'no_reentry_or_valid_padding':True}
def reservoir_replay(trace,seed):
 rng=random.Random(seed);past=[];seen=0
 for row in trace:
  seen+=1;incoming=row['offered']
  if len(past)<16:past.append(incoming)
  else:
   j=rng.randrange(seen)
   if j<16:past[j]=incoming
  assert row['kept']==past
 return True

def main():
 out=HERE/'analysis_output'/('job_'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
 def save(n,d):(out/n).write_text(json.dumps(d,indent=2)+'\n')
 p=json.loads((HERE/'protocol.json').read_text());proof=json.loads((HERE/'input_proof.json').read_text())
 receipt=json.loads((HERE/'evaluation_submission.json').read_text());job=receipt['job_id']
 new=HERE/'output'/('job_'+str(job));old=OLD/'output/job_7426'
 a={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_job':job,'no_model_forward_or_GPU':True,'sources':{},'sacct':{str(j):native(j) for j in [job,7426,7435]},'files':{}}
 for base,lock,h in [(HERE,HERE/'source_lock.json',receipt['source_lock_sha256']),(OLD,OLD/'source_lock.json',p['old_source_lock_sha256']),(OLDA,OLDA/'source_lock.json',p['old_analysis_lock_sha256']),(UP,ROOT/'upstream_source_lock.json',json.loads((HERE/'analysis_protocol.json').read_text())['upstream_source_lock_sha256'])]:
  a['sources'][str(lock)]=locked(base,lock,h)
 assert sha(OLD/'input_proof.json')==p['old_input_proof_sha256'] and sha(OLD/'protocol.json')==p['old_protocol_sha256']
 assert sha(HERE/'protocol.json')==proof['protocol_sha256']
 for e in proof['files']+[{'path':'models/'+x['path'],'sha256':x['sha256']} for x in proof['models']]:
  q=UP/e['path'];b=q.stat();assert sha(q)==e['sha256'];c=q.stat();assert (b.st_size,b.st_mtime_ns)==(c.st_size,c.st_mtime_ns)
 cert=Path('/scratch/xl4358/phase1_far_metadata_keys/output/job_7421/certificate.json');assert sha(cert)==proof['source_certificate_sha256']
 for e in json.loads(cert.read_text())['metrics']:assert sha(UP/e['path'])==e['sha256']
 a['current_selected_data11models_and_metric_cache_SHA_match']=True
 oldaudit=json.loads((HERE/'baseline_native_audit.json').read_text())
 for n,v in oldaudit['files'].items():assert sha(old/n)==v['sha256'],n
 a['all_old96_raw_files_unchanged_from_native7435_audit']=True
 for n in ['completion.json','failure.json','progress.json','first_clip_budget.json','executed_model_proof.json','executed_context_proof.json']:
  if (new/n).exists():a[n]=json.loads((new/n).read_text())
 if not all(complete(a['sacct'][str(j)],j) for j in [job,7426,7435]) or len(list(new.glob('*.npz')))!=48 or not (new/'completion.json').exists():
  a['status']='PARTIAL_OR_FAILED_NO_COMPLETE_RESULT';save('audit.json',a);return
 assert a['completion.json']['records']==48 and a['completion.json']['completed_steps']==3498
 assert a['executed_model_proof.json']['all_actual_loaded_floating_tensors_finite']
 assert a['executed_model_proof.json']['input_proof_sha256']==sha(HERE/'input_proof.json')
 assert a['first_clip_budget.json']['passed']
 for policy,n in [('Current_only',1),('Recent_only',12)]:
  e=a['executed_context_proof.json'][policy];assert e['expected_true_frames']==n and e['actual_context_shape'][1]==n and e['learned_retrieval_disabled'] and not e['padding_or_duplicate_history']
 rows=[];scores={};axes={};traces={}
 policies=p['old_policies']+p['policies'];keys=[m+'_'+w for m in ['dreamsim','lpips','psnr'] for w in ['tail','full']]
 for idx,steps,video in proof['selected_clips']:
  for seed in p['seeds']:
   for policy in policies:
    folder=new if policy in p['policies'] else old;n=f'{idx}_{seed}_{policy}'
    q=folder/(n+'.npz');a['files'][str(q)]={'sha256':sha(q),'bytes':q.stat().st_size}
    with np.load(q,allow_pickle=False) as r:
     assert str(r['video_id'])==video and int(r['seed'])==seed
     frames=r['frame_idx'];assert frames.shape==(steps,) and np.issubdtype(frames.dtype,np.integer) and (np.diff(frames)>0).all()
     if idx not in axes:axes[idx]=frames.copy()
     else:assert np.array_equal(axes[idx],frames)
     row={'clip_index':idx,'scene':video.split('/')[4],'route':video.split('/')[3],'seed':seed,'policy':policy,'steps':steps}
     for m in ['dreamsim','lpips','psnr']:
      assert r[m].shape==(steps,) and np.isfinite(r[m]).all()
      for tail in [False,True]:
       k=m+('_tail' if tail else '_full');row[k]=scores[idx,seed,policy,k]=window(r[m],tail)
     rows.append(row)
    trace=json.loads((folder/('trace_'+n+'.json')).read_text())
    if policy in p['policies']:traces[n]=controls_trace(trace,json.loads((folder/('reads_'+n+'.json')).read_text()),policy,axes[idx])
    else:
     traces[n]=audit_trace(trace,policy)
     if policy=='Reservoir':assert reservoir_replay(trace,seed+idx)
 assert len(rows)==144 and sum(r['steps'] for r in rows)==10494
 save('storage_and_read_audit.json',traces)
 with (out/'record_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={'status':'COMPLETE_8_SCENE_FAR_PHASE1_MEMORY_CONTROLS_NOT_ORIGINAL_GO','records':144,'target_steps':10494,'policies':{},'paired_comparisons':{},'limits':['controls added after old results were inspected; Reservoir selected from old pilot','same8scenes, not independent confirmation','Current_only removes explicit history; generated current can carry implicit past information','Current_only uses one context frame, distribution shift from usual12-frame inference','Archive has greater storage','no I-JEPA/SPACE/full-FAR reproduction claim; original GO unchanged']}
 for policy in policies:
  summary['policies'][policy]={}
  for k in keys:
   v=[float(np.mean([scores[idx,seed,policy,k] for idx,_,_ in proof['selected_clips']])) for seed in p['seeds']]
   summary['policies'][policy][k]={'mean':float(np.mean(v)),'seed_SD':float(np.std(v,ddof=1)),'seed_means':v}
 pairs=[('Reservoir','Current_only'),('Reservoir','Recent_only'),('Recent_only','Current_only'),('FIFO','Recent_only'),('FAR_current','Current_only'),('FAR_current','Recent_only'),('FAR_current','Reservoir')]
 for candidate,baseline in pairs:
  name=candidate+'_over_'+baseline;summary['paired_comparisons'][name]={}
  for k in keys:
   sign=-1 if k.startswith('psnr') else 1
   d=[float(np.mean([sign*(scores[idx,s,baseline,k]-scores[idx,s,candidate,k]) for s in p['seeds']])) for idx,_,_ in proof['selected_clips']]
   summary['paired_comparisons'][name][k]={**paired_stats(d),'scene_improvements':d}
 primary=[summary['paired_comparisons']['Reservoir_over_'+b]['dreamsim_tail'] for b in ['Current_only','Recent_only']]
 order=np.argsort([x['exact_one_sided_signflip_p'] for x in primary]);running=0
 for rank,i in enumerate(order):
  running=max(running,min(1,(2-rank)*primary[i]['exact_one_sided_signflip_p']));primary[i]['Holm_adjusted_p']=running
 summary['bounded_primary_two_control_criterion_passed']=all(x['paired_scene_bootstrap95'][0]>0 and x['Holm_adjusted_p']<.05 for x in primary)
 summary['route_means']={route:{policy:float(np.mean([r['dreamsim_tail'] for r in rows if r['route']==route and r['policy']==policy])) for policy in policies} for route in ['ABA','ABCA']}
 a['status']='COMPLETE_NATIVE_SOURCE_ASSET_AXIS_SCALAR_STORAGE_AND_CONTROL_AUDIT';a['all144_finite_paired_records']=True;a['all_old_Reservoir_Algorithm_R_replayed']=True
 save('summary.json',summary);save('audit.json',a)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  q=HERE/'analysis_output'/('job_'+os.environ.get('SLURM_JOB_ID','unknown'));q.mkdir(parents=True,exist_ok=True)
  (q/'failure.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'no_science_retry':True},indent=2)+'\n');raise
