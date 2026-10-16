"""Independent final CPU verifier. Reads existing outputs; never launches science."""
from pathlib import Path
import csv,datetime,hashlib,json,math,os,random,subprocess,sys
ROOT=Path('/scratch/xl4358/phase1_far_reference');UP=ROOT/'upstream'
SCI=Path('/scratch/xl4358/phase1_far_phase1_controls');ANA=Path('/scratch/xl4358/phase1_far_phase1_controls_audit')
OLD=Path('/scratch/xl4358/phase1_far_phase1');OLDA=Path('/scratch/xl4358/phase1_far_phase1_analysis')
OUT=Path('/scratch/xl4358/phase1_far_controls_final');OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def frozen(base,expected):
 lock=base/'source_lock.json'
 if base==UP:lock=ROOT/'upstream_source_lock.json'
 assert sha(lock)==expected
 files=read(lock)['files']
 if isinstance(files,list):files={e['path']:e['sha256'] for e in files}
 assert all(sha(base/n)==h for n,h in files.items())
 return {'lock_sha256':expected,'files':len(files),'all_current_match':True}
def stable(p,expected=None):
 p=Path(p);a=p.stat();h=sha(p);b=p.stat();assert (a.st_size,a.st_mtime_ns)==(b.st_size,b.st_mtime_ns)
 if expected:assert h==expected,str(p)
 return {'bytes':a.st_size,'sha256':h,'mtime_ns':a.st_mtime_ns}
def stats(d):
 d=np.asarray(d,dtype=np.float64);assert d.shape==(8,)
 draw=np.random.default_rng(0).integers(8,size=(10000,8))
 weights=np.eye(8,dtype=np.int64)[draw].sum(axis=1)
 boot=weights.dot(d)/8
 null=[sum((1 if (mask>>i)&1 else -1)*d[i] for i in range(8))/8 for mask in range(256)]
 observed=math.fsum(d)/8
 return observed,np.quantile(boot,[.025,.975]).tolist(),sum(v>=observed-1e-12 for v in null)/256

def trace_check(rows,policy,seed,reads,axis):
 seen=set();deleted=set();prior=[];arrivals=[];rng=random.Random(seed);reservoir=[];n=0;generated=0
 for row in rows:
  offered=row['offered'];arrived=row['arrived'];kept=row['kept'];gone=row['deleted']
  if not arrivals:seen.add(offered);arrivals.append(offered)
  assert offered==arrivals[-1] and arrived not in seen
  candidates=set(prior)|{offered};assert len(kept)==len(set(kept))
  assert set(kept)<=candidates and set(gone)==candidates-set(kept) and not set(kept)&deleted
  assert arrived not in kept
  cap=0 if policy=='Current_only' else 16
  if policy=='Archive_reference':assert set(kept)==candidates and not gone
  else:assert len(kept)==min(cap,len(seen))
  if policy in ['FIFO','Recent_only']:assert set(kept)==set(arrivals[-16:])
  if policy=='Reservoir':
   n+=1
   if n<=16:reservoir.append(offered)
   else:
    j=rng.randrange(n)
    if j<16:reservoir[j]=offered
   assert kept==sorted(reservoir)
  if arrived.startswith('g:'):
   assert int(arrived[2:])==int(axis[generated])
   if reads is not None:
    rr=reads[generated];expected=[offered] if policy=='Current_only' else prior[-11:]+[offered]
    assert rr['target']==int(axis[generated]) and rr['read']==expected
    assert len(expected)==(1 if policy=='Current_only' else 12)
    assert all(int(x[2:])<int(axis[generated]) for x in expected)
   generated+=1
  else:assert arrived.startswith('o:') and generated==0
  deleted.update(gone);seen.add(arrived);arrivals.append(arrived);prior=kept
 obs=[int(x[2:]) for x in arrivals if x.startswith('o:')]
 assert obs==sorted(set(obs)) and max(obs)<int(axis[0]) and generated==len(axis)
 if reads is not None:assert len(reads)==len(axis)
 return {'updates':len(rows),'predicted_steps':generated,'exact_causal_capacity_and_no_reentry':True,'exact_actual_control_readIDs_checked':reads is not None,'Algorithm_R_replayed':policy=='Reservoir'}

def main():
 report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verifier_sha256':sha(__file__),'no_model_forward_or_science_submission':True,'source_groups':{},'files':{},'assets':{},'trace_audits':{}}
 jobs=[7454,7456,7426,7435]
 native=subprocess.check_output(['/opt/slurm/bin/sacct','-j',','.join(map(str,jobs)),'-n','-P','-o','JobID,State,ExitCode,Account,Start,End,Elapsed'],text=True);report['sacct']=native
 for j in jobs:assert any(x.split('|')[:4]==[str(j),'COMPLETED','0:0','csci_ga_2271-2026fa'] for x in native.splitlines()),str(j)
 p=read(SCI/'protocol.json');proof=read(SCI/'input_proof.json')
 for base,h in [(SCI,'321b69b2c8316b269661a6c571cc64bddeab09a6b067992d08562f5a977ff97e'),(ANA,'fd730fda92064c0b7a07eea47f1aa166c45b83adcc152163de3cb61bf5bff0fb'),(OLD,p['old_source_lock_sha256']),(OLDA,p['old_analysis_lock_sha256']),(UP,'f5dee7da4dc3eb14eacf2ab0595c1303cd13176e5a3bc913e3f19881846fe11f')]:report['source_groups'][str(base)]=frozen(base,h)
 assert sha(SCI/'protocol.json')=='ae901e693765f566cdedcd809bdd64fda8d91bcd7ef786f4bbf31bc93c1fc83b'==proof['protocol_sha256']
 assert sha(OLD/'input_proof.json')==p['old_input_proof_sha256'] and sha(OLD/'protocol.json')==p['old_protocol_sha256']
 for e in proof['files']:report['assets'][str(UP/e['path'])]=stable(UP/e['path'],e['sha256'])
 for e in proof['models']:report['assets'][str(UP/'models'/e['path'])]=stable(UP/'models'/e['path'],e['sha256'])
 cert=Path('/scratch/xl4358/phase1_far_metadata_keys/output/job_7421/certificate.json');assert sha(cert)==proof['source_certificate_sha256']
 for e in read(cert)['metrics']:report['assets'][str(UP/e['path'])]=stable(UP/e['path'],e['sha256'])
 data=SCI/'output/job_7454';old=OLD/'output/job_7426';an=ANA/'output/job_7456'
 assert read(data/'completion.json')['records']==48 and read(data/'completion.json')['completed_steps']==3498
 assert not (data/'failure.json').exists() and read(data/'first_clip_budget.json')['passed']
 model=read(data/'executed_model_proof.json');assert model['all_actual_loaded_floating_tensors_finite'] and model['threads']==4 and model['input_proof_sha256']==sha(SCI/'input_proof.json')
 assert len(model['tensors'])==487
 cp=read(data/'executed_context_proof.json')
 for name,length in [('Current_only',1),('Recent_only',12)]:assert cp[name]['actual_context_shape'][1]==length and cp[name]['learned_retrieval_disabled'] and not cp[name]['padding_or_duplicate_history']
 assert read(an/'audit.json')['status']=='COMPLETE_NATIVE_SOURCE_ASSET_AXIS_SCALAR_STORAGE_AND_CONTROL_AUDIT'
 summary=read(an/'summary.json');assert summary['records']==144 and summary['target_steps']==10494
 baseline=read(SCI/'baseline_native_audit.json')
 for n,v in baseline['files'].items():assert sha(old/n)==v['sha256']
 with (an/'record_summary.csv').open() as f:csvrows=list(csv.DictReader(f))
 assert len(csvrows)==144
 bykey={(int(r['clip_index']),int(r['seed']),r['policy']):r for r in csvrows};assert len(bykey)==144
 values={};axes={};policies=p['old_policies']+p['policies'];keys=['dreamsim_tail','dreamsim_full','lpips_tail','lpips_full','psnr_tail','psnr_full'];count=0;steps_total=0
 for idx,steps,video in proof['selected_clips']:
  for seed in p['seeds']:
   for policy in policies:
    stem=f'{idx}_{seed}_{policy}';root=data if policy in p['policies'] else old;q=root/(stem+'.npz');report['files'][str(q)]=stable(q)
    with np.load(q,allow_pickle=False) as z:
     frame=z['frame_idx'];assert str(z['video_id'])==video and int(z['seed'])==seed and len(frame)==steps and (np.diff(frame)>0).all()
     if idx in axes:assert np.array_equal(frame,axes[idx])
     else:axes[idx]=frame.copy()
     row=bykey[idx,seed,policy];assert int(row['steps'])==steps and row['scene']==video.split('/')[4] and row['route']==video.split('/')[3]
     for metric in ['dreamsim','lpips','psnr']:
      x=z[metric];assert x.shape==(steps,) and np.isfinite(x).all()
      for tail in [False,True]:
       key=metric+('_tail' if tail else '_full');v=float(np.mean(x[-((steps+3)//4):] if tail else x));assert math.isclose(v,float(row[key]),abs_tol=1e-12);values[idx,seed,policy,key]=v
    trace=root/('trace_'+stem+'.json');report['files'][str(trace)]=stable(trace);reads=None
    if policy in p['policies']:
     rp=root/('reads_'+stem+'.json');report['files'][str(rp)]=stable(rp);reads=read(rp)
    report['trace_audits'][stem]=trace_check(read(trace),policy,seed+idx,reads,axes[idx]);count+=1;steps_total+=steps
 assert count==144 and steps_total==10494
 for policy in policies:
  for key in keys:
   seedmeans=[math.fsum(values[idx,seed,policy,key] for idx,_,_ in proof['selected_clips'])/8 for seed in p['seeds']]
   saved=summary['policies'][policy][key];assert np.allclose(seedmeans,saved['seed_means'],rtol=0,atol=1e-12)
   assert math.isclose(math.fsum(seedmeans)/3,saved['mean'],abs_tol=1e-12)
   assert math.isclose(float(np.std(seedmeans,ddof=1)),saved['seed_SD'],abs_tol=1e-12)
 checked=0
 for name,metrics in summary['paired_comparisons'].items():
  candidate,baseline=name.split('_over_')
  for key,saved in metrics.items():
   sign=-1 if key.startswith('psnr') else 1
   d=[math.fsum(sign*(values[idx,s,baseline,key]-values[idx,s,candidate,key]) for s in p['seeds'])/3 for idx,_,_ in proof['selected_clips']]
   mean,ci,pval=stats(d)
   assert np.allclose(d,saved['scene_improvements'],rtol=0,atol=1e-12)
   assert math.isclose(mean,saved['mean_improvement'],abs_tol=1e-12)
   assert np.allclose(ci,saved['paired_scene_bootstrap95'],rtol=0,atol=1e-12)
   assert pval==saved['exact_one_sided_signflip_p'];checked+=1
 primary=[summary['paired_comparisons']['Reservoir_over_'+b]['dreamsim_tail'] for b in ['Current_only','Recent_only']]
 raw=[x['exact_one_sided_signflip_p'] for x in primary];order=sorted(range(2),key=lambda i:raw[i]);adj=[0.,0.];adj[order[0]]=min(1,2*raw[order[0]]);adj[order[1]]=max(adj[order[0]],raw[order[1]])
 for x,padj in zip(primary,adj):assert x['Holm_adjusted_p']==padj
 criterion=all(x['paired_scene_bootstrap95'][0]>0 and padj<.05 for x,padj in zip(primary,adj));assert criterion==summary['bounded_primary_two_control_criterion_passed']
 for route,entry in summary['route_means'].items():
  for policy,v in entry.items():assert math.isclose(v,math.fsum(values[idx,s,policy,'dreamsim_tail'] for idx,_,video in proof['selected_clips'] if video.split('/')[3]==route for s in p['seeds'])/12,abs_tol=1e-12)
 for f in ['audit.json','summary.json','record_summary.csv','storage_and_read_audit.json']:report['files'][str(an/f)]=stable(an/f)
 for f in ['completion.json','executed_model_proof.json','executed_context_proof.json','first_clip_budget.json']:report['files'][str(data/f)]=stable(data/f)
 # Every raw source and output must still equal the hash captured before its read.
 for n,v in report['files'].items():assert sha(n)==v['sha256'],n
 report.update({'status':'PASS_FINAL_INDEPENDENT_144_RECORD_AUDIT','records':count,'target_steps':steps_total,'source_files':sum(v['files'] for v in report['source_groups'].values()),'current_asset_files':len(report['assets']),'all_raw_metrics_CSV_and_pair_axes_match':True,'all_24_Algorithm_R_and_48_control_read_traces_pass':True,'independent_paired_statistic_checks':checked,'independent_Holm_and_bounded_criterion_match':True,'selected_loaded_floating_tensor_count':487,'whole_original_Phase1_GO_claimed':False})
 (OUT/'final_native_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['status','records','target_steps','source_files','current_asset_files','independent_paired_statistic_checks']}))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  (OUT/'failure.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'no_science_retry_or_metric_change':True},indent=2)+'\n');raise
