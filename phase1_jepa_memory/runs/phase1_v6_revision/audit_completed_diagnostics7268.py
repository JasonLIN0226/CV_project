"""Final read-only provenance, scalar and irreversible-trace replay; no inference."""
import argparse, csv, datetime, hashlib, json, math, socket, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/scratch/xl4358'); CONTROL=ROOT/'phase1_v6_real_diagnostics_7251'
OUT=ROOT/'phase1_v6_real_diagnostic_results/job_7268'; TRAIN=ROOT/'phase1_v6_real_wd020/job_7251'
def sha(path):
    path=Path(path);assert path.is_file() and not any(p.is_symlink() for p in (path,*path.parents))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def accumulate(table,key,vals):
    vals=np.asarray(vals,dtype=np.float64);assert np.isfinite(vals).all()
    if key not in table:table[key]=[np.zeros(len(vals)),0]
    table[key][0]+=vals;table[key][1]+=1
def hierarchical(table):
    records={}
    for key,(value,n) in table.items():accumulate(records,key[:-1],value/n)
    people={}
    for key,(value,n) in records.items():accumulate(people,key[:-1],value/n)
    return {key:value/n for key,(value,n) in people.items()}
def ci(values):
    vals=np.array([values[k] for k in sorted(values)]);assert len(vals)==10
    draws=np.random.default_rng(0).integers(0,10,size=(10000,10));lo,hi=np.quantile(vals[draws].mean(axis=1),[.025,.975],axis=0)
    return {'point':vals.mean(axis=0),'lower':lo,'upper':hi,'supported':lo>0}
def main(job,host,report_sha):
    assert socket.gethostname()==host
    queue=subprocess.check_output(['/opt/slurm/bin/squeue','-h','-u','xl4358','-o','%i|%T|%a|%P|%N'],text=True)
    assert f'{job}|RUNNING|csci_ga_2271-2026fa|interactive|{host}' in queue
    assert not any('|RUNNING|' in row and not row.startswith(str(job)+'|') for row in queue.splitlines())
    slurm=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j',f'7187,7188,7209,7226,7251,7253,7268,{job}','-n','-P','--format=JobIDRaw,State,ExitCode,Account,Partition,Start,End,Elapsed,AllocTRES'],text=True)
    records={row.split('|')[0]:row.split('|') for row in slurm.splitlines()}
    assert records['7268'][:5]==['7268','COMPLETED','0:0','csci_ga_2271-2026fa','n2c48m24']
    assert records['7251'][:5]==['7251','COMPLETED','0:0','csci_ga_2271-2026fa','c12m85-a100-1']
    report_path=OUT/'diagnostic_report.json';assert len(report_sha)==64 and sha(report_path)==report_sha
    report=json.loads(report_path.read_text());assert report['status']=='COMPLETE_AVAILABLE_VISUAL_DIAGNOSTICS_SEMANTIC_COMPONENTS_INCOMPLETE_NOT_PHASE1_GO'
    assert report['stages_completed']==['A_all3_visual_health','B_all17_common32_visual_deletion','C_all_episode_irreversible_visual_replay']
    assert [report[k] for k in ('health_scalar_rows','deletion_candidate_rows','replay_scalar_rows')]==[113088,31824,812592]
    locks={'phase1_jepa_memory_v5_encoding_frame': 'cd667fc4f029660c198353570686a8ed8f069a0e2dab1434e912776f5521985e', 'phase1_v5_a100_cache_audit_7187_frame': 'fc02564863f8af84a79eed2bf0861b210252667c2f70411518a32a9eeb219d00', 'phase1_jepa_memory_v5_real_frame_training': '1e5ba4b9242469df3bc7f6fe6f03eec53a62cdfd9c47f5fdda73845ebfb5504f', 'phase1_v5_real_diagnostics_7209': '8d00977db2202c22f303badc1d3008125426224e8147b6692238ddf39625419a', 'phase1_jepa_memory_v6_real_wd020': '24116c56a15ead3f47d5109be0a389eab179f01fbc63ba96fb69e9813bf3b5e5', 'phase1_v6_diagnostic_handoff_7251': '63ace4bb0ab4adada2cd55d0618d4a6de0fde1840966b80887b8dadddd49a662', 'phase1_v6_real_diagnostics_7251': '7ba613fb2c3fb9419059a8bb8035846bc2ec74a4f5ef78b17458c7682db3e379'}
    sources=[]
    for name,want in locks.items():
        directory=ROOT/name;lock=directory/'source_lock.json';assert sha(lock)==want
        entries=json.loads(lock.read_text())['files']
        for rel,digest in entries.items():assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(directory/rel)==digest
        sources.append({'source':str(directory),'sha256':want,'files':len(entries),'current_all_match':True})
    output_proof=[]
    for name,value in report['files'].items():
        file=OUT/name;assert file.name==name and file.stat().st_size==value['bytes'] and sha(file)==value['sha256']
        output_proof.append({'name':name,**value})
    sys.path[:0]=[str(CONTROL/'src'),str(CONTROL/'scripts')]
    import torch
    torch.set_num_threads(1)
    from phase1.v5_real_diagnostics import completion_proof
    protocol=json.loads((CONTROL/'protocol.json').read_text());models=completion_proof(TRAIN,protocol['actual_checkpoint_files'],7251,.20)
    assert models==json.loads((OUT/'all_models_completion.json').read_text())==json.loads((CONTROL/'actual_all6_model_proof.json').read_text())
    audit_path=ROOT/'phase1_v5_real_encoding_frame/job_7187/full_cache_audit.json';assert sha(audit_path)=='afbbc24866cb97c7ae1f8d29e6bb440373fee890088a4e1d6ef53c9c9e3b3649'
    cache=[]
    for item in json.loads(audit_path.read_text())['actual_feature_files']:
        path=Path(item['path']);before=sha(path);size=path.stat().st_size;after=sha(path);assert before==after==item['sha256']
        cache.append({'path':str(path),'bytes':size,'before_sha256':before,'after_sha256':after})
    assert len(cache)==78 and sha(TRAIN/'train_only_fit.npz')=='d90f99e4a91d290b7d27311de2820775981afe08dc053dc642e8a40636ed1fb3'
    # Recompute joint oracle exactly from existing all17 candidate scalar records.
    groups=defaultdict(list);bpath=OUT/'deletion_candidate_records.csv';bhash=sha(bpath)
    with bpath.open() as f:
        for row in csv.DictReader(f):
            key=(int(row['seed']),row['split'],row['participant'],row['recording'],int(row['episode']),int(row['time']),row['source_policy'])
            groups[key].append(row)
    assert len(groups)==1872 and sum(map(len,groups.values()))==31824
    counts=defaultdict(int);delta={};max_math=0.
    for key,rows in groups.items():
        seed,split,person,record,episode,time,source=key;assert seed in (17,29,43) and source in ('FIFO','Reservoir') and 40<=time<=159
        counts[(seed,record,source)]+=1;rows.sort(key=lambda row:int(row['deleted_id']));ids=[int(row['deleted_id']) for row in rows];assert len(set(ids))==17
        immediate=np.array([[float(row[f'immediate_h{h}']) for h in (1,4,16,64)] for row in rows]);future=np.array([[float(row[f'continuation_h{h}']) for h in (1,4,16,64)] for row in rows]);cost=.5*immediate+.5*future
        stored=np.array([[float(row[f'joint_h{h}']) for h in (1,4,16,64)] for row in rows]);joint=np.array([float(row['joint_cost']) for row in rows]);assert np.isfinite(cost).all()
        max_math=max(max_math,float(np.max(np.abs(cost-stored))),float(np.max(np.abs(cost.mean(axis=1)-joint))));assert max_math<1e-12
        selected=min(range(17),key=lambda i:(joint[i],ids[i]));oracle=ids[selected]
        assert all(int(row['oracle_deleted_id'])==oracle and int(row['tie_count'])==int(np.sum(joint==joint[selected])) and int(row['FIFO_deleted_id'])==min(ids) for row in rows)
        for baseline in ('FIFO','Reservoir'):
            deleted=int(rows[0][baseline+'_deleted_id']);assert all(int(row[baseline+'_deleted_id'])==deleted for row in rows) and deleted in ids
            value=cost[ids.index(deleted)]-cost[selected]
            if split=='validation':accumulate(delta,(source+'_'+baseline,seed,person,record,episode),[value.mean(),value[2],value[3]])
    assert len(counts)==3*78*2 and set(counts.values())=={4}
    people=hierarchical(delta);saved_b=json.loads((OUT/'deletion_summary.json').read_text());bchecks={};max_ci=0.
    for group in ('FIFO_FIFO','FIFO_Reservoir','Reservoir_FIFO','Reservoir_Reservoir'):
        each={seed:{person:value for (g,s,person),value in people.items() if g==group and s==seed} for seed in (17,29,43)}
        pooled={person:np.mean([each[s][person] for s in (17,29,43)],axis=0) for person in each[17]};bchecks[group]={}
        for label,values in [*( (str(s),each[s]) for s in (17,29,43)),('pooled',pooled)]:
            value=ci(values);saved=saved_b['checks'][group][label]
            for ours,theirs in [('point','point'),('lower','ci95_lower'),('upper','ci95_upper')]:max_ci=max(max_ci,float(np.max(np.abs(value[ours]-np.array(saved[theirs])))))
            assert max_ci<1e-12 and list(value['supported'])==saved['supported']
            bchecks[group][label]={k:v.tolist() for k,v in value.items()}
    assert sha(bpath)==bhash
    # Stream all replay scalars; no feature/model inference and no huge file download.
    cpath=OUT/'replay_scalar_records.csv';chash=sha(cpath);agg={};cn=0;scope=defaultdict(int)
    with cpath.open() as f:
        for row in csv.DictReader(f):
            s=int(row['seed']);cap=int(row['capacity']);h=int(row['horizon']);t=int(row['time']);policy=row['policy'];assert s in (17,29,43) and cap in (8,16,32) and h in (1,4,16,64) and t in range(40,192,4) and policy in ('FIFO','Reservoir','Uniform')
            assert row['unstratified_predictive_change']=='UNAVAILABLE_NOT_SUBSTITUTE_FOR_ACTIVITY_METRIC' and int(row['full_memory_updates'])==max(0,t-7-cap)
            key=(s,row['split'],policy,cap,h,row['participant'],row['recording'],int(row['episode']));accumulate(agg,key,[float(row['mse'])]);cn+=1;scope[key]+=1
    assert cn==812592 and len(agg)==3*198*9*4 and set(scope.values())=={38}
    cp=hierarchical(agg);saved_c=json.loads((OUT/'replay_participant_summary.json').read_text());max_c=0.;compact={}
    for key,value in cp.items():
        s,split,policy,cap,h,person=key;group='|'.join(map(str,(s,split,policy,cap,h)));max_c=max(max_c,abs(float(value[0])-saved_c['participant_means'][group][person]));assert max_c<1e-12
        compact.setdefault(group,{})[person]=float(value[0])
    assert sha(cpath)==chash
    # Recreate every retention transition independently from the fixed seed recipe.
    trace_path=OUT/'irreversible_replay_trace.csv';trace_hash=sha(trace_path);current=None;seen_groups=set();tn=0;retained=[];discarded=set()
    with trace_path.open() as f:
        for row in csv.DictReader(f):
            group=(row['recording'],int(row['episode']),row['policy'],int(row['capacity']));record,episode,policy,cap=group;t=int(row['time']);assert row['seed']=='shared_retention_0'
            if group!=current:
                assert current is None or previous==255
                assert group not in seen_groups;seen_groups.add(group);current=group;previous=-1;retained=[];discarded=set()
                seed=int.from_bytes(hashlib.sha256(f'v5-real-retention-0|{record}|{episode}|{policy}'.encode()).digest()[:8],'big');draws=np.random.default_rng(seed).random(248)
            assert t==previous+1;previous=t;deleted=None
            if t>=8:
                incoming=t-8;candidates=retained+[incoming]
                if len(candidates)>cap:
                    draw=float(draws[incoming]);index=0 if policy=='FIFO' else min(int(draw*len(candidates)),len(candidates)-1) if policy=='Uniform' else min(int(draw*(incoming+1)),incoming) if int(draw*(incoming+1))<cap else cap
                    deleted=candidates[index];retained=[v for i,v in enumerate(candidates) if i!=index];discarded.add(deleted)
                else:retained=candidates
                assert row['incoming']==str(incoming)
            else:assert row['incoming']==''
            actual=[int(v) for v in row['retained_ids'].split()];recent=[int(v) for v in row['recent_ids'].split()]
            assert actual==retained and recent==list(range(max(0,t-7),t+1)) and not discarded.intersection(actual+recent)
            assert row['deleted']==('' if deleted is None else str(deleted)) and int(row['persistent_visual_records'])==len(actual)+len(recent) and len(actual)<=cap
            tn+=1
    assert previous==255 and tn==456192 and len(seen_groups)==198*9 and sha(trace_path)==trace_hash
    result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FINAL_NATIVE_SOURCE_MODEL_CACHE_ALL_SCALAR_COUNTS_AND_INDEPENDENT_DELETION_REPLAY_TRACE_AUDIT_PASS','actual_control_job':job,'actual_control_host':host,'actual_queue':queue,'actual_slurm':slurm,'executed_report_sha256':sha(report_path),'all6_current_proof_recomputed_equals_executed':True,'all6_proof':models,'all_frozen_sources':sources,'all78_cache_before_after_stable':cache,'executed_output_files_current_hashes_match':output_proof,'deletion_candidate_rows':31824,'deletion_all3_actual_states':624,'deletion_joint_math_max_difference':max_math,'deletion_all48_CI_replay_max_difference':max_ci,'deletion_checks_replayed':bchecks,'deletion_status':saved_b['visual_joint_and_h16_h64_status'],'replay_scalar_rows':cn,'replay_participant_mean_max_difference':max_c,'replay_participant_means':compact,'irreversible_retention_trace_rows':tn,'actual_retention_trajectories':len(seen_groups),'all_fixed_FIFO_Reservoir_Uniform_transitions_match_no_reentry':True,'no_new_model_inference_fit_training':True,'not_independent_delegated_review':True,'all_large_raw_files_remain_HPC':True,'semantic_components_unavailable_not_fabricated':True,'phase1_go':False}
    path=OUT/'root_final_audit_7268.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(path),'sha256':sha(path),'B_status':result['deletion_status'],'B_replay_max':max_ci,'C_replay_max':max_c,'trace_rows':tn}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control-job',type=int,required=True);p.add_argument('--control-host',required=True);p.add_argument('--expected-report-sha',required=True);args=p.parse_args();main(args.control_job,args.control_host,args.expected_report_sha)
