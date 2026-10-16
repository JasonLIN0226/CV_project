#!/usr/bin/env python3
"""One fixed two-hour CPU diagnostic of already trained GPU readers.

No encoder inference, fit, training, labels, sealed data, or learned policy.
Missing annotation alignment remains missing in the final feasibility decision.
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import torch


class BudgetStop(RuntimeError):pass


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def locked(control,expected):
    path=control/'source_lock.json'
    if path.is_symlink() or sha(path)!=expected:raise ValueError('Diagnostic source lock differs')
    for name,want in json.loads(path.read_text())['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe source entry')
        file=control/relative
        if any(p.is_symlink() for p in (file,*file.parents)) or sha(file)!=want:
            raise ValueError('Diagnostic source/input bytes differ: '+name)


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def writer(path,fields):
    file=path.open('w',newline='');out=csv.DictWriter(file,fieldnames=fields);out.writeheader()
    return file,out


def fixed_draws(text,n):
    seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')
    return np.random.default_rng(seed).random(n)


def execute(control,expected,output):
    began=time.monotonic()
    if (os.getenv('SLURM_JOB_ACCOUNT')!='csci_ga_2271-2026fa'
            or os.getenv('SLURM_JOB_PARTITION')!='n2c48m24'
            or not os.getenv('SLURM_JOB_ID') or not socket.gethostname().startswith('b-')):
        raise ValueError('Actual allocated course CPU diagnostics required')
    actual=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j',os.environ['SLURM_JOB_ID'],
        '-n','-P','--format=JobIDRaw,Account,Partition,AllocTRES,Timelimit'],text=True)
    resource=actual.strip().split('|')
    tres=dict(v.split('=',1) for v in resource[3].split(',') if '=' in v)
    if (resource[:3]!=[os.environ['SLURM_JOB_ID'],'csci_ga_2271-2026fa','n2c48m24']
            or tres.get('gres/gpu','0')!='0' or tres.get('mem') not in ('16G','16384M')
            or resource[4]!='02:00:00' or os.getenv('SLURM_CPUS_PER_TASK')!='4'):
        raise ValueError('Actual diagnostic CPU/memory/GPU/time resources differ')
    locked(control,expected)
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('Fresh regular output required; no repeated evaluation overwrite')
    output.mkdir(parents=True)
    sys.path[:0]=[str(control/'src'),str(control/'scripts')]
    from phase1.v5_real_reader import (CausalReader,context_payload,pad_payloads,full_feature_prediction)
    from phase1.v5_real_training import (load_verified_features,prepared_episodes,query_rows,
                                         verify_external_cache_audit,HORIZONS,POLICIES)
    from phase1.v5_real_diagnostics import (completion_proof,load_frozen_fit,visual_health_batch,
        visual_health_summary,participant_values,paired_participant_interval,pooled_seed_participants,
        sampled_states,joint_oracle_cost,IrreversibleReplay,frozen_context_losses)
    from v5_retention import HistoryState,insert
    from v5_deletion_continuation import common_reservoir_continuations
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if hasattr(os,'sched_getaffinity'):
        cores=sorted(os.sched_getaffinity(0))
        if len(cores)<4:raise ValueError('Actual allocated four CPU cores required')
        os.sched_setaffinity(0,cores[:4])
    protocol=json.loads((control/'protocol.json').read_text())
    if (protocol['status']!='ACTUAL_ALL6_MODEL_HASH_BOUND_FROZEN_REAL_DIAGNOSTICS'
            or protocol['training_job_id']!=7251 or protocol['seeds']!=[17,29,43]
            or protocol['horizons']!=[1,4,16,64] or protocol['capacities']!=[8,16,32]
            or protocol['budget']!={'cpu_threads':4,'memory_gb':16,'hours':2,'soft_stop_seconds':6900,'preflight_states':8}
            or protocol['annotation_frame_phase_proved'] is not False):
        raise ValueError('Fixed diagnostic/configuration/information boundary changed')
    training=Path(protocol['training_root']);cache=Path(protocol['encoding_root'])
    if (str(training)!='/scratch/xl4358/phase1_v6_real_wd020/job_7251'
            or str(cache)!='/scratch/xl4358/phase1_v5_real_encoding_frame/job_7187'
            or protocol['training_control']!='/scratch/xl4358/phase1_jepa_memory_v6_real_wd020'
            or protocol['encoding_control']!='/scratch/xl4358/phase1_jepa_memory_v5_encoding_frame'):
        raise ValueError('Actual model/cache/source paths differ')
    report={'status':'RUNNING_FROZEN_REAL_DIAGNOSTICS_NOT_PHASE1_GO','job_id':os.environ['SLURM_JOB_ID'],
        'course':os.environ['SLURM_JOB_ACCOUNT'],'source_lock_sha256':expected,
        'protocol_sha256':sha(control/'protocol.json'),'stages_completed':[],
        'labels_read':False,'sealed_data_read':False,'fit_or_training':False,'phase1_go':False,
        'semantic_components':'UNAVAILABLE_ANNOTATION_FRAME_PHASE_UNPROVED'}
    opened=[]

    def persist():
        report['elapsed_seconds']=time.monotonic()-began
        for file in opened:
            if not file.closed:file.flush()
        write_json(output/'diagnostic_report.json',report)

    def deadline():
        if time.monotonic()-began>=6900:raise BudgetStop('Original two-hour diagnostic budget exhausted')

    def interrupted(signum,_):
        report.update(status='INTERRUPTED_INCOMPLETE_NOT_PHASE1_GO',signal=signum);persist();raise SystemExit(143)

    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    persist()
    try:
        parent=subprocess.check_output(['/opt/slurm/bin/sacct','-X','-j','7251','-n','-P',
            '--format=JobIDRaw,State,ExitCode,Account,Partition,AllocTRES'],text=True)
        row=parent.strip().split('|')
        if row[:5]!=['7251','COMPLETED','0:0','csci_ga_2271-2026fa','c12m85-a100-1'] or 'gres/gpu=1' not in row[5].split(','):
            raise ValueError('Actual GPU reader parent not COMPLETED0:0 with one GPU')
        training_control=Path(protocol['training_control'])
        locked(training_control,'24116c56a15ead3f47d5109be0a389eab179f01fbc63ba96fb69e9813bf3b5e5')
        if sha(training/'training_report.json')!=protocol['actual_training_report_sha256']:
            raise ValueError('Actual completed training report differs')
        model_proof=completion_proof(training,protocol['actual_checkpoint_files'],7251,.20)
        write_json(output/'all_models_completion.json',model_proof)
        original=json.loads((training_control/'protocol.json').read_text())
        verify_external_cache_audit(original,cache,7187)
        if sha(training/'fit_proof.json')!=protocol['actual_fit_proof_sha256']:
            raise ValueError('Actual fit metadata differs')
        fit_proof=json.loads((training/'fit_proof.json').read_text())
        if (fit_proof['fit_split']!='train' or fit_proof['normalization_and_input_rows']!=34816
                or fit_proof['future_residual_rows']!=20672 or fit_proof['basis_rank']!=128
                or fit_proof['validation_or_test_fit_rows']!=0):
            raise ValueError('Actual train-only fit scope differs')
        fit=load_frozen_fit(training/'train_only_fit.npz',fit_proof['actual_npz_sha256'])
        episodes,cache_proof=load_verified_features(cache,Path(protocol['encoding_control']),7187,'cpu')
        episodes=prepared_episodes(episodes,fit)
        valid=query_rows(episodes,'validation')
        if len(valid)!=2356 or len(episodes)!=198:raise ValueError('Actual complete diagnostic scope differs')
        expected_queries=[(str(episodes[ei]['participant']),episodes[ei]['recording'],episodes[ei]['episode'],t) for ei,t in valid]
        if len({v[0] for v in expected_queries})!=10:raise ValueError('Actual validation participant count differs')
        models={}
        for seed in (17,29,43):
            model=CausalReader()
            state=torch.load(training/f'seed_{seed}'/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True);models[seed]=model.eval()
        report.update(all6_current_models_verified_before_inference=True,
                      all_models_completion_sha256=sha(output/'all_models_completion.json'),
                      actual_cache_audit_sha256=original['actual_cache_audit_sha256'],
                      actual_fit_sha256=fit_proof['actual_npz_sha256'])

        # A: every original validation query, condition and horizon, all seeds.
        fields=['seed','split','participant','recording','episode','time','policy','horizon',
                'mse','persistence_mse','basis_oracle_mse']
        file,out=writer(output/'health_scalar_records.csv',fields);opened.append(file);health=[]
        for seed,model in models.items():
            for policy in POLICIES:
                for start in range(0,len(valid),64):
                    deadline();batch=tuple((ei,t,policy) for ei,t in valid[start:start+64])
                    errors=visual_health_batch(model,episodes,batch,fit)
                    for bi,(ei,t,_) in enumerate(batch):
                        e=episodes[ei]
                        for hi,h in enumerate(HORIZONS):
                            row={'seed':seed,'split':e['split'],'participant':e['participant'],
                                'recording':e['recording'],'episode':e['episode'],'time':t,
                                'policy':policy,'horizon':h,**{k:float(v[bi,hi]) for k,v in errors.items()}}
                            out.writerow(row);health.append(row)
                report['progress']={'stage':'A','seed':seed,'condition_completed':policy,
                                    'health_scalar_rows':len(health)}
                persist();print(json.dumps(report['progress']),flush=True)
        if len(health)!=113088:raise ValueError('Incomplete original health scalar count')
        summary=visual_health_summary(health,expected_queries);write_json(output/'visual_health.json',summary)
        # Fixed old-trial anchors from already executed scalars; no old-model re-inference.
        reference=Path('/scratch/xl4358/phase1_v5_real_diagnostic_results/job_7226/health_scalar_records.csv')
        if reference.is_symlink() or sha(reference)!='d158c869a132f1db35fbdc2491802c6899a4770dfc13a0338c0facca30eeed2c':
            raise ValueError('Completed old visual scalar reference changed')
        with reference.open() as handle:
            previous=list(csv.DictReader(handle))
        if len(previous)!=113088:raise ValueError('Incomplete old reference scalars')
        def scalar_key(row):
            return (int(row['seed']),row['recording'],int(row['episode']),int(row['time']),row['policy'],int(row['horizon']))
        old={scalar_key(row):row for row in previous}
        if len(old)!=len(previous) or set(old)!={scalar_key(row) for row in health}:
            raise ValueError('Old/new query/policy/horizon identity differs')
        comparisons={}
        for h in HORIZONS:
            for reference_policy in ('recent','full'):
                by_seed={}
                for seed in (17,29,43):
                    delta=[]
                    for row in health:
                        if row['seed']==seed and row['policy']=='full' and row['horizon']==h:
                            key=list(scalar_key(row));key[4]=reference_policy
                            value={**row,'old_mse_minus_new_full_mse':float(old[tuple(key)]['mse'])-row['mse']}
                            delta.append(value)
                    by_seed[seed]=participant_values(delta,['old_mse_minus_new_full_mse'])
                checks={str(seed):paired_participant_interval(values) for seed,values in by_seed.items()}
                checks['pooled']=paired_participant_interval(pooled_seed_participants(by_seed))
                comparisons[f'old_{reference_policy}_vs_new_full_h{h}']=checks
        primary=[check['supported'][0] for key,group in comparisons.items() if key.endswith(('h16','h64')) for check in group.values()]
        write_json(output/'fixed_old_reference_comparison.json',{'old_training_job':7209,'old_evaluation_job':7226,
            'old_raw_sha256':sha(reference),'no_old_model_reinference':True,'same_development_participants_not_confirmation':True,
            'checks':comparisons,'additional_current_candidate_adoption_status':'SUPPORTED' if all(primary) else 'REJECTED',
            'primary_checks_passed':sum(primary),'primary_checks_required':16,'original_AB_visual_gates_unchanged':True})

        report['stages_completed'].append('A_all3_visual_health');report['visual_history_status']=summary['visual_history_status'];persist()

        def context_losses(model,cases):
            return frozen_context_losses(model,episodes,cases,fit,deadline)

        # B: exactly four states per original recording and source trajectory.
        recordings={}
        for ei,e in enumerate(episodes):recordings.setdefault(e['recording'],[]).append(ei)
        states=[]
        for recording,indices in sorted(recordings.items()):
            for policy in ('FIFO','Reservoir'):
                contexts={episodes[ei]['episode']:episodes[ei]['contexts'][policy] for ei in indices}
                by_episode={episodes[ei]['episode']:ei for ei in indices}
                for episode,t in sampled_states(recording,policy,contexts):states.append((by_episode[episode],t,policy))
        if len(states)!=624:raise ValueError('Original all78 two-source four-state count differs')
        bfields=['seed','split','participant','recording','episode','time','source_policy','deleted_id',
                 'oracle_deleted_id','FIFO_deleted_id','Reservoir_deleted_id','tie_count','joint_cost']
        for stem in ('immediate','continuation','joint'):
            bfields += [f'{stem}_h{h}' for h in HORIZONS]
        file,bout=writer(output/'deletion_candidate_records.csv',bfields);opened.append(file)
        delta=[];b_started=time.monotonic()
        for si,(ei,t,source_policy) in enumerate(states):
            deadline();e=episodes[ei];retained=e['contexts'][source_policy][t-1][1]
            history=HistoryState(16,tuple(retained),t-8,t-9);incoming=t-8
            draws=fixed_draws(f'v5-deletion-common-0|{e["recording"]}|{e["episode"]}|{t}|{source_policy}',33)
            paths=common_reservoir_continuations(history,incoming,draws[1:],t)
            candidates=tuple(deleted for deleted,path in paths)
            _,reservoir_deleted=insert(history,incoming,'Reservoir',float(draws[0]))
            fifo_deleted=min(candidates)
            cases=[]
            for deleted,path in paths:
                for offset,snapshot in enumerate(path):
                    clock=t+offset;recent=tuple(range(clock-7,clock+1))
                    payload=context_payload(e['compressed'],clock,(recent,snapshot.retained))
                    cases.append((ei,clock,payload))
            for seed,model in models.items():
                losses=context_losses(model,cases).reshape(17,33,4)
                oracle=joint_oracle_cost(candidates,losses[:,0],losses[:,1:])
                costs=np.array(oracle['per_horizon_costs']);selected=oracle['chosen_index']
                for ci,deleted in enumerate(candidates):
                    row={'seed':seed,'split':e['split'],'participant':e['participant'],'recording':e['recording'],
                        'episode':e['episode'],'time':t,'source_policy':source_policy,'deleted_id':deleted,
                        'oracle_deleted_id':oracle['deleted_id'],'FIFO_deleted_id':fifo_deleted,
                        'Reservoir_deleted_id':reservoir_deleted,'tie_count':oracle['tie_count'],'joint_cost':oracle['joint_costs'][ci]}
                    for hi,h in enumerate(HORIZONS):
                        row.update({f'immediate_h{h}':float(losses[ci,0,hi]),
                            f'continuation_h{h}':float(losses[ci,1:,hi].mean()),f'joint_h{h}':float(costs[ci,hi])})
                    bout.writerow(row)
                for baseline,deleted in [('FIFO',fifo_deleted),('Reservoir',reservoir_deleted)]:
                    values=costs[candidates.index(deleted)]-costs[selected]
                    delta.append({'seed':seed,'split':e['split'],'participant':e['participant'],
                        'recording':e['recording'],'episode':e['episode'],'source_policy':source_policy,
                        'baseline':baseline,'joint':float(values.mean()),'h16':float(values[2]),'h64':float(values[3])})
            if si==7:
                projection=time.monotonic()-b_started
                total_projection=(b_started-began)+projection*len(states)/8
                report['first8_complete_all3_deletion_states_projected_seconds']=total_projection
                persist()
                if total_projection>6900:raise BudgetStop('Fixed first8 deletion states projection exceeds original budget')
            if (si+1)%8==0:
                report['progress']={'stage':'B','states_completed_all3seeds':si+1,'planned_states':624}
                persist();print(json.dumps(report['progress']),flush=True)
        deletion_checks={};b_pass=True
        for source_policy in ('FIFO','Reservoir'):
            for baseline in ('FIFO','Reservoir'):
                each={s:participant_values([v for v in delta if v['seed']==s and v['split']=='validation'
                    and v['source_policy']==source_policy and v['baseline']==baseline],['joint','h16','h64']) for s in models}
                checked={str(s):paired_participant_interval(each[s]) for s in models}
                checked['pooled']=paired_participant_interval(pooled_seed_participants(each))
                deletion_checks[source_policy+'_'+baseline]=checked
                b_pass &= all(all(result['supported']) for result in checked.values())
        write_json(output/'deletion_summary.json',{'visual_joint_and_h16_h64_status':'SUPPORTED' if b_pass else 'REJECTED',
            'checks':deletion_checks,'states':624,'candidates_per_state':17,'continuation_steps':32,
            'same_joint_oracle_choice_for_all_horizons':True,'oracle_not_deployable':True,
            'verb_NLL_unavailable':True,'phase1_go':False})
        report['stages_completed'].append('B_all17_common32_visual_deletion');persist()

        # C: persistent content records only; no deleted visual archive access.
        cfields=['seed','split','participant','recording','episode','time','policy','capacity','horizon',
                 'mse','unstratified_predictive_change','full_memory_updates']
        file,cout=writer(output/'replay_scalar_records.csv',cfields);opened.append(file)
        file,trace=writer(output/'irreversible_replay_trace.csv',['seed','recording','episode','policy','capacity',
            'time','incoming','deleted','retained_ids','recent_ids','persistent_visual_records']);opened.append(file)
        aggregates={};c_count=0
        for ei,e in enumerate(episodes):
            for capacity in (8,16,32):
                for policy in ('FIFO','Reservoir','Uniform'):
                    deadline();online=IrreversibleReplay(policy,capacity)
                    draws=fixed_draws(f'v5-real-retention-0|{e["recording"]}|{e["episode"]}|{policy}',248)
                    cases=[];full_updates={};discarded=set()
                    for t in range(256):
                        deleted=online.advance(e['compressed'][t],float(draws[t-8]) if t>=8 else .5)
                        if deleted is not None:discarded.add(deleted)
                        visual,clock,ids=online.payload()
                        if discarded.intersection(ids):raise ValueError('Deleted observation re-entered online state')
                        trace.writerow({'seed':'shared_retention_0','recording':e['recording'],'episode':e['episode'],
                            'policy':policy,'capacity':capacity,'time':t,'incoming':t-8 if t>=8 else '',
                            'deleted':deleted if deleted is not None else '',
                            'retained_ids':' '.join(map(str,online.state.retained)),
                            'recent_ids':' '.join(str(i) for i,p in online.recent),'persistent_visual_records':len(ids)})
                        if t in range(40,192,4):cases.append((ei,t,(visual,clock)));full_updates[t]=online.full_updates
                    # One online trajectory is shared by all three frozen models.
                    for seed,model in models.items():
                        losses=context_losses(model,cases)
                        # No activity labels: semantic drift/response/recovery is
                        # explicitly unavailable, not replaced by a quiet predictor.
                        for bi,(index,t,payload) in enumerate(cases):
                            for hi,h in enumerate(HORIZONS):
                                row={'seed':seed,'split':e['split'],'participant':e['participant'],
                                    'recording':e['recording'],'episode':e['episode'],'time':t,
                                    'policy':policy,'capacity':capacity,'horizon':h,'mse':float(losses[bi,hi]),
                                    'unstratified_predictive_change':'UNAVAILABLE_NOT_SUBSTITUTE_FOR_ACTIVITY_METRIC',
                                    'full_memory_updates':full_updates[t]}
                                cout.writerow(row);c_count+=1
                                key=(seed,e['split'],policy,capacity,h,str(e['participant']),e['recording'],e['episode'])
                                aggregates.setdefault(key,[]).append(row['mse'])
                    persist()
            report['progress']={'stage':'C','episodes_completed_all3seeds_all3capacities_all3policies':ei+1,
                                'planned_episodes':198,'replay_scalar_rows':c_count}
            persist()
            if (ei+1)%8==0:print(json.dumps(report['progress']),flush=True)
        if c_count!=3*198*3*3*38*4:raise ValueError('Incomplete fixed irreversible replay scalar count')
        by_group=defaultdict(list)
        for (seed,split,policy,capacity,h,person,recording,episode),values in aggregates.items():
            by_group[(seed,split,policy,capacity,h)].append({'participant':person,'recording':recording,
                'episode':episode,'mse':float(np.mean(values))})
        csummary={}
        for key,records in by_group.items():
            values=participant_values(records,['mse'])
            csummary['|'.join(map(str,key))]={person:float(value[0]) for person,value in values.items()}
        write_json(output/'replay_participant_summary.json',{'participant_means':csummary,'scalar_rows':c_count,
            'chronology_and_persistent_content_no_reentry_verified':True,'main_capacity':16,'secondary_capacities':[8,32],
            'activity_strata_task_NLL_and_corruption_recovery':'UNAVAILABLE_ANNOTATION_FRAME_PHASE_UNPROVED',
            'no_fabricated_activity_donor_or_semantic_success':True,'phase1_go':False})
        report['stages_completed'].append('C_all_episode_irreversible_visual_replay')
        report.update(status='COMPLETE_AVAILABLE_VISUAL_DIAGNOSTICS_SEMANTIC_COMPONENTS_INCOMPLETE_NOT_PHASE1_GO',
            real_feasibility_status='REJECTED' if summary['visual_history_status']=='REJECTED' or not b_pass else 'INCONCLUSIVE',
            health_scalar_rows=len(health),deletion_candidate_rows=3*624*17,replay_scalar_rows=c_count)
        # Only source/numerical provenance is repeated; no second model inference.
        if completion_proof(training,protocol['actual_checkpoint_files'],7251,.20)!=model_proof:
            raise ValueError('Actual current model proof changed during diagnostics')
        locked(control,expected)
        if sha(training/'train_only_fit.npz')!=fit_proof['actual_npz_sha256']:
            raise ValueError('Fixed train-only basis changed during diagnostics')
        for item in cache_proof['actual_feature_files']:
            path=Path(item['path'])
            if path.is_symlink() or sha(path)!=item['sha256']:
                raise ValueError('Frozen actual feature bytes changed during diagnostics')
        report['actual_current_models_sources_cache_basis_before_after_unchanged']=True
        persist()
        for file in opened:file.close()
        report['files']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in output.iterdir()
                         if p.is_file() and p.name!='diagnostic_report.json'}
        persist();print(json.dumps({'status':report['status'],'real_feasibility_status':report['real_feasibility_status'],
            'output':str(output),'report_sha256':sha(output/'diagnostic_report.json')}),flush=True)
        return 0
    except BudgetStop as error:
        report.update(status='STOPPED_ORIGINAL_DIAGNOSTIC_BUDGET_INCOMPLETE',error=str(error),
                      real_feasibility_status='REJECTED' if report.get('visual_history_status')=='REJECTED' else 'INCONCLUSIVE')
        persist();return 2
    except Exception as error:
        report.update(status='FAILED_SOURCE_MODEL_OR_NUMERICAL_BOUNDARY_NO_COMPLETE_DIAGNOSTICS',
                      error_type=type(error).__name__,error=str(error)[:300]);persist();raise
    finally:
        for file in opened:
            if not file.closed:file.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True)
    p.add_argument('--source-lock-sha',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.exit(execute(args.control,args.source_lock_sha,args.output))
