"""Build a budget-only amendment after native settlement; old sources stay intact."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent.parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def once(text,old,new):
    if text.count(old)!=1: raise ValueError('Nonunique amendment boundary: '+old[:70])
    return text.replace(old,new)


def main():
    settle_path=ROOT/'native_final7529/native_budget_stop.json'
    settled=json.loads(settle_path.read_text())
    assert settled['GPU_cap_seconds']==18000 and settled['jobs']['7524']['state']=='FAILED'
    assert len(settled['partials'])==4 and settled['complete_models']==0
    used=settled['GPU_used_seconds'];remaining=28800-used
    auth=json.loads((ROOT/'budget_authorization_8h.json').read_text())
    assert auth['approved'] and auth['cumulative_GPU_cap_seconds']==28800
    out=ROOT/'approved_8h';control=out/'control';audit=out/'independent_audit'
    if out.exists():raise ValueError('Refuse overwrite of the next frozen execution')
    control.mkdir(parents=True);audit.mkdir()
    old=ROOT/'provisional_control'
    oldlock=json.loads((old/'source_lock.json').read_text())
    assert sha(old/'source_lock.json')=='9ab65cd2e422085672d753a9b8bb1ff462861271cfbb42899ce54c116e838ab4'
    for rel,expected in oldlock['files'].items():
        assert sha(old/rel)==expected
        p=control/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/rel,p)
    oldroot='/scratch/xl4358/phase1_predictor_repair/repair_execution'
    newroot='/scratch/xl4358/phase1_predictor_repair/approved_8h'
    prior={int(k):v for k,v in settled['GPU_ledger'].items()}
    declaration={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'kind':'DIRECT_HUMAN_EXTRA3H_BUDGET_ONLY_EXACT_EPOCH_RESUME',
      'authorization_sha256':sha(ROOT/'budget_authorization_8h.json'),
      'prior_source_lock_sha256':sha(old/'source_lock.json'),
      'native_partial_proof_sha256':sha(settle_path),
      'actual_prior_GPU_seconds':used,'cumulative_cap_seconds':28800,
      'remaining_seconds':remaining,'old_GPU_budget_stop_preserved':True,
      'same_model_optimizer_loss_precision_data_seeds_epochs_selection_and_gates':True,
      'input_packing_or_other_new_acceleration_adopted':False,
      'resume':'Only original Gate17/Motion17 epoch1 actual model/AdamW/scheduler/RNG state; new signature explicitly binds the old-source epoch and original checkpoint SHA. Other four models start from their original seed. No old signature is relabeled as executed by new code.'}
    write(control/'budget_resume_declaration.json',declaration)
    write(control/'budget_authorization_8h.json',auth)
    write(control/'native_partial_origin.json',settled)
    p=json.loads((control/'protocol.json').read_text())
    p.update(status='FROZEN_SAME_REPAIR_HUMAN8H_BUDGET_AND_EXACT_EPOCH_RESUME',
             budget_resume_declaration_sha256=sha(control/'budget_resume_declaration.json'),
             native_partial_origin_sha256=sha(control/'native_partial_origin.json'))
    p['resume']={'old_control':oldroot+'/control','old_training_root':oldroot+'/training/job_7524',
                 'old_source_lock_sha256':sha(old/'source_lock.json'),
                 'partials':settled['partials'], 'native_final_proof':oldroot+'/native_budget_stop.json',
                 'native_final_proof_sha256':sha(settle_path)}
    p['resources'].update(cumulative_GPU_cap_seconds=28800,prior_actual_GPU_seconds=used,
                          remaining_shared_GPU_seconds=remaining,new_job_wall_seconds=remaining-60,
                          soft_remaining_seconds=remaining-180)
    p['human_extra_3h_authorization_sha256']=sha(control/'budget_authorization_8h.json')
    p['frozen_utc']=declaration['UTC']
    approval=json.loads((control/'authorization.json').read_text())
    approval.update(GPU_total_cap_seconds=28800,prior_actual_GPU_seconds=used,remaining_GPU_seconds=remaining,
                    extra3h_direct_human_authorization=auth)
    write(control/'authorization.json',approval)
    guard=control/'src/phase1/resource_guard.py'
    text=guard.read_text().replace('CAP = 18000','CAP = 28800')
    text=once(text,'PRIOR_GPU_SECONDS = {7477: 100, 7480: 126, 7485: 6021}',f'PRIOR_GPU_SECONDS = {prior!r}')
    guard.write_text(text)
    train=control/'src/phase1/repair_trial.py';text=train.read_text()
    text=text.replace("!= 6247",f"!= {used}").replace("!= 11753",f"!= {remaining}").replace('!= 18000','!= 28800')
    text=text.replace(' + 11753 - elapsed_from_allocation',f' + {remaining} - elapsed_from_allocation')
    text=once(text,"'optimizer': 'AdamW_lr0.0003_wd0.0001_clip1_warmup5_cosine120'}", "'optimizer': 'AdamW_lr0.0003_wd0.0001_clip1_warmup5_cosine120',\n            **({'epoch_execution_provenance': {'epoch1': protocol['resume']['partials'][task+'/latest'],\n                  'epoch1_original_job': 7524, 'epoch1_original_source_lock_sha256': protocol['resume']['old_source_lock_sha256'],\n                  'epochs2_to120_source_lock_sha256': file_hash(Path(control)/'source_lock.json')}}\n               if protocol.get('resume') and seed == 17 else {})}")
    text=once(text,"logs, best_score, step, seconds, initial_epoch = [], float('inf'), 0, 0., 0", "logs, best_score, step, seconds, initial_epoch = [], float('inf'), 0, 0., 0\n        resumed_from_original = False")
    old_else="""        else:
            model_root.mkdir(parents=True)
        def snapshot(epoch):"""
    new_else="""        else:
            model_root.mkdir(parents=True)
            if seed == 17:
                origin = p['resume']
                prior_path = Path(origin['old_training_root']) / 'predictor' / task / 'seed_17' / 'latest.pt'
                if file_hash(prior_path) != origin['partials'][task+'/latest']['sha256']:
                    raise ValueError('Original actual partial checkpoint changed')
                ck = torch.load(prior_path, map_location='cpu', weights_only=False)
                expected = signature(origin['old_control'], read_json(Path(origin['old_control'])/'protocol.json'),
                                     cfg, store, task, seed, reference_sha)
                if ck['signature'] != expected or ck['epoch'] != 1 or ck['step'] != 64:
                    raise ValueError('Original epoch1 provenance/boundary differs')
                if not finite_tree(ck['model']) or not finite_tree(ck['optimizer']):
                    raise ValueError('Original actual partial is nonfinite')
                model.load_state_dict(ck['model'], strict=True)
                opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
                logs, best_score, step, seconds, initial_epoch = ck['logs'], ck['best'], ck['step'], ck['seconds'], 1
                restore_rng(ck['rng']); resumed_from_original = True
        def snapshot(epoch):"""
    text=once(text,old_else,new_else)
    text=once(text,"        if initial_epoch == 0:\n            save_torch(model_root / 'latest.pt', snapshot(0))", """        if resumed_from_original:
            # New provenance container, same actual old-source tensors/state/logs.
            save_torch(model_root / 'latest.pt', snapshot(1))
            save_torch(model_root / 'best.pt', snapshot(1))
            write_csv(model_root / 'training.csv', logs)
            write_json(model_root / 'training.json', {'signature':sig,'logs':logs,'seed':seed})
        elif initial_epoch == 0:
            save_torch(model_root / 'latest.pt', snapshot(0))
        elapsed = logs[-1]['epoch_seconds'] if logs else 0.
        completed_counts[task][seed] = initial_epoch""")
    text=once(text,"            resources = measure_remaining_sequence_resources(contexts, root, stop, deadline)","""            old_resource_path = Path(p['resume']['old_training_root']) / 'native_resource_probe.json'
            expected_resource_sha = p['resume']['resource_probe_sha256']
            if file_hash(old_resource_path) != expected_resource_sha:
                raise ValueError('Original actual complete-sequence cost proof changed')
            resources = read_json(old_resource_path)
            write_json(root / 'native_resource_probe.json', {**resources,
                       'inherited_actual_cost_measurement_from_job':7524,
                       'original_native_resource_probe_sha256':expected_resource_sha})""")
    text=text.replace('Actual reconciled shared GPU budget proof required','Actual reconciled eight-hour shared GPU budget proof required')
    text=text.replace('Original human shared18000s budget and actual6247s ledger required','Direct human cumulative28800s budget and complete actual prior ledger required')
    train.write_text(text)
    # Native full original-input preflight is repeated under this new source lock;
    # no old proof is relabeled as a new execution.
    pre=control/'native_preflight.py';text=pre.read_text().replace(oldroot,newroot)
    text=text.replace("'7477,7480,7485'","'7477,7480,7485,7524'")
    pre.write_text(text)
    # Only path fields change in evaluator and independent audit runners.
    for rel in ('repair_train.sbatch','repair_evaluate.sbatch','preflight.sbatch'):
        f=control/rel;t=f.read_text().replace(oldroot,newroot)
        if rel=='repair_train.sbatch':
            seconds=remaining-60;limit=f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'
            t=t.replace('--time=03:15:00','--time='+limit)
        f.write_text(t)
    for rel in ('run_audit.py','verify_scalar_results.py','audit.sbatch'):
        t=(ROOT/'repair_independent_audit'/rel).read_text().replace(oldroot,newroot)
        (audit/rel).write_text(t)
    write(audit/'source_lock.json',{'files':{f.name:sha(f) for f in sorted(audit.iterdir())}})
    p['independent_source_lock_sha256']=sha(audit/'source_lock.json')
    p['resume']['resource_probe_sha256']=sha(ROOT/'native_final7529/training/job_7524/native_resource_probe.json')
    write(control/'protocol.json',p)
    # Functions which implement the scientific method are exactly unchanged.
    for symbol in ('train_epoch','learning_rate_factor','trainable_parameters','measure_remaining_sequence_resources'):
        def node(path):
            return ast.dump(next(n for n in ast.parse(path.read_text()).body if getattr(n,'name',None)==symbol),include_attributes=False)
        assert node(train)==node(old/'src/phase1/repair_trial.py'),symbol
    for rel in ('src/phase1/relational_memory.py','src/phase1/repair_evaluation.py',
                'src/phase1/exact_runtime_cache.py','src/phase1/common_anchor.py',
                'src/phase1/inputs.py','src/phase1/proposal_trial.py','src/phase1/training.py'):
        assert sha(control/rel)==oldlock['files'][rel]
    for f in control.rglob('*.py'):compile(f.read_text(),str(f),'exec')
    lock={'files':{str(f.relative_to(control)):sha(f) for f in sorted(control.rglob('*')) if f.is_file()}}
    write(control/'source_lock.json',lock)
    archive=ROOT/'approved_8h.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,'phase1_predictor_repair/approved_8h/'+str(f.relative_to(out)))
    result={'UTC':declaration['UTC'],'source_files':len(lock['files']),
      'source_lock_sha256':sha(control/'source_lock.json'),'protocol_sha256':sha(control/'protocol.json'),
      'independent_source_lock_sha256':sha(audit/'source_lock.json'),
      'archive_sha256':sha(archive),'archive':archive.name,'GPU_prior_actual':used,'GPU_remaining':remaining,
      'GPU_cap':28800,'old4_partial_origins_preserved':True,'scientific_functions_AST_and_core_source_exact':True,
      'native_submitted':False,'fresh9137_opened':False}
    write(ROOT/'approved_8h_ready.json',result);print(json.dumps(result,indent=2))
    deploy=(ROOT/'preparation/deploy_repair_execution.py').read_text()
    deploy=deploy.replace(oldroot,newroot)
    deploy=deploy.replace('/scratch/xl4358/phase1_predictor_repair_execution.zip','/scratch/xl4358/phase1_predictor_repair_approved8h.zip')
    deploy=deploy.replace('0f8cf9f71eb6bf6a29e614caa9e8168ea0a393a1efad7cf1b22ea33e028f9938',sha(archive))
    deploy=deploy.replace('9ab65cd2e422085672d753a9b8bb1ff462861271cfbb42899ce54c116e838ab4',sha(control/'source_lock.json'))
    deploy=deploy.replace('559935e65e6e54effe3cad4ee61902db92f6adff1423575f05a05d448232b3f3',sha(control/'protocol.json'))
    deploy=deploy.replace('379304f565c5c7fe896c8017113f7747c75c8ce9a746a6bece4eac8a42628da9',sha(audit/'source_lock.json'))
    deploy=deploy.replace("prefix = 'phase1_predictor_repair/repair_execution/'","prefix = 'phase1_predictor_repair/approved_8h/'")
    with zipfile.ZipFile(archive) as z:count=len(z.namelist())
    deploy=deploy.replace('len(archive.namelist()) != 60',f'len(archive.namelist()) != {count}')
    (ROOT/'preparation/deploy_8h_resume.py').write_text(deploy)
    compile(deploy,'deploy_8h_resume.py','exec')


if __name__=='__main__':main()
