"""Freeze one new candidate only after native read-only diagnosis is received."""
from pathlib import Path
import json,hashlib,datetime,shutil,ast,zipfile

BASE=Path(__file__).resolve().parents[1]
OLD=BASE.parent/'phase1_predictor_repair/approved_8h'
NATIVE=BASE/'native_diagnosis7562'
HPC='/scratch/xl4358/phase1_predictor_repair_v2/execution'
OLDHPC='/scratch/xl4358/phase1_predictor_repair/approved_8h'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
    diagnosis=read(NATIVE/'diagnostic_results.json')
    transfer=read(NATIVE/'transfer_integrity.json')
    assert transfer['native_before_local_native_after_all_match']
    assert diagnosis['no_optimizer_or_training'] and diagnosis['GPU']==0 and len(diagnosis['slices'])==48
    assert diagnosis['current_sources_verified']==58 and len(diagnosis['checkpoint_hashes'])==6
    assert sha(NATIVE/'diagnostic_results.json')==transfer['files']['diagnostic_results.json']['sha256']
    dest=BASE/'execution'
    if dest.exists():raise ValueError('Execution already built; no implicit overwrite or retry')
    control=dest/'control';control.mkdir(parents=True)
    oldlock=read(OLD/'control/source_lock.json')
    excluded={'budget_resume_declaration.json','native_partial_origin.json','test_relational_memory.py'}
    for name,expected in oldlock['files'].items():
        assert sha(OLD/'control'/name)==expected
        if name in excluded:continue
        target=control/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD/'control'/name,target)
    shutil.copyfile(BASE/'preparation/centered_transport.py',control/'src/phase1/centered_transport.py')
    shutil.copyfile(BASE/'preparation/test_centered_transport.py',control/'test_centered_transport.py')
    for path in control.rglob('*'):
        if path.is_file() and path.suffix in {'.py','.json','.sbatch'}:
            s=path.read_text().replace(OLDHPC,HPC).replace('RelationalMemoryPredictor','CenteredTransportPredictor')
            s=s.replace('from .relational_memory import CenteredTransportPredictor','from .centered_transport import CenteredTransportPredictor')
            path.write_text(s)
    # The unused old pooling implementation is retained only as an attributed
    # dependency artifact; no architecture alias is introduced into it.
    shutil.copyfile(OLD/'control/src/phase1/relational_memory.py',control/'src/phase1/relational_memory.py')
    trial=control/'src/phase1/repair_trial.py';s=trial.read_text()
    for a,b in [('6411','9797'),('22389','19003')]:s=s.replace(a,b)
    start=s.index('            if seed == 17:\n                origin = p[\'resume\']')
    end=s.index('        def snapshot(epoch):',start)
    s=s[:start]+s[end:]
    start=s.index('        if resumed_from_original:\n')
    end=s.index('        elapsed = logs[-1]',start)
    s=s[:start]+"        if initial_epoch == 0:\n            save_torch(model_root / 'latest.pt', snapshot(0))\n"+s[end:]
    s=s.replace("old_resource_path = Path(p['resume']['old_training_root']) / 'native_resource_probe.json'", "old_resource_path = Path(p['inherited_resource_proof']['path'])")
    s=s.replace("expected_resource_sha = p['resume']['resource_probe_sha256']", "expected_resource_sha = p['inherited_resource_proof']['sha256']")
    s=s.replace("            resources = read_json(old_resource_path)\n", """            actual_gpu = read_json(p['inherited_resource_proof']['actual_gpu_path'])
            if file_hash(p['inherited_resource_proof']['actual_gpu_path']) != p['inherited_resource_proof']['actual_gpu_sha256']:
                raise ValueError('Inherited actual GPU identity changed')
            if actual_gpu['GPU'] != torch.cuda.get_device_name(0) or actual_gpu['torch'] != torch.__version__:
                raise ValueError('Inherited reference/encoder rate requires same actual A100 and Torch runtime')
            for name, expected_sha in p['inherited_resource_proof']['encoder_source_sha256'].items():
                if file_hash(Path(__file__).parent/name) != expected_sha:
                    raise ValueError('Inherited encoder/reference source changed')
            resources = read_json(old_resource_path)
""")
    # Old-source epoch provenance is inapplicable: every new candidate is trained
    # from its fixed seed, while the within-job first-epoch restore is preserved.
    a=s.index("            **({'epoch_execution_provenance'")
    b=s.index("\n\n\ndef actual_allocation_start",a)
    s=s[:a]+"            'new_candidate_all_seeds_from_scratch': True}\n"+s[b:]
    trial.write_text(s)
    resource=control/'src/phase1/resource_guard.py';s=resource.read_text().replace('7524: 164}', '7524: 164, 7535: 3386}');resource.write_text(s)
    preflight=control/'native_preflight.py';s=preflight.read_text().replace("'7485,7486,7487,7518'","'7485,7486,7487,7518,7535,7536,7537'").replace('(7485, 7486, 7487, 7518)','(7485, 7486, 7487, 7518, 7535, 7536, 7537)').replace("'7477,7480,7485,7524'","'7477,7480,7485,7524,7535'");preflight.write_text(s)
    train=control/'repair_train.sbatch';s=train.read_text().replace('--time=06:12:09','--time=05:16:00').replace('cv-phase1-repair','cv-phase1-centered-v2');train.write_text(s)
    independent=dest/'independent_audit';independent.mkdir()
    for name in read(OLD/'independent_audit/source_lock.json')['files']:
        s=(OLD/'independent_audit'/name).read_text().replace(OLDHPC,HPC)
        (independent/name).write_text(s)
    write(independent/'source_lock.json',{'files':{p.name:sha(p) for p in sorted(independent.iterdir()) if p.is_file()}})
    decision=read(control/'repair_decision.json');decision.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='ONE_SECOND_REPAIR_FIXED_AFTER_READONLY_DIAGNOSIS',native_diagnosis_sha256=sha(NATIVE/'diagnostic_results.json'),candidate='CenteredTransportPredictor',candidate_module='src/phase1/centered_transport.py',candidate_module_sha256=sha(control/'src/phase1/centered_transport.py'))
    decision['architecture']={'hidden':128,'heads':4,'history_queries':4,'relation_queries':2,'read_all_history':True,'no_selector_TopK':True,'no_absolute_visual_level_in_correction':True,'no_query_gap_or_extrapolated_values_in_learned_keys':True,'elapsed_time_only_explicit_linear_transport_value':True,'zero_value_history_gives_exact_reference':True,'all_value_readouts_linear_without_bias':True,'small_nonzero_readout_std':.001}
    decision['method_references']=['https://arxiv.org/abs/2205.13504','https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py','https://proceedings.mlr.press/v97/lee19d.html']
    decision['native_result_sha256']=sha(NATIVE/'diagnostic_results.json')
    decision['why_one_composite_trial']='Observed failures motivate testing common-offset invariance and separating learned content weights from explicit elapsed transport; this is one fixed composite, not an ablation proving a unique cause.'
    decision['adaptation_not_paper_reproduction']=True;decision['full_dev_and_confirmation_budget_must_fit_remaining19003s']=True
    decision.pop('full_dev_and_confirmation_budget_must_be_measured_and_fit_shared11753s',None)
    write(control/'repair_decision.json',decision)
    auth=read(BASE/'authorization.json');auth['scientific_sources_not_active_before_freeze']=True;write(control/'authorization.json',auth)
    p=read(control/'protocol.json');p.pop('resume',None);p['repair_decision_sha256']=sha(control/'repair_decision.json');p['independent_source_lock_sha256']=sha(independent/'source_lock.json')
    for obsolete in ('budget_resume_declaration_sha256','native_partial_origin_sha256'):
        p.pop(obsolete,None)
    p.update(declared_utc=decision['UTC'],frozen_utc=decision['UTC'],
        status='FROZEN_SECOND_REPAIR_ALL_NEW_MODELS_WITH_SHARED8H_CAP',
        human_authorization=auth['direct_human_instruction'],
        scope='Second single post-diagnosis repair on revealed6301 development; original full acceptance unchanged; fresh9137 closed until both supported.',
        candidate='CenteredTransportPredictor; last-observation-centered content weights and linear unbiased values; explicit elapsed transport outside learned keys; exact frozen recent anchor.',
        side_net=decision['architecture'],
        literature=[{'title':'Are Transformers Effective for Time Series Forecasting','url':'https://arxiv.org/abs/2205.13504','official_code':'https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py','borrowed':'NLinear last-observation centering principle; adapted to a frozen recent residual, not the original architecture'},
                    {'title':'Set Transformer','url':'https://proceedings.mlr.press/v97/lee19d.html','borrowed':'learned query content weighting over all history; no complete PMA reproduction claim'}],
        native_diagnosis_result_sha256=sha(NATIVE/'diagnostic_results.json'),
        reference_cost_profile='Inherited actual reference and encoder workload probes from7524, bound through7535 compact proof; require exact encoder/reference sources and same actual A100/Torch runtime. New candidate rates measured from full first epochs on both tasks.')
    p['resources'].update(prior_actual_GPU_seconds=9797,remaining_shared_GPU_seconds=19003,new_job_wall_seconds=18960,soft_remaining_seconds=18823)
    p['inherited_resource_proof']={'path':OLDHPC+'/training/job_7535/native_resource_probe.json','sha256':sha(OLD/'native_final7553/training/job_7535/native_resource_probe.json'),'actual_gpu_path':OLDHPC+'/training/job_7535/actual_gpu.json','actual_gpu_sha256':sha(OLD/'native_final7553/training/job_7535/actual_gpu.json'),'encoder_source_sha256':{n:sha(OLD/'control/src/phase1'/n) for n in ('encoders.py','cache.py','model.py','training.py','v5_encoding.py')}}
    p['second_repair_diagnostic_sha256']=sha(NATIVE/'diagnostic_results.json');p['all_candidate_models_from_scratch']=True
    write(control/'protocol.json',p)
    write(control/'software_preflight.json',{'local_synthetic_fixtures_passed':10,'no_scientific_data_or_model_training':True,'new_translation_invariance_and_zero_delta_boundary_tests':True,'full_native_input_RNG_and_CUDA_boundary_required':True})
    for path in control.rglob('*.py'):ast.parse(path.read_text())
    # Executable training loss/sampling, full metric loop and acceptance rule
    # sources are inherited exactly apart from the reader loading/provenance.
    def definition(path,name):return next(n for n in ast.parse(path.read_text()).body if getattr(n,'name',None)==name)
    assert ast.dump(definition(trial,'train_epoch'),include_attributes=False)==ast.dump(definition(OLD/'control/src/phase1/repair_trial.py','train_epoch'),include_attributes=False)
    for name in ('reporting.py','v5_synthetic_evaluation.py','v5_motion_evaluation.py','training.py','inputs.py','data.py','envs.py','exact_runtime_cache.py'):
        assert sha(control/'src/phase1'/name)==sha(OLD/'control/src/phase1'/name)
    write(control/'source_lock.json',{'files':{str(p.relative_to(control)):sha(p) for p in sorted(control.rglob('*')) if p.is_file() and p.suffix in {'.py','.json','.txt','.sbatch'} and p.name!='source_lock.json'}})
    archive=BASE/'execution.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for folder in (control,independent):
            for path in sorted(folder.rglob('*')):
                if path.is_file():z.write(path,'execution/'+str(path.relative_to(dest)))
    ready={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':str(archive),'archive_sha256':sha(archive),'science_source_lock_sha256':sha(control/'source_lock.json'),'science_source_files':len(read(control/'source_lock.json')['files']),'protocol_sha256':sha(control/'protocol.json'),'independent_source_lock_sha256':sha(independent/'source_lock.json'),'old_models_not_resumed':True,'train_epoch_AST_exact':True,'original_acceptance_sources_exact':True,'GPU_prior_seconds':9797,'GPU_remaining_seconds':19003,'GPU_cap_seconds':28800,'fresh9137_opened':False,'native_submission_exists':False}
    write(BASE/'execution_ready.json',ready);print(json.dumps(ready))

if __name__=='__main__':main()
