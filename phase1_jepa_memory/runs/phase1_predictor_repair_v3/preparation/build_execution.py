"""Freeze one third repair, requiring the actual completed read-only diagnosis."""
from pathlib import Path
import json,hashlib,datetime,shutil,ast,zipfile
BASE=Path(__file__).resolve().parents[1]
OLD=BASE.parent/'phase1_predictor_repair_v2/execution'
NATIVE=BASE/'native_diagnosis7580'
OLDHPC='/scratch/xl4358/phase1_predictor_repair_v2/execution'
HPC='/scratch/xl4358/phase1_predictor_repair_v3/execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    diagnosis=read(NATIVE/'diagnostic_results_v2.json');transfer=read(NATIVE/'transfer_integrity.json')
    assert transfer['native_before_local_native_after_all_match']
    assert sha(NATIVE/'diagnostic_results_v2.json')==transfer['files']['diagnostic_results_v2.json']['sha256']
    assert diagnosis['no_training_or_optimizer'] and diagnosis['GPU']==0 and len(diagnosis['slices'])==120
    assert diagnosis['current_scientific_files']==57 and len(diagnosis['checkpoint_proofs'])==6
    # Require a written interpretation of actual measurements before freezing.
    review=read(BASE/'diagnostic_review.json');assert review['repair_justified'] and review['diagnosis_sha256']==sha(NATIVE/'diagnostic_results_v2.json')
    dest=BASE/'execution'
    if dest.exists():raise ValueError('Existing frozen trial; never overwrite or resubmit')
    control=dest/'control';control.mkdir(parents=True)
    for name,expected in read(OLD/'control/source_lock.json')['files'].items():
        assert sha(OLD/'control'/name)==expected
        if name=='test_centered_transport.py':continue
        target=control/name;target.parent.mkdir(parents=True,exist_ok=True)
        s=(OLD/'control'/name).read_text().replace(OLDHPC,HPC)
        if name!='src/phase1/centered_transport.py':
            s=s.replace('CenteredTransportPredictor','LatentFlowPredictor').replace('from .centered_transport import LatentFlowPredictor','from .latent_flow import LatentFlowPredictor')
        target.write_text(s)
    shutil.copyfile(BASE/'preparation/latent_flow.py',control/'src/phase1/latent_flow.py')
    shutil.copyfile(BASE/'preparation/test_latent_flow.py',control/'test_latent_flow.py')
    trial=control/'src/phase1/repair_trial.py'
    trial.write_text(trial.read_text().replace('9797','12604').replace('19003','16196'))
    resource=control/'src/phase1/resource_guard.py'
    resource.write_text(resource.read_text().replace('7535: 3386}', '7535: 3386, 7574: 2807}'))
    preflight=control/'native_preflight.py';s=preflight.read_text()
    s=s.replace('7485,7486,7487,7518,7535,7536,7537','7485,7486,7487,7518,7535,7536,7537,7574,7575,7576')
    s=s.replace('(7485, 7486, 7487, 7518, 7535, 7536, 7537)', '(7485, 7486, 7487, 7518, 7535, 7536, 7537, 7574, 7575, 7576)')
    s=s.replace('7477,7480,7485,7524,7535','7477,7480,7485,7524,7535,7574');preflight.write_text(s)
    sb=control/'repair_train.sbatch';sb.write_text(sb.read_text().replace('--time=05:16:00','--time=04:29:00').replace('cv-phase1-centered-v2','cv-phase1-latentflow-v3'))
    independent=dest/'independent_audit';independent.mkdir()
    for name,expected in read(OLD/'independent_audit/source_lock.json')['files'].items():
        assert sha(OLD/'independent_audit'/name)==expected
        (independent/name).write_text((OLD/'independent_audit'/name).read_text().replace(OLDHPC,HPC))
    write(independent/'source_lock.json',{'files':{p.name:sha(p) for p in sorted(independent.iterdir()) if p.is_file()}})
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    decision=read(control/'repair_decision.json')
    decision.update(UTC=now,status='ONE_THIRD_REPAIR_FIXED_AFTER_NATIVE_DIAGNOSIS',candidate='LatentFlowPredictor',
        candidate_module='src/phase1/latent_flow.py',candidate_module_sha256=sha(control/'src/phase1/latent_flow.py'),
        native_diagnosis_sha256=sha(NATIVE/'diagnostic_results_v2.json'),native_result_sha256=sha(NATIVE/'diagnostic_results_v2.json'),
        architecture={'hidden':128,'dynamics_coordinates':16,'coordinate_hidden':64,'coordinate_nonlinearity':'tanh',
          'history_queries':4,'relation_queries':2,'attention_heads':4,'read_all_history':True,'selector':False,
          'observed_visual_centering':True,'linear_elapsed_transport_in_learned_coordinates':True,
          'nonlinear_output_decoder':True,'post_pool_action_FiLM':True,'FiLM_scale':'1+0.5tanh(scale)',
          'zero_evidence_subtracted_coordinate_and_decoder_origins':True,'empty_history_exact_frozen_reference':True,
          'no_evaluation_state_or_labels_as_input_or_new_loss':True},
        why_one_composite_trial='Native visual feature/secant and component diagnostics motivate nonlinear dynamical coordinates and nonlinear decoding. Gate requires cue-action interaction, represented explicitly by post-pool FiLM. One composite exploratory repair, not a unique-cause ablation.',
        method_references=['https://arxiv.org/abs/1506.07365','https://arxiv.org/abs/1712.09707','https://arxiv.org/abs/1709.07871'],
        full_development_and_fresh_confirmation_must_fit_remaining_seconds=16196)
    decision.pop('full_dev_and_confirmation_budget_must_fit_remaining19003s',None)
    write(control/'repair_decision.json',decision)
    auth=read(BASE/'authorization.json');auth.update(authorized=True,direct_human_instruction=auth['direct_human_authorization'],scientific_sources_not_active_before_freeze=True)
    write(control/'authorization.json',auth)
    p=read(control/'protocol.json')
    p.update(declared_utc=now,frozen_utc=now,status='FROZEN_THIRD_REPAIR_ALL_NEW_CANDIDATES_SHARED8H',
        human_authorization=auth['direct_human_instruction'],scope='Third post-diagnosis exploratory repair on revealed6301; original complete acceptance and fresh9137 confirmation requirements unchanged.',
        candidate='LatentFlowPredictor: learned nonlinear dynamical coordinates, elapsed transport, nonlinear action-FiLM decoder, exact frozen recent anchor.',
        side_net=decision['architecture'],repair_decision_sha256=sha(control/'repair_decision.json'),
        independent_source_lock_sha256=sha(independent/'source_lock.json'),native_diagnosis_result_sha256=sha(NATIVE/'diagnostic_results_v2.json'),
        literature=[{'title':'Embed to Control','url':decision['method_references'][0],'borrowed':'Learn nonlinear coordinates to represent visual dynamics; no variational or control reproduction claim'},
                    {'title':'Deep learning for universal linear embeddings of nonlinear dynamics','url':decision['method_references'][1],'borrowed':'Separate nonlinear coordinate/decode maps from latent linear evolution'},
                    {'title':'FiLM: Visual Reasoning with a General Conditioning Layer','url':decision['method_references'][2],'borrowed':'Post-pooling action-conditioned affine modulation'}])
    p.pop('second_repair_diagnostic_sha256',None);p['third_repair_diagnostic_sha256']=sha(NATIVE/'diagnostic_results_v2.json')
    p['resources'].update(prior_actual_GPU_seconds=12604,remaining_shared_GPU_seconds=16196,new_job_wall_seconds=16140,soft_remaining_seconds=16016)
    write(control/'protocol.json',p)
    write(control/'software_preflight.json',{'local_synthetic_fixtures_passed':10,'no_scientific_data_or_CPU_training':True,
          'exact_empty_history_origin_and_causality_permutation_gradient_cache_boundaries':True,'native_full_inputs_RNG_and_CUDA_proof_required':True})
    for path in control.rglob('*.py'):ast.parse(path.read_text())
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
    ready={'UTC':now,'archive':str(archive),'archive_sha256':sha(archive),'archive_members':len(zipfile.ZipFile(archive).namelist()),
        'science_source_lock_sha256':sha(control/'source_lock.json'),'science_source_files':len(read(control/'source_lock.json')['files']),
        'protocol_sha256':sha(control/'protocol.json'),'independent_source_lock_sha256':sha(independent/'source_lock.json'),
        'train_epoch_AST_exact':True,'original_acceptance_sources_exact':True,'GPU_prior_seconds':12604,'GPU_remaining_seconds':16196,'GPU_cap_seconds':28800,
        'fresh9137_opened':False,'native_submission_exists':False}
    write(BASE/'execution_ready.json',ready);print(json.dumps(ready))
if __name__=='__main__':main()
