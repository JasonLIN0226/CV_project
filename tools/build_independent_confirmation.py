"""Freeze the complete approved fresh trial; metadata/code only, never data.

Run the software preflight separately first. This refuses to rebuild a frozen
source package or overwrite an earlier scientific execution directory.
"""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

REPO=Path(__file__).resolve().parents[1]
PROJECT=REPO.parent
SOURCE=REPO/'experiments/independent_confirmation'
RESULT=REPO/'results/phase1_independent_confirmation'
HPC=Path('/scratch/xl4358/phase1_independent_confirmation')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    if (SOURCE/'source_lock.json').exists():raise ValueError('Already frozen; no implicit rebuild or execution retry')
    proof=read(RESULT/'software_preflight.json')
    if proof['status']!='PASS_SOFTWARE_ONLY' or proof['pytest_passed']!=12 or proof['new_GPU_seconds']!=0:
        raise ValueError('Complete12-fixture preflight required')
    for name,expected in proof['source_files'].items():
        if sha(PROJECT/name)!=expected:raise ValueError('Source changed since software preflight: '+name)
    core=REPO/'experiments/predictor/control/src/phase1'
    package=SOURCE/'original_src/phase1'
    if package.exists():
        if {p.name:sha(p) for p in package.glob('*.py')}!={p.name:sha(p) for p in core.glob('*.py')}:
            raise ValueError('Predeployment original core snapshot changed')
    else:
        package.mkdir(parents=True)
        for p in sorted(core.glob('*.py')):shutil.copy2(p,package/p.name)
    original=read(REPO/'results/reference/gate/resolved_config.json')
    second=read(REPO/'results/reference/motion/resolved_config.json')
    exclude={'output','device','threads','experiment'}
    if {k:v for k,v in original.items() if k not in exclude}!={k:v for k,v in second.items() if k not in exclude}:
        raise ValueError('Shared original two-task data/training contract differs')
    c=copy.deepcopy(original);c.pop('experiment');c.update(output=str(HPC/'fresh'),device='cuda',threads=4)
    c['data']['seed']=9137
    c['data']['pairs']={'train':512,'validation':96,'test_id':256,'test_long_gap':256,'test_distractor_shift':256}
    write(SOURCE/'config.json',c)
    evidence={}
    def bind(local,remote):evidence[str(remote)]=sha(local)
    for task,folder,job in [('gate','gate_locked_test',8332),('motion','motion_observation_mix_locked_test',8522)]:
        local=REPO/'results'/folder/'native_completed';remote=Path('/scratch/xl4358')/('phase1_'+folder)
        s=read(local/f'output/job_{job}/summary.json');a=read(local/f'output/job_{job}/independent_audit.json')
        if s['status']!=task.upper()+'_DEVELOPMENT_REPAIR_SUPPORTED' or not a['status'].startswith('PASS_INDEPENDENT_'):
            raise ValueError('Both complete independently audited development tasks required')
        for rel in ['source_lock.json','candidate_lock.json',f'output/job_{job}/summary.json',f'output/job_{job}/independent_audit.json']:
            bind(local/rel,remote/rel)
    old=read(REPO/'experiments/predictor/control/protocol.json')
    for task in ('gate','motion'):
        local=REPO/f'results/reference/{task}/resolved_config.json'
        if sha(local)!=old['immutable_inputs'][task]['data_source_config_sha256']:
            raise ValueError('Original native immutable config bytes differ')
        bind(local,Path(old['immutable_inputs'][task]['data_source_config']))
    parent=PROJECT/'phase1_jepa_memory/runs/phase1_predictor_repair_v2/native_final7579/training/job_7574'
    resource=read(parent/'native_resource_probe.json');gpu=read(parent/'actual_gpu.json')
    native_parent=Path('/scratch/xl4358/phase1_predictor_repair_v2/execution/training/job_7574')
    for name in ('native_resource_probe.json','actual_gpu.json'):bind(parent/name,native_parent/name)
    protocol=dict(status='FROZEN_COMPLETE_FRESH_CONFIRMATION_PROTOCOL',
        frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        user_authorized_conditional_fresh_confirmation=True,GPU_cap_seconds=28800,prior_settled_seconds=13131,
        prior_native_GPU_jobs={7477:100,7480:126,7485:6021,7524:164,7535:3386,7574:2807,
            7582:119,8496:42,8511:312,8512:27,8518:27},
        expected_GPU=gpu['GPU'],expected_torch=gpu['torch'],encoder_metadata=resource['original_encoder_metadata'],
        native_rate_proof_sha256=sha(parent/'native_resource_probe.json'),development_evidence=evidence,
        fresh_seed=9137,tasks=['gate','motion'],seeds=[17,29,43],
        reference_models=6,candidate_models=6,epochs_each=120,steps_each=7680,
        original_data_config_preserved_except_fresh_seed_and_test_pairs=True,
        pairs_per_task=c['data']['pairs'],normalization='Original shared all-training-episode float64 sum/squared-sum order',
        basis_readout='Original training-only128basis plus8readouts, unchanged originalvalidation selection',
        conditions=['recent_only','full_history','privileged_small_budget','random_small_budget','wrong_evidence','persistence'],
        candidate_training='Exact CenteredTransport: Gate anchor1/Motion0, originalMSE+2pair+.5rank, AdamW.0003/wd.0001/clip1/warmup5cosine120',
        reference_training='Original DynamicsPredictor, AdamW.001/wd.0001/clip1/StepLR40gamma.5, MSE+2pair+.5rank',
        checkpoint_selection='Earliest equal-four-condition/two-horizon validation MSE minimum',
        gate_inference='Whole time-extrapolation projection OFF for all contexts; no per-test switching',
        motion_inference='Same-history fresh reference visual output, current fresh candidate guarded position, fixed.5 observed-geometry innovation map',
        motion_maps=dict(heads=2,shared_across_three_neural_seeds=True,fit_split='train',pairs=512,
            inputs='2Dfuture-position from actual two retained observed latent positions +known actions',
            target='126D readout-null projection oftraining future-minus-current features',
            recipe='Gaussian KRR; train-only mean/std+median-distance; input-onlydf<=32;condition<=1e6;CUDAfit',
            pair_grouped_crossfit_folds=2,fold_seed=0,crossfit_repeats=2000,crossfit_requirement='Positive95%pair-gain lowerbound versus matched train-mean innovation',
            coefficient=.5,coefficient_search=False,hyperparameter_or_target_based_regularizer_search=False),
        test_access='No generation/opening/encoding of any heldout9137 tests until all12 full neural models plus2maps/alltransforms/source are current-SHA locked',
        acceptance=dict(original_prediction_checks_per_task=51,fixed_reference_checks_per_task=32,
            gate_deletion='All72 critical-vs-other cue-rank cells: bothmetric effect>=10% andpaired95%lower>0; originalrules unchanged',
            motion_deletion='All72 cross-horizon future-feature-selected oracle versus uniform/FIFO/reject cells:bothfeature+position paired95%gainlower>0',
            all_original_structural_readout_full_history_requirements=True,failed_seed_or_slice_removed=False),
        statistics=dict(core_and_fixed_bootstrap=2000,deletion_bootstrap=10000,seed=0,
            clustering='Average siblings/pools within pair; average3seeds only for mean cells; bootstrap basepairs, retain eachseed'),
        raw_prediction_rows_per_task_per_role=62208,total_prediction_rows_all_roles=248832,
        raw_deletion_rows=248832,paired_deletion_rows=18432,static_pool_diagnostic_not_a_policy=True,
        resource_guard=dict(full_RGB_frame_upper_bound=198656,encoder_batch=2,RGB_deduplication=False,
            reference_and_candidate_rate_proof='native7574',full_training_raw_seconds=5194.312324919995,
            full_encoding_raw_seconds=4910.9994851844385,map_fit_raw_seconds=27,
            native_equivalence_reserved_seconds=300,safety=1.15,fixed_remaining_reserve_seconds=968.295233886,
            preflight_GPU_hard_limit=300,encoding_train_GPU_hard_limit=2700,
            full_training_GPU_hard_limit=6600,encoding_tests_GPU_hard_limit=3900,
            all_actual_allocated_seconds_charged=True,no_scope_shrink_to_fit=True),
        no_retuning_after_fresh_failure=True,no_retention_policy_training=True,no_GitHub_update=True,
        scope='Controlled synthetic I-JEPA;fixed supervised readout andrenderer-specific observed-RGB guard;notV-JEPA/notrealvideo',
        software_preflight=dict(status=proof['status'],pytest_passed=12,proof_sha256=sha(RESULT/'software_preflight.json')),
        predeployment_execution_amendment=read(RESULT/'predeployment_amendment.json'))
    write(SOURCE/'protocol.json',protocol)
    write(SOURCE/'authorization.json',dict(authorized=True,conditional_development_now_supported=True,
        cumulative_GPU_cap_seconds=28800,prior_actual_seconds=13131,
        human_request='这轮如果通过，就直接继续做独立确认，在我批准的GPU总预算内把proposal里的Phase1验证做完整；累计5h后另加3h=8h。',
        latest_request='自行进行研究，每轮分析并针对性计划直到显著进步；两任务都通过后直接独立确认。',
        no_threshold_or_sample_or_seed_or_epoch_change=True,no_policy_training=True,no_GitHub_update=True))
    files={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()
        and (p.suffix in ('.py','.sh') or p.name in ('config.json','protocol.json','authorization.json'))
        and '__pycache__' not in p.parts}
    write(SOURCE/'source_lock.json',dict(status='FROZEN_COMPLETE_INDEPENDENT_CONFIRMATION_SOURCE',files=files))
    RESULT.mkdir(exist_ok=True,parents=True)
    archive=RESULT/'deployment.zip'
    if archive.exists():raise ValueError('Do not replace a pre-existing deployment archive')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in (*sorted(files),'source_lock.json'):z.write(SOURCE/name,name)
    write(RESULT/'ready.json',dict(status='SOURCE_FROZEN_READY_FOR_NATIVE_DEV_PREFLIGHT_NOT_FRESH_GENERATION',
        source_files=len(files),source_lock_sha256=sha(SOURCE/'source_lock.json'),protocol_sha256=sha(SOURCE/'protocol.json'),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,HPC_root=str(HPC),
        software_tests=12,prior_GPU_seconds=13131,remaining_GPU_seconds=15669,
        full_estimated_GPU_including_native_equivalence=12965.453815506098,
        fresh9137_generated=False,submitted=False,phase1_go=False))
    print(json.dumps(read(RESULT/'ready.json'),indent=2))


if __name__=='__main__':main()
