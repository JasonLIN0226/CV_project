"""CPU-only recovery checks: identity formats, accounting, and unchanged science.

The extended accounting fixture here is explicitly synthetic; its first12
fields and retained control are copied from the preserved17761 native records.
No Slurm, CUDA, arrays, encoder or scientific runner is executed.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

BASE = Path(__file__).parent
REPO = BASE.parents[1]
RAW = REPO/'results/motion_v2/step3_4/job17761_stop/raw'


def module(name):
    spec = importlib.util.spec_from_file_location('recovery_test_' + name, BASE/(name+'.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


R = module('step34_runtime_recovery1')
A = module('step34_accounting_recovery1')
S = module('submit_step34_recovery1')
E = module('export_step34_recovery1')
UUID = '12345678-1234-1234-1234-123456789abc'
NAME = 'NVIDIA A100-SXM4-40GB'


class Torch251CUuuid:
    """Mimic the official2.5.1 CUuuid binding, not a real device observation."""
    def __str__(self):
        return UUID


def torch(uuid=UUID, name=NAME, count=1):
    return SimpleNamespace(__version__='2.5.1+cu121', cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: count,
        get_device_name=lambda _: name, get_device_properties=lambda _: SimpleNamespace(uuid=uuid)))


@pytest.mark.parametrize('value', [UUID, UUID.upper(), 'GPU-'+UUID, 'gpu-'+UUID.upper(), Torch251CUuuid()])
def test_torch251_and_smi_uuid_representations_have_same_identity(value):
    events = []
    def smi(argv, timeout):
        assert argv[-1] == '--id=GPU-'+UUID and timeout == 30
        return NAME+', GPU-'+UUID+'\n'
    result = R.verify_gpu(torch_module=torch(value), command_fn=smi, environ={'CUDA_VISIBLE_DEVICES':'0'}, record_probe=events.append)
    assert result['gpu_uuid'] == 'GPU-'+UUID and result['status']=='passed'
    assert events == [result] and result['torch_uuid_str'] == str(value)


@pytest.mark.parametrize('value', ['', 'GPU-1', 'MIG-'+UUID, UUID+' ', 'GPU-GPU-'+UUID, b'123', UUID.replace('-',''), UUID+'0'])
def test_unsupported_uuid_remains_blocking_and_is_recorded(value):
    events=[]
    def forbidden(*a, **k): raise AssertionError('No query after malformed torch identity')
    with pytest.raises(RuntimeError):
        R.verify_gpu(torch_module=torch(value), command_fn=forbidden, environ={}, record_probe=events.append)
    assert len(events)==1 and events[0]['status']=='failed' and events[0]['torch_uuid_repr']==repr(value)


def test_valid_format_does_not_hide_different_gpu():
    events=[]
    with pytest.raises(RuntimeError, match='UUIDs disagree'):
        R.verify_gpu(torch_module=torch(Torch251CUuuid()), command_fn=lambda *a, **k: NAME+', GPU-aaaaaaaa-1234-1234-1234-123456789abc',
                     environ={},record_probe=events.append)
    assert events[0]['nvidia_smi'] and events[0]['torch_uuid_str']==UUID


def native_start():
    return (RAW/'scientific_job_started/scontrol_start.txt').read_text()


def extended_synthetic_rows():
    raw=(RAW/'runtime/attempt1/monitor-20261102T0144Z/sacct.stdout.txt').read_text()
    rows=[]
    for line in raw.splitlines():
        cells=line.split('|'); assert len(cells)==12
        current=cells[3]=='csci_ga_2271-2026fa'
        # DBIndex/Cluster/Submit/Restarts/node/limit additions are synthetic.
        cells += ['synthetic-cluster', '880026' if current else '880024',
                  '2026-11-02T01:33:01' if current else '2024-11-12T19:49:00',
                  '0', 'b-19-49' if current else 'synthetic-v100', '175' if current else '60']
        rows.append(dict(zip(A.FIELDS,cells)))
    return rows


def raw_text(rows): return '\n'.join('|'.join(row[k] for k in A.FIELDS) for row in rows)+'\n'
def identity(): return S.expected_submission(S.load('slurm_records').parse_control(native_start()))
def old_context():
    root='/scratch/xy3257/motion-v2-20261101-step34-deployment1/repo'
    return dict(S.context(),workdir=root,wrapper_path=root+'/experiments/motion_v2/run_step34.sbatch')
def expired(): return {'returncode':1,'stdout':'','stderr':'slurm_load_jobs error: Invalid job id specified\n'}


def test_recycled_job_number_keeps_all_raw_rows_and_selects_database_identity():
    rows=extended_synthetic_rows();text=raw_text(rows)
    result=A.select_allocation(text,'17761',identity())
    assert result['all_rows']==rows and len(result['excluded_historical_rows'])==3
    assert len(result['selected_rows'])==3 and result['batch_step_ids']==['17761.batch']
    assert result['allocation_row']['ElapsedRaw']=='48'
    terminal=A.terminal_from_accounting(result['allocation_row'],old_context(),native_start(),expired())
    assert terminal['actual_a100_gpu_seconds']==48 and terminal['restarts']==0
    assert terminal['fresh_terminal_scontrol_available'] is False and terminal['terminal_scontrol_fabricated'] is False


def test_actual_extended17761_native_record_and_export_replay(monkeypatch):
    path=REPO/'results/motion_v2/step3_4_recovery1/native_inspection17761/sacct.stdout.txt'
    raw=path.read_text()
    assert hashlib.sha256(raw.encode()).hexdigest()=='ea0fb88dc55d13ef123673073ed56c2007765fa03e2730e782e37ae1174cfa85'
    parsed=A.select_allocation(raw,'17761',identity())
    assert len(parsed['all_rows'])==6 and len(parsed['selected_rows'])==3
    assert parsed['selection_keys']['DBIndex']=='15404147354823130112'
    assert parsed['batch_step_ids']==['17761.batch']
    terminal=A.terminal_from_accounting(parsed['allocation_row'],old_context(),native_start(),expired())
    assert terminal['actual_a100_gpu_seconds']==48 and terminal['restarts']==0
    # Replay the new export's read-only validator using the old job's real context.
    monkeypatch.setattr(E,'native_context',old_context)
    control=dict(expired(),name='final_scontrol',argv=['scontrol','show','job','--oneliner','17761'])
    receipt=dict(mode='settle',attempt=1,job_id='17761',expected_identity=identity(),
                 complete_accounting=parsed,terminal_accounting=terminal,startup_control=native_start(),
                 startup_control_sha256=hashlib.sha256(native_start().encode()).hexdigest(),
                 terminal_scontrol='',terminal_scontrol_sha256=hashlib.sha256(b'').hexdigest(),
                 cumulative_a100_seconds=3753,commands=[
                    dict(name='final_sacct_duplicates',argv=['sacct','--duplicates','-j','17761','--noheader','--parsable2',
                         '--format='+','.join(A.FIELDS)],returncode=0,stdout=raw,stderr=''),control])
    status=dict(status='passed',native_state='FAILED',restarts=0,actual_a100_gpu_seconds=48,cumulative_a100_seconds=3753)
    exported=E.validate_terminal_evidence(receipt,status,raw,'','17761',1)
    assert exported['native_terminal_verified'] and exported['actual_a100_gpu_seconds']==48
    assert exported['native_commands_executed_by_export']==0
    changed=copy.deepcopy(receipt);changed['startup_control']+='x'
    with pytest.raises(ValueError): E.validate_terminal_evidence(changed,status,raw,'','17761',1)


@pytest.mark.parametrize('kind',['double_top','double_step','no_current','wrong_submit','wrong_dbindex','malformed'])
def test_ambiguous_identity_still_stops(kind):
    rows=extended_synthetic_rows();expected=identity()
    if kind=='double_top': rows.append(copy.deepcopy(rows[3]))
    if kind=='double_step': rows.append(copy.deepcopy(rows[4]))
    if kind=='no_current': rows=rows[:3]
    if kind=='wrong_submit': expected['Submit']='2026-11-02T01:33:02'
    if kind=='wrong_dbindex': rows[3]['DBIndex']=''
    text=raw_text(rows)+(('17761|bad\n') if kind=='malformed' else '')
    with pytest.raises(ValueError): A.select_allocation(text,'17761',expected)


@pytest.mark.parametrize('key,value',[('Restarts','1'),('User','other'),('AllocTRES','cpu=12,gres/gpu=2'),('State','RUNNING'),
                                    ('ElapsedRaw','10501'),('NodeList','wrong-node'),('TimelimitRaw','180'),('Submit','2026-11-02T01:33:02')])
def test_terminal_contradictions_are_not_relaxed(key,value):
    row=extended_synthetic_rows()[3];row[key]=value
    with pytest.raises(ValueError): A.terminal_from_accounting(row,old_context(),native_start(),expired())


@pytest.mark.parametrize('error',[{'returncode':1,'stdout':'','stderr':'permission denied'},
                                {'returncode':None,'stdout':'','stderr':'timeout'},
                                {'returncode':1,'stdout':'unexpected','stderr':'slurm_load_jobs error: Invalid job id specified'}])
def test_only_explicit_native_controller_expiry_is_accepted(error):
    with pytest.raises(ValueError): A.terminal_from_accounting(extended_synthetic_rows()[3],old_context(),native_start(),error)


def history(seconds=0):
    return [{'attempt':1,'job_id':'99999','terminal':True,'restarts':0,'actual_a100_gpu_seconds':seconds,
             'four_facts':dict.fromkeys(S.FOUR_FACTS,True)}]


def test_full_cap_includes_previous17761_and_all_future_infrastructure_seconds():
    assert S.validate_admission(1,[],marker_absent=True,output_absent=True)['projected_cumulative_a100_seconds']==14253
    assert S.validate_admission(2,history(147),marker_absent=True,output_absent=True)['projected_cumulative_a100_seconds']==14400
    with pytest.raises(ValueError): S.validate_admission(2,history(148),marker_absent=True,output_absent=True)
    with pytest.raises(ValueError): S.validate_admission(3,history(),marker_absent=True,output_absent=True)


@pytest.mark.parametrize('fact',S.FOUR_FACTS)
def test_recovery_scientific_entry_cannot_be_retried(fact):
    rows=history();rows[0]['four_facts'][fact]=False
    with pytest.raises(ValueError):S.validate_admission(2,rows,marker_absent=True,output_absent=True)


def test_original72_files_and_scientific_config_unchanged():
    protocol=json.loads((BASE/'protocol_step34.json').read_text())
    for relative,sha in protocol['source_sha256'].items():
        assert hashlib.sha256((REPO/relative).read_bytes()).hexdigest()==sha
    old=json.loads((BASE/'config_step34.json').read_text());new=json.loads((BASE/'config_step34_recovery1.json').read_text())
    assert {k:v for k,v in old.items() if k!='budget'}=={k:v for k,v in new.items() if k!='budget'}
    assert new['budget']==dict(old['budget'],prior_a100_seconds=3753,remaining_a100_seconds=10647)


def test_runner_scientific_tail_and_encoder_are_identical():
    def nodes(name):
        return {n.name:n for n in ast.parse((BASE/name).read_text()).body if isinstance(n,ast.FunctionDef)}
    old,new=nodes('run_step34.py'),nodes('run_step34_recovery1.py')
    for name in ('cpu','encode_new','validate_cpu_receipt','finish','environment'):
        assert ast.dump(old[name])==ast.dump(new[name])
    def scientific_tail(node):
        body=next(n.body for n in node.body if isinstance(n,ast.Try))
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='core')
        return ast.dump(ast.Module(body=body[start:],type_ignores=[]))
    assert scientific_tail(old['gpu'])==scientific_tail(new['gpu'])


def test_preflight_wrapper_and_export_agree_on_new_paths(monkeypatch):
    monkeypatch.setattr(S, "REPO", S.ROOT / "repo")
    directives=S.validate_wrapper(BASE/'run_step34_recovery1.sbatch')
    assert directives['--gres']=='gpu:a100:1' and directives['--time']=='02:55:00'
    assert S.ROOT==E.ROOT and E.SCIENCE_RELATIVE=='repo/'+S.SCIENCE_OUTPUT
    assert S.ACCOUNTING_FIELDS==A.FIELDS==E.ACCOUNTING_FIELDS
    assert S.PRIOR_SECONDS==R.PRIOR_A100_SECONDS==3753 and S.MAX_ATTEMPTS==2
    assert E.native_context()==S.context()
