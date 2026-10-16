import ast
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/scratch/xl4358/phase1_jepa_memory_v5_real_frame_training')
if not OLD.is_dir():OLD=ROOT.parents[1]/'phase1_v5_encoding_l4'/'real_frame_training_software'

def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def test_reader_and_all_math_unchanged():
    assert (ROOT/'src/phase1/v5_real_reader.py').read_bytes()==(OLD/'src/phase1/v5_real_reader.py').read_bytes()
    old=functions(OLD/'src/phase1/v5_real_training.py');new=functions(ROOT/'src/phase1/v5_real_training.py')
    assert [k for k in old if old[k]!=new[k]]==['main']
    assert set(new)-set(old)=={'reuse_completed_fit'}

def test_protocol_preserves_all_scientific_dimensions_and_no_threshold_change():
    p=json.loads((ROOT/'protocol.json').read_text())
    assert p['optimizer']=={'name':'AdamW','lr':.0003,'weight_decay':.2,'warmup_fraction':.05,'schedule':'cosine_to_zero'}
    assert p['seeds']==[17,29,43] and p['epochs']==30
    assert p['reader']=={'rank':128,'hidden':256,'layers':2,'heads':4,'ffn':1024,'dropout':.1}
    assert p['horizons']==[1,4,16,64] and p['encoding_job_id']==7187
    assert p['reuse_fit']['npz_sha256']=='d90f99e4a91d290b7d27311de2820775981afe08dc053dc642e8a40636ed1fb3'

def test_execution_copies_original_fit_without_refit_and_matching_optimizer():
    tree=ast.parse((ROOT/'src/phase1/v5_real_training.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Name) and n.func.id=='training_fit' for n in calls)
    optimizer=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='AdamW']
    assert len(optimizer)==1
    assert next(k.value.value for k in optimizer[0].keywords if k.arg=='weight_decay')==.2
