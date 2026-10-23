import json,hashlib
import pytest
from cpu_equivalence_reuse import check_scientific_identity

def test_inherited_science_must_be_byte_identical(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    for p in [old,new]:(p/'core.py').write_text('fixed science\n')
    old.joinpath('source_lock.json').write_text(json.dumps({'files':{'core.py':hashlib.sha256((old/'core.py').read_bytes()).hexdigest()}}))
    assert check_scientific_identity(old,new)
    (new/'core.py').write_text('changed science\n')
    with pytest.raises(ValueError,match='Scientific bytes'):check_scientific_identity(old,new)

def test_original_proof_source_tampering_is_rejected(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    for p in [old,new]:(p/'core.py').write_text('fixed\n')
    old.joinpath('source_lock.json').write_text(json.dumps({'files':{'core.py':'wrong'}}))
    with pytest.raises(ValueError,match='Original completed CPU'):check_scientific_identity(old,new)
