import copy,importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'experiments/age_control_a1/provenance_guard.py'
spec=importlib.util.spec_from_file_location('age_provenance_guard',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def inputs():
    return ['7574|COMPLETED|0:0\n7575|COMPLETED|0:0\n7576|COMPLETED|0:0\n','7590|CANCELLED by 3850228|0:0\n',dict(job_id=7590,status='COMPLETE_WITH_INDEPENDENT_AUDIT',assay_returncode=0,independent_returncode=0),dict(status='INDEPENDENT_SCALAR_AUDIT_PASS',raw_rows=55296,independent_scalar_comparisons=180,all_equal_capacity_and_pair_axis=True,assets_stable=True,summary_sha256='expected'),'expected']
def test_released_interactive_after_success_is_accepted():
    assert m.validate_parent_provenance(*inputs())
@pytest.mark.parametrize('kind',['failed_batch','timeout','failed_assay','failed_audit','wrong_summary','incomplete_raw'])
def test_invalid_parent_is_rejected(kind):
    x=inputs()
    if kind=='failed_batch':x[0]=x[0].replace('7574|COMPLETED','7574|FAILED')
    if kind=='timeout':x[1]='7590|TIMEOUT|0:0'
    if kind=='failed_assay':x[2]['assay_returncode']=1
    if kind=='failed_audit':x[3]['status']='FAIL'
    if kind=='wrong_summary':x[3]['summary_sha256']='wrong'
    if kind=='incomplete_raw':x[3]['raw_rows']=55295
    with pytest.raises(AssertionError):m.validate_parent_provenance(*x)
