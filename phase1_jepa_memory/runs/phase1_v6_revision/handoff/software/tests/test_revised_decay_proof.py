import pytest
from test_v5_real_diagnostics import final_fixture
from phase1.v5_real_diagnostics import validate_final_seed

def test_expected_revised_decay_accepts_actual020_only():
    latest,best,rows=final_fixture()
    latest['optimizer']['param_groups'][0]['weight_decay']=.20
    assert validate_final_seed(latest,best,rows,17,'signature',.20)['best_epoch']==4
    with pytest.raises(ValueError,match='optimizer/schedule'):
        validate_final_seed(latest,best,rows,17,'signature',.05)

def test_old_optimizer_cannot_masquerade_as_revised_trial():
    latest,best,rows=final_fixture()
    with pytest.raises(ValueError,match='optimizer/schedule'):
        validate_final_seed(latest,best,rows,17,'signature',.20)
