import pytest
from cpu_incarnation import claim, require


def native(restarts=0, **changes):
    return dict(JobId='12', JobState='RUNNING', Account='csci_ga_2271-2026fa',
                AllocTRES='cpu=24,mem=1500M', NodeList='b-10-1',
                Restarts=str(restarts)) | changes


@pytest.mark.parametrize('restarts', [0, 1, 3])
def test_first_batch_pins_actual_provisioning_count(tmp_path, restarts):
    n = native(restarts)
    claim(tmp_path, 'primary', n, 'locked')
    assert require(tmp_path, 'primary', 12, n, 'locked')['restarts'] == restarts


def test_started_guardian_cannot_start_again(tmp_path):
    claim(tmp_path, 'primary', native(), 'locked')
    with pytest.raises(FileExistsError):
        claim(tmp_path, 'primary', native(1), 'locked')


@pytest.mark.parametrize('changes', [dict(Restarts='1'), dict(NodeList='b-10-2'),
                                  dict(JobId='13'), dict(JobState='PENDING'),
                                  dict(AllocTRES='cpu=12,gres/gpu=1')])
def test_changed_incarnation_is_rejected(tmp_path, changes):
    claim(tmp_path, 'peer', native(), 'locked')
    with pytest.raises(ValueError):
        require(tmp_path, 'peer', 12, native(**changes), 'locked')


def test_guardian_cannot_be_replaced_after_gpu_attempt(tmp_path):
    (tmp_path/'receipts').mkdir()
    (tmp_path/'receipts/encode_train_submission_attempt.json').write_text('{}')
    with pytest.raises(ValueError):
        claim(tmp_path, 'peer', native(), 'locked')
