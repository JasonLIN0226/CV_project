import hashlib
import numpy as np
import pytest
from handoff.inference import pack
from handoff.verify_handoff import verify_supplement


def test_packing_never_reads_future_features():
    z = np.arange(40*5, dtype=np.float32).reshape(40,5)
    actions = np.zeros((40,2),np.float32)
    expected = pack(z,actions,24,(2,10))
    z[25:] = np.nan
    actual = pack(z,actions,24,(2,10))
    for key in expected:
        assert np.array_equal(expected[key],actual[key])
    assert np.array_equal(actual['z'],z[[21,22,23,24,2,10]])


def test_invalid_retention_and_incomplete_actions_rejected():
    z = np.zeros((40,5),np.float32)
    actions = np.zeros((40,2),np.float32)
    for kept in ((10,2),(2,2),(-1,2),(2,21),(2,25)):
        with pytest.raises(ValueError):
            pack(z,actions,24,kept)
    with pytest.raises(ValueError):
        pack(z,actions[:29],24,(2,10))


def test_supplement_integrity_and_path_boundary(tmp_path):
    payload = b'proof'
    (tmp_path/'proof.json').write_bytes(payload)
    record = dict(relative_path='proof.json',bytes=5,sha256=hashlib.sha256(payload).hexdigest())
    m = dict(status='PUBLISHED_COPY_MATCHES_TEAMMATE_EXPECTED_SHA',files=[record])
    assert verify_supplement(m,tmp_path,True) == 1
    (tmp_path/'proof.json').write_bytes(b'wrong')
    with pytest.raises(ValueError):
        verify_supplement(m,tmp_path,True)
    record['relative_path'] = '../proof.json'
    with pytest.raises(ValueError):
        verify_supplement(m,tmp_path,True)
