import math
from heartbeat import fresh

def test_heartbeat_written_during_local_slurm_rpc_is_fresh():
    loop_start=1792427434.522896
    peer_written=1792427434.5421834
    assert loop_start-peer_written < 0
    assert fresh({'updated_epoch':peer_written},clock=lambda:loop_start+0.3)

def test_bounded_clock_skew_does_not_hide_expiration():
    assert fresh({'updated_epoch':101.},clock=lambda:100.)
    assert not fresh({'updated_epoch':101.01},clock=lambda:100.)
    assert fresh({'updated_epoch':75.},clock=lambda:100.)
    assert not fresh({'updated_epoch':74.99},clock=lambda:100.)

def test_invalid_missing_timestamps_fail_closed():
    for v in [None,True,'100',math.nan,math.inf]:
        assert not fresh({'updated_epoch':v},clock=lambda:100.)
    assert not fresh({},clock=lambda:100.)
