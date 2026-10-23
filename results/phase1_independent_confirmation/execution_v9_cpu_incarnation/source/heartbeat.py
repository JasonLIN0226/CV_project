"""Compare a heartbeat against a clock sampled AFTER its snapshot was read."""
import math,time

def fresh(snapshot, *, clock=time.time, maximum_age=25., clock_skew=1.):
    stamp=snapshot.get('updated_epoch')
    if isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp):
        return False
    age=clock()-stamp
    return math.isfinite(age) and -clock_skew <= age <= maximum_age
