"""Small mathematical/software fixtures; no scientific dataset, model or GPU."""
import json
import numpy as np
from geometry import observed_forecast,propagate
from audit import independent_roll

def main():
    rng=np.random.default_rng(0);worst=0.
    for _ in range(100):
        pos=rng.uniform(.1,.9,2);v=rng.uniform(-.3,.3);actions=rng.uniform(-1,1,(100,2))
        for enabled in (True,False):
            p,n=propagate(pos,v,actions,100,enabled);other=independent_roll(pos,v,actions,enabled)
            worst=max(worst,float(np.max(np.abs(p-other))));assert worst<1e-12
    p,v,n=observed_forecast([[.3,.5],[.35,.5]],[2,12],np.zeros((10,2)),22)
    assert np.max(np.abs(p-[.4,.5]))<1e-12 and abs(v-.005)<1e-12
    p2,_,_=observed_forecast([[.3,.5],[.35,.5]],[2,12],np.ones((10,2)),22)
    assert abs(p2[0]-.425)<1e-12
    p3,_,_=observed_forecast([[.3,.5],[.35,.5]],[2,12],np.r_[np.zeros((10,2)),np.ones((10,2))*100],22)
    assert np.array_equal(p3,p)
    try: propagate([.5,.5],.01,[],10)
    except ValueError: pass
    else: raise AssertionError('Must reject missing actions')
    return dict(status='SOFTWARE_FIXTURES_PASS_NOT_SCIENTIFIC_RESULT',random_reflection_cases=200,max_independent_difference=worst,constant_velocity=True,known_action_displacement=True,future_action_suffix_ignored=True,incomplete_action_rejected=True,GPU=0,scientific_data=False)

if __name__=='__main__':print(json.dumps(main()))
