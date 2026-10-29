"""Bounded online selectors; no dataset labels, targets or future action plan."""
import itertools, time
import numpy as np
import torch

POLICIES=('fifo','reservoir','temporal_buckets','visual_diversity','prediction_surprise','attention')

def update(name, previous, arrival, clock, recent, vectors, anchor, scores, rng):
    pool=list(previous)+[arrival]
    assert len(previous)<=2 and arrival not in previous and len(set(pool))==len(pool)
    if len(pool)<=2:return sorted(pool)
    pairs=list(itertools.combinations(pool,2))
    if name=='fifo':chosen=sorted(pool)[-2:]
    elif name=='reservoir':
        # arrival timestamps start at zero: third arrival has ordinal two.
        j=int(rng.integers(0,arrival+1));chosen=list(previous)
        if j<2:chosen[j]=arrival
    elif name=='temporal_buckets':
        def score(pair):
            ages=[clock-t for t in pair]
            return (len({int(np.floor(np.log2(max(1,a/recent)))) for a in ages}),max(ages)-min(ages),sum(pair),tuple(pair))
        chosen=max(pairs,key=score)
    elif name=='visual_diversity':
        assert set(vectors)==set(pool)
        def score(pair):
            a,b=(vectors[t] for t in pair)
            return (float(np.linalg.norm(anchor-a)+np.linalg.norm(anchor-b)+np.linalg.norm(a-b)),tuple(pair))
        chosen=max(pairs,key=score)
    elif name in ('prediction_surprise','attention'):
        assert set(scores)==set(pool)
        chosen=sorted(pool,key=lambda t:(scores[t],t),reverse=True)[:2]
    else:raise ValueError(name)
    assert len(chosen)==2 and set(chosen)<=set(pool)
    return sorted(chosen)

@torch.inference_mode()
def replay(z, projected, actions, query, model, collate, device, rng_seed):
    """z is observation stream; only z[:clock+1] may be read in loop.
    Auxiliary state: at most 3 pending forecast coefficients and scalar scores
    for the recent buffer plus two retained surprise items. No archive rescans.
    """
    recent=4;mem={p:[] for p in POLICIES};rng=np.random.default_rng(rng_seed)
    pending={};surprise={};trace=[];elapsed={p:0. for p in POLICIES};max_aux=0
    coeff=[]
    hook=model.output.register_forward_hook(lambda module,args,out:coeff.append(out.detach().cpu().numpy().copy()))
    try:
        for clock in range(query+1):
            if clock in pending:
                origin,c=pending.pop(clock)
                assert origin==clock-3 and origin>=clock-recent+1
                forecast=z[origin]+model.target_mean.cpu().numpy()+c@model.output_basis.cpu().numpy().T
                surprise[clock]=float(np.mean((forecast-z[clock])**2))
            else:surprise[clock]=-np.inf
            # Forecast issued now using only currently observed R frames/actions.
            if clock>=recent-1 and clock+3<=query-recent:
                times=np.arange(clock-recent+1,clock+1)
                past=actions[:clock]
                prefix=np.concatenate([np.zeros((1,2),np.float32),np.cumsum(past,axis=0)])
                held=actions[clock-1] if clock else np.zeros(2,np.float32)
                sample={'z':z[times].copy(),'age':(clock-times).astype(np.float32),'type':np.zeros(recent,np.int64),'past_actions':prefix[clock]-prefix[times],'actions':np.tile(held,(6,1)),'current':z[clock].copy(),'projected_z':projected[times].copy(),'projected_current':projected[clock].copy()}
                coeff.clear();model(collate([sample],device));assert len(coeff)==2
                pending[clock+3]=(clock,coeff[0][0].copy());coeff.clear()
            arrival=clock-recent
            if arrival>=0:
                for name in POLICIES:
                    start=time.perf_counter();previous=mem[name];pool=previous+[arrival]
                    vectors={t:projected[t].copy() for t in pool} if name=='visual_diversity' else {}
                    scores={}
                    if name=='attention':
                        p=np.stack([projected[t] for t in pool]);delta=p-projected[clock]
                        x=torch.as_tensor(np.concatenate([p,delta],-1),dtype=torch.float32,device=device)
                        sc=(model.score(x).squeeze(-1)+torch.log1p(torch.as_tensor(delta,device=device).square().mean(-1))).cpu().numpy()
                        scores=dict(zip(pool,map(float,sc)))
                    if name=='prediction_surprise':scores={t:surprise[t] for t in pool}
                    chosen=update(name,previous,arrival,clock,recent,vectors,projected[clock],scores,rng)
                    elapsed[name]+=time.perf_counter()-start
                    trace.append({'policy':name,'clock':clock,'previous':previous.copy(),'arrival':arrival,'selected':chosen.copy()})
                    mem[name]=chosen
                # Scores for discarded observations are removed immediately.
                keep=set(mem['prediction_surprise'])|set(range(max(0,clock-recent+1),clock+1))
                surprise={t:v for t,v in surprise.items() if t in keep}
            max_aux=max(max_aux,sum(c.nbytes for _,c in pending.values())+len(surprise)*16)
        return mem,trace,elapsed,max_aux
    finally:hook.remove()
