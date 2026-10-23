"""Orthogonal decompositions of frozen normalized feature residuals; no fitting."""
import numpy as np

COMPONENTS=('total','visible','null','basis_sensitive','basis_null','outside_basis',
            'readout_consistency','target_readout_error','task_cross','task_reconstructed')
SCORES={'full':'source_feature_mse','full64':'total','visible':'visible','null':'null',
        'basis_sensitive':'basis_sensitive','basis_null':'basis_null',
        'readout_consistency':'readout_consistency','uniform':None,'task_optimal':'source_task_error'}

def operators(B,weight,std,mean,bias):
    B=np.asarray(B,float);A=np.asarray(weight,float)[:,1:]/np.asarray(std,float)[:,None]
    b=np.asarray(bias,float)[1:]-np.asarray(mean,float)@A
    Qa,Ra=np.linalg.qr(A,mode='reduced');Qb,Rb=np.linalg.qr(B,mode='reduced')
    C=Qb.T@A;Qc,Rc=np.linalg.qr(C,mode='reduced');Qs=Qb@Qc
    assert np.linalg.matrix_rank(Ra)==np.linalg.matrix_rank(Rc)==2 and np.linalg.matrix_rank(Rb)==128
    gram=B.T@B;F=B.T@A;G=np.linalg.solve(gram,F);right=np.linalg.solve(F.T@G,G.T)
    return dict(B=B,A=A,b=b,Qa=Qa,Qb=Qb,Qs=Qs,right=right)

def restore(base,geometry,enabled,op):
    p=np.asarray(base,float)
    d=geometry-(p@op['A']+op['b']);corrected=p+(d@op['right'])@op['B'].T
    return np.where(np.asarray(enabled)[:,None],corrected,p)

def decompose(pred,target,truth,op):
    e=np.asarray(pred,float)-np.asarray(target,float);D=e.shape[1]
    total=np.sum(e*e,axis=1)/D
    visible=np.sum((e@op['Qa'])**2,axis=1)/D
    basis_energy=np.sum((e@op['Qb'])**2,axis=1)/D
    sensitive=np.sum((e@op['Qs'])**2,axis=1)/D
    null=total-visible;basis_null=basis_energy-sensitive;outside=total-basis_energy
    assert min(null.min(),basis_null.min(),outside.min())>-1e-9
    d=e@op['A'];c=np.asarray(target,float)@op['A']+op['b']-truth
    consistency=np.mean(d*d,axis=1);calibration=np.mean(c*c,axis=1);cross=np.sum(d*c,axis=1)
    task=np.mean((d+c)**2,axis=1)
    assert np.max(abs(task-consistency-calibration-cross))<1e-9
    return dict(total=total,visible=visible,null=null,basis_sensitive=sensitive,basis_null=basis_null,
                outside_basis=outside,readout_consistency=consistency,target_readout_error=calibration,
                task_cross=cross,task_reconstructed=task)

def selected(values,tol):
    a=np.asarray(values,float);return np.flatnonzero(a<=a.min()+tol)

def group_metrics(rows):
    task=np.array([float(r['source_task_error']) for r in rows]);best=task.min();ts=selected(task,1e-12)
    f=selected([float(r['source_feature_mse']) for r in rows],1e-8)
    result={}
    for name,column in SCORES.items():
        idx=np.arange(3) if column is None else selected([float(r[column]) for r in rows],1e-12 if name in ('task_optimal','readout_consistency') else 1e-8)
        result[name+'_task_error']=float(task[idx].mean());result[name+'_task_regret']=float(task[idx].mean()-best)
        result[name+'_task_optimal']=float(np.isin(idx,ts).mean())
    for column in COMPONENTS:
        a=np.array([float(r[column]) for r in rows]);result[column+'_mean']=float(a.mean())
        result[column+'_range']=float(np.ptp(a))
        result[column+'_full_minus_task']=float(a[f].mean()-a[ts].mean())
    idx=selected([float(r['total']) for r in rows],1e-8)
    result['full_float64_choice_differs']=float(set(idx)!=set(f))
    result['feature_flat']=float(np.ptp([float(r['source_feature_mse']) for r in rows])<=1e-8)
    result['task_flat']=float(np.ptp(task)<=1e-12)
    conflict=null_override=basis_null_override=visible_conflict=0
    for i,j in ((0,1),(0,2),(1,2)):
        df=float(rows[i]['source_feature_mse'])-float(rows[j]['source_feature_mse']);dt=task[i]-task[j]
        if abs(df)<=1e-8 or abs(dt)<=1e-12 or df*dt>=0:continue
        conflict+=1
        dn=float(rows[i]['null'])-float(rows[j]['null']);dv=float(rows[i]['visible'])-float(rows[j]['visible'])
        dbn=float(rows[i]['basis_null'])-float(rows[j]['basis_null']);dbs=float(rows[i]['basis_sensitive'])-float(rows[j]['basis_sensitive'])
        null_override+=int(dn*dt<0 and dv*dt>0 and abs(dn)>1e-8 and abs(dv)>1e-8)
        basis_null_override+=int(dbn*dt<0 and dbs*dt>0 and abs(dbn)>1e-8 and abs(dbs)>1e-8)
        visible_conflict+=int(dv*dt<0 and abs(dv)>1e-8)
    result.update(order_conflict=conflict/3,null_overrides_visible=null_override/3,
                  basis_null_overrides_sensitive=basis_null_override/3,visible_itself_conflicts=visible_conflict/3)
    return result
