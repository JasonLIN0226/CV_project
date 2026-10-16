from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;A=D/'native_completed/analysis/output/job_7456';OUT=D/'report';OUT.mkdir(exist_ok=True)
S=json.loads((A/'summary.json').read_text());rows=list(csv.DictReader((A/'record_summary.csv').open()))
F=json.loads((D/'native_completed/final/final_native_audit.json').read_text());assert F['status']=='PASS_FINAL_INDEPENDENT_144_RECORD_AUDIT'
order=['Current_only','Recent_only','FIFO','Reservoir','FAR_current','Archive_reference']
labels=['Current only','Recent only','FIFO with FAR reader','Reservoir','FAR current','Archive reference']
clips=list(dict.fromkeys(r['clip_index'] for r in rows));scene={p:[float(np.mean([float(r['dreamsim_tail']) for r in rows if r['clip_index']==idx and r['policy']==p])) for idx in clips] for p in order}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(7.1,2.85));y=np.arange(6)
means=[S['policies'][p]['dreamsim_tail']['mean'] for p in order]
ax.barh(y,means,height=.58,color=['white','0.92','0.84','0.60','0.76','white'],edgecolor='black',linewidth=.8)
for i,p in enumerate(order):ax.scatter(scene[p],i+np.linspace(-.15,.15,8),s=12,c='black',edgecolors='white',linewidths=.25,zorder=3)
for i,v in enumerate(means):ax.text(.57,i,f'{v:.4f}',va='center',ha='right',fontsize=9)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,.59);ax.set_xticks(np.arange(0,.6,.1));ax.set_xlabel('Final quarter DreamSim error  lower is better');ax.tick_params(axis='y',length=0);ax.grid(axis='x',color='.88',linewidth=.5);ax.set_axisbelow(True)
fig.subplots_adjust(left=.265,right=.98,bottom=.21,top=.97)
for ext in ['png','svg']:fig.savefig(OUT/('six_condition_error.'+ext),dpi=240)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7.1,2.0));comparisons=['Reservoir_over_Current_only','Reservoir_over_Recent_only'];ys=[1,0]
for yy,n in zip(ys,comparisons):
 q=S['paired_comparisons'][n]['dreamsim_tail'];lo,hi=q['paired_scene_bootstrap95'];v=q['mean_improvement']
 ax.scatter(q['scene_improvements'],yy+np.linspace(-.13,.13,8),s=17,c='.55',zorder=2)
 ax.errorbar(v,yy,xerr=[[v-lo],[hi-v]],fmt='D',color='black',markersize=5,capsize=4,lw=1.25,zorder=4)
ax.axvline(0,color='.35',ls='--',lw=.8);ax.set_yticks(ys,['Reservoir vs current only','Reservoir vs recent only']);ax.set_ylim(-.5,1.5);ax.set_xlim(-.025,.40);ax.set_xticks([0,.1,.2,.3,.4]);ax.set_xlabel('Baseline error minus Reservoir error  positive favors Reservoir');ax.grid(axis='x',color='.9',lw=.5);ax.tick_params(axis='y',length=0)
fig.subplots_adjust(left=.30,right=.98,bottom=.32,top=.95)
for ext in ['png','svg']:fig.savefig(OUT/('primary_paired_effects.'+ext),dpi=240)
plt.close(fig)
(OUT/'figure_data.json').write_text(json.dumps({'policy_order':order,'means':dict(zip(order,means)),'scene_order':clips,'scene_mean_DreamSim_tail':scene,'primary_comparisons':{n:S['paired_comparisons'][n]['dreamsim_tail'] for n in comparisons}},indent=2)+'\n')
print('Two standalone scientific figures created from independently verified data.')
