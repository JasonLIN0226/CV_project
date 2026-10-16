import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SRC=HERE/'native_completed7426_7435/analysis_outputs'
OUT=HERE/'report';OUT.mkdir(exist_ok=True)
summary=json.loads((SRC/'summary.json').read_text())
with (SRC/'record_summary.csv').open() as f:rows=list(csv.DictReader(f))
policies=['FIFO','Reservoir','FAR_current','Archive_reference']
labels=['FIFO','Reservoir','FAR current','Archive\nreference']
clips=list(dict.fromkeys(int(r['clip_index']) for r in rows))
scene=np.array([[np.mean([float(r['dreamsim_tail']) for r in rows if r['policy']==p and int(r['clip_index'])==i]) for i in clips] for p in policies])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(6.5,2.6),layout='constrained')
for j in range(4):
    ax.bar(j,scene[j].mean(),width=.62,color=['white','.8','.45','white'][j],edgecolor='black',linewidth=1,hatch=['','//','','..'][j])
    ax.scatter(j+np.linspace(-.18,.18,8),scene[j],s=16,facecolors='white',edgecolors='black',linewidths=.7,zorder=4)
ax.set_xticks(range(4),labels);ax.set_ylabel('DreamSim error\nLower is better');ax.set_ylim(0,max(.5,float(scene.max())*1.08));ax.grid(axis='y',color='.88',linewidth=.5);ax.set_axisbelow(True)
for ext in ('png','svg'):fig.savefig(OUT/('prediction_error.'+ext),dpi=230)
plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,2.15),layout='constrained')
for y,p in [(1,'FIFO'),(0,'Reservoir')]:
    t=summary['paired_comparisons'][p]['dreamsim_tail'];lo,hi=t['paired_scene_bootstrap95'];m=t['mean_improvement']
    ax.plot([lo,hi],[y,y],color='black',lw=1.7);ax.scatter([m],[y],color='black',s=30,zorder=3)
ax.axvline(0,color='black',ls='--',lw=.9);ax.set_yticks([1,0],['FAR versus FIFO','FAR versus Reservoir']);ax.set_ylim(-.5,1.5);ax.set_xlim(-.24,.045);ax.set_xlabel('DreamSim improvement  baseline minus FAR\nPositive values favor FAR');ax.grid(axis='x',color='.88',lw=.5);ax.set_axisbelow(True)
for ext in ('png','svg'):fig.savefig(OUT/('paired_improvement.'+ext),dpi=230)
plt.close(fig)
data={'independent_scenes':8,'scene_means':{p:scene[j].tolist() for j,p in enumerate(policies)},'clips':clips,'primary':'dreamsim_tail','positive_improvement':'baseline minus FAR_current','Reservoir_error_reduction_vs_FIFO_percent':100*(1-summary['policies']['Reservoir']['dreamsim_tail']['mean']/summary['policies']['FIFO']['dreamsim_tail']['mean'])}
(OUT/'figure_data.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'figures':2,'scene_count':8,'descriptive_Reservoir_error_reduction_percent':data['Reservoir_error_reduction_vs_FIFO_percent']}))
