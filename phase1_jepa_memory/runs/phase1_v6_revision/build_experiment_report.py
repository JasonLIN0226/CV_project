"""Render actual compact experiment evidence; missing stages remain missing."""
from pathlib import Path
import csv, datetime, hashlib, json
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'native7251_7268'
OLD=ROOT.parent/'phase1_v5_real_evaluation/native_completed7209_diagnostic7226'
def read(name):return json.loads((DATA/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(x):return f'{x:.8f}'
models=read('all_models_completion.json')
health=read('visual_health.json')
fixed=read('fixed_old_reference_comparison.json')
replay=read('root_health_scalar_replay_7268_20261014T0144.json')
fixed_replay=read('root_fixed_old_reference_replay_7268_20261014T0144.json')
PUBLIC=ROOT/'public7268'
full=read('diagnostic_report.json') if (DATA/'diagnostic_report.json').exists() else json.loads((PUBLIC/'diagnostic_report.json').read_text()) if (PUBLIC/'diagnostic_report.json').exists() else None
B=read('deletion_summary.json') if (DATA/'deletion_summary.json').exists() else json.loads((PUBLIC/'deletion_summary.json').read_text()) if (PUBLIC/'deletion_summary.json').exists() else None
C=read('replay_participant_summary.json') if (DATA/'replay_participant_summary.json').exists() else json.loads((PUBLIC/'replay_participant_summary.json').read_text()) if (PUBLIC/'replay_participant_summary.json').exists() else None
final=read('root_final_audit_7268.json') if (DATA/'root_final_audit_7268.json').exists() else None
new_rows={s:list(csv.DictReader((DATA/f'seed_{s}_training.csv').open())) for s in (17,29,43)}
old_rows={s:list(csv.DictReader((OLD/f'seed_{s}_training.csv').open())) for s in (17,29,43)}
primary=sum(check['supported'][0] for h,group in health['full_vs_recent_paired_checks'].items() if int(h) in (16,64) for check in group.values())
assert primary==6 and fixed['primary_checks_passed']==9

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(14,8.4),constrained_layout=True)
for col,s in enumerate((17,29,43)):
    ax=axes[0,col]
    for rows,color,label in ((old_rows[s],'#8f98a4','v5 wd=.05'),(new_rows[s],'#246e9f','v6 wd=.20')):
        ep=[int(r['epoch']) for r in rows]
        ax.plot(ep,[float(r['train_full_mse']) for r in rows],ls='--',color=color,label=label+' train')
        ax.plot(ep,[float(r['validation_full_mse']) for r in rows],color=color,label=label+' validation')
    best=next(x['best_epoch'] for x in models['actual_seeds'] if x['seed']==s)
    val=float(new_rows[s][best-1]['validation_full_mse']);ax.scatter([best],[val],s=38,color='#246e9f',zorder=5)
    ax.set(title=f'Seed {s}: all 30 epochs; selected best {best}',xlabel='Epoch',ylabel='Query-weighted normalized MSE')
    ax.legend(fontsize=7.5)
for col,(h,title) in enumerate(((16,'History benefit at 8-second prediction horizon'),(64,'History benefit at 32-second prediction horizon'),(0,'Fixed old-reference checks'))):
    ax=axes[1,col];labels=[];point=[];low=[];high=[];colors=[]
    if h:
        for label,x in health['full_vs_recent_paired_checks'][str(h)].items():
            labels.append(label);point.append(x['point'][0]);low.append(x['ci95_lower'][0]);high.append(x['ci95_upper'][0]);colors.append('#23806a' if x['supported'][0] else '#b14a4a')
    else:
        for key,group in fixed['checks'].items():
            if not key.endswith(('h16','h64')):continue
            for label,x in group.items():
                labels.append(key.replace('old_','').replace('_vs_new_full_',' / ')+' / '+label);point.append(x['point'][0]);low.append(x['ci95_lower'][0]);high.append(x['ci95_upper'][0]);colors.append('#23806a' if x['supported'][0] else '#b14a4a')
    for i,(p,l,u,c) in enumerate(zip(point,low,high,colors)):ax.errorbar(p,i,xerr=[[p-l],[u-p]],fmt='o',color=c,capsize=3,ms=4)
    ax.axvline(0,color='#666',lw=1);ax.set_yticks(range(len(labels)),labels,fontsize=7.5 if not h else 10);ax.invert_yaxis();ax.set(title=title,xlabel='Baseline MSE minus full-history MSE; 95% CI')
    if not h:
        ax.set_xticks([-.005,0,.005,.010,.015]);ax.tick_params(axis='x',labelsize=8)
        ax.set_xlabel('Old-reference minus v6 full MSE; 95% CI')
fig.suptitle('Actual v6 GPU training and completed stage A: original health 6/8, fixed references 9/16',fontsize=14)
figure=ROOT/'training7251_health7268.png';fig.savefig(figure,dpi=160);plt.close(fig)

now=datetime.datetime.now(datetime.timezone.utc).astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M EDT')
complete=bool(full and final)
lines=[f'# v6 GPU 实验报告：实际结果与未完成项',f'更新时间：{now}。'+('固定视觉计算及最终原生数值审计完成。' if complete else '全部固定视觉计算已完成；最终原生独立数值审计及语义评估未完成。'),
'**三组 GPU 训练7251已全部COMPLETED0:0，各30轮/9690步；正式推理前和root独立重算的六个实际模型证明一致。原健康门槛6/8、固定旧参考比较9/16，均REJECTED。'+('B/C完整结果如下。' if full else 'B/C仍在HPC自动运行。')+' 标签精确相位仍未证明，语义任务评估未完成，Phase1未通过。**',
'## 本轮改变与失败解释',
'唯一科学改动为AdamW weight_decay从0.05增加到0.20。原架构、官方冻结编码器、完整缓存、训练数据、三个种子、30轮、学习率/调度和最早等权validation best规则保持。7209 train-only归一化和128rank输入/输出基底逐字节复用，未重新拟合。',
'上一轮的训练误差持续下降，而验证误差在5/6/7轮达到最低后上升，支持泛化落差假设；增加正则化是一次固定单因素检验，不代表已确认唯一失败机制。新best验证MSE略有下降，但本轮最早best仍为5/6/7。',
'本轮h16/h64的seed29历史收益95%区间均跨零，原要求是每个种子与pooled都通过，所以6/8仍失败。seed43恢复为正区间，不能替代seed29。与上一轮固定近期/完整历史预测比较也只有9/16支持改善，不能据pooled或平均误差较小称稳定采用。标签缺口独立于这些无标签视觉失败；解锁/启动问题也不用于解释科学失败。',
'## 作业、资源与范围',
'| 阶段 | job | 实际状态/时间UTC | 内容 |','|---|---:|---|---|',
'| 复用真实编码/缓存审计 | 7187/7188 | 已完整COMPLETED0:0 | 全78录像/50688时点与完整缓存证明 |',
'| 本轮GPU训练 | 7251 | COMPLETED0:0；00:47:36–01:37:50 | 一A100-SXM4-40GB、4科学线程、60G、原4h预算；实际50分14秒 |',
'| 六模型证明与自动评估交接 | 7253 | COMPLETED0:0；01:39:07–01:39:30 | 来源/数值审计后冻结协议并自动提交唯一7268 |',
'| 冻结GPU模型固定评估 | 7268 | '+('原生COMPLETED0:0；01:40:37–02:08:47，28分10秒' if final else '正常门户COMPLETED；01:40:37–02:08:47，最终原生终态待核验' if full else 'RUNNING；01:40:37起')+' | 原4科学线程/16G/零GPU/2h；原全部A/B/C及旧标量比较 |',
'冻结GPU模型的CPU评估属于原设计，不是CPU模型训练替代。未执行CPU编码/reader训练fallback，未发送支持邮件或访问申请，未增加科学预算、删seed或样本。',
'固定开发数据为58train/20validation RGB、198episodes、50688时点。train136episodes/34816观察/5168查询，validation62episodes/15872观察/2356查询；封存18RGB/9人及fresh9137确认未打开。',
'官方I-JEPA revision f157467ea509bc356ff9f61fd3c0d840eec5e04e，float32、grid2、batch2、TF32关闭；输出5120D。Reader hidden256、2层4heads、FFN1024、dropout.1、historydropout.15、recent8/capacity16、h1/4/16/64、recent/FIFO/Reservoir/full；1779072参数。AdamW3e-4/wd.20、5%warmup+cosine。',
'## 完整训练与选模',
'| seed | epochs/steps | 最早best | v5 best val MSE | v6 best val MSE | v6末轮train MSE | v6末轮val MSE |','|---|---:|---:|---:|---:|---:|---:|']
for s in (17,29,43):
    best=next(x['best_epoch'] for x in models['actual_seeds'] if x['seed']==s)
    lines.append(f'| {s} | 30/9690 | {best} | {min(float(r["validation_full_mse"]) for r in old_rows[s]):.8f} | {float(new_rows[s][best-1]["validation_full_mse"]):.8f} | {float(new_rows[s][-1]["train_full_mse"]):.8f} | {float(new_rows[s][-1]["validation_full_mse"]):.8f} |')
lines += ['选模统计按查询等权；下方科学评估按参与者分层等权，两种均值不能直接相减。六个best/latest在任何评估推理前已核验有限值、source/cache/fit/config/RNG、optimizer实际wd.20、原schedule、完整轮数/步数和CSV最早best。',f'![本轮与上一轮训练及实际健康/固定参考区间]({figure})',
'## A：完整预测健康评估',
'完整113088条标量：3seeds×2356validation queries×4conditions×4h；完整5120D normalized MSE、persistence与固定基底truefuture oracle。query→episode→recording→participant等权，10验证参与者，10000次seed0 bootstrap，双侧95%CI。原主门槛要求h16和h64的3seed+pooled全部正区间；实际6/8，REJECTED。h1/h4各0/4，完整保留。',
'| h/秒 | seed/汇总 | Recent−Full MSE | 95%CI | 判定 |','|---|---|---:|---|---|']
for h,group in health['full_vs_recent_paired_checks'].items():
    for label,x in group.items():lines.append(f'| {h}/{int(h)/2:g}s | {label} | {fmt(x["point"][0])} | [{fmt(x["ci95_lower"][0])}, {fmt(x["ci95_upper"][0])}] | '+('PASS' if x['supported'][0] else 'FAIL')+' |')
lines += ['### 参与者等权绝对MSE', '| seed | h | recent | FIFO | Reservoir | full | persistence | fixedbasis oracle |','|---|---:|---:|---:|---:|---:|---:|---:|']
for s,hs in replay['participant_equal_weight_mse'].items():
    for h,p in hs.items():lines.append(f'| {s} | {h} | '+' | '.join(fmt(p[k]['mse']) for k in ('recent','FIFO','Reservoir','full'))+f' | {fmt(p["full"]["persistence_mse"])} | {fmt(p["full"]["basis_oracle_mse"])} |')
lines += ['truefuture oracle仅是诊断下界，不可部署、不能证明可学习性。root使用原113088标量独立重放原分层均值/CI，最大差'+str(replay['max_difference_from_executed_summary'])+'；全部判断一致，没有新的模型推理/拟合/训练。',
'## 预先固定的旧参考比较',
'比较同一开发查询上的new full与旧7226已执行的recent/full标量，未重新推理旧模型；所有113088个seed/recording/episode/time/policy/horizon键严格一致。h16/h64×2参考×3seed+pooled共16项，实际9/16，REJECTED。原A/B规则保持，不用旧参考结果替代原门槛。',
'| 固定比较 | seed/汇总 | Old−NewFull MSE | 95%CI | 判定 |','|---|---|---:|---|---|']
for key,group in fixed['checks'].items():
    for label,x in group.items():lines.append(f'| {key} | {label} | {fmt(x["point"][0])} | [{fmt(x["ci95_lower"][0])}, {fmt(x["ci95_upper"][0])}] | '+('PASS' if x['supported'][0] else 'FAIL')+' |')
lines += ['root独立旧/新原标量重放最大差'+str(fixed_replay['max_difference'])+'，全部32判断一致。旧参考SHA '+fixed['old_raw_sha256']+'。',
'## B：全部17删除分支与32共同延续']
if B:
    lines += [('已原生取回并复核。' if final else '来源：已认证正常门户View紧凑摘要；最终原生三SHA取回与原始标量独立复核尚待连接恢复。'),
        '全部624states=78recordings×2source policies×4固定均匀状态，每状态3models×17primary删除×32共同Reservoir延续。共同oracle按.5instant四h均值+.5future32四h均值选一次，同一选择用于h16/h64，exacttie删最早时间，不能逐h重选。',
        '科学状态：'+B['visual_joint_and_h16_h64_status']+'；48项必要检查实际47项支持。失败项为Reservoir来源/FIFO基线/seed43/h64，点收益0.0000736861，95%CI[-0.0000007579,0.0001566059]。不能忽略区间跨零或逐horizon重选oracle。',
        '| source/基线 | seed | 指标 | 点收益 | 95%CI | 判定 |','|---|---|---|---:|---|---|']
    for group,checks in B['checks'].items():
        for label,x in checks.items():
            for i,metric in enumerate(('joint','h16','h64')):lines.append(f'| {group} | {label} | {metric} | {fmt(x["point"][i])} | [{fmt(x["ci95_lower"][i])}, {fmt(x["ci95_upper"][i])}] | '+('PASS' if x['supported'][i] else 'FAIL')+' |')
    lines += ['Oracle只报告删除可行性上界，不可部署、不能证明learnability。']
else:lines += ['服务器正在自动完成原全量624states；未提交重复评估，不把partial当最终门槛。']
lines += ['## C：不可逆持久载荷重放']
if C:
    lines += [('来源：原生最终取回与独立审计。' if final else '来源：正常已认证门户View紧凑执行报告/摘要，非最终原生独立取证。'),
        '全198episodes×3模型×FIFO/Reservoir/Uniform×capacity16主及8/32次×38queries×4h，实际812592标量、456192行保留轨迹、1782条完整重放轨迹。当前原始标量的参与者均值独立重放最大差为0；当前轨迹逐项固定FIFO/Algorithm R/Uniform随机状态、容量、chronology及no-reentry独立重放通过。没有外部archive取回已删载荷。',
        'root最终独立固定RNG/容量/chronology/no-reentry核验：'+('PASS' if final else '尚待原生取回与独立复核；不冒称完成')+'。',
        '| seed | split | policy | capacity | h | participant等权MSE |','|---|---|---|---:|---:|---:|']
    for key,values in C['participant_means'].items():
        s,split,policy,capacity,h=key.split('|');lines.append(f'| {s} | {split} | {policy} | {capacity} | {h} | {fmt(float(np.mean(list(values.values()))))} |')
else:lines += ['C同一作业在B后自动接续。完整812592标量/456192轨迹尚未取证，不声称完整重放已完成。']
lines += ['## 标签对齐：已做工作与剩余依赖',
'已核验固定官方HF完整README、0based/inclusive消费代码、官方action-recognition的37个文本Git blob及13次历史变更；未找到原30fps抽帧生产脚本。消费代码不能单独决定原RGB抽帧相位。[官方基线固定版本](https://github.com/assembly-101/assembly101-action-recognition/tree/00d3c3bb3af75ce38ae406614a7cb034aecdebc0)。',
'固定官方HF发布的5763条路径元数据已完整检查。进一步在HPC用受限range读取72GB AssemblyPoses.zip的658938字节目录，发现3381项中包含逐录像timestamp。仅读取预先批准的9011-c13b…160915开发录像419325字节时间戳元数据，keys0..21275连续，首9881.016/末10235.599秒；未读取pose/像素/标签/封存数据，完整member留HPC。',
'用户完成本机解锁后，已核验同一批准公钥并恢复原生取证。对同批准C10095 RGB MP4已实际读取container/stream元数据，前后文件SHA稳定：60fps、21277帧、start_time=0、time_base=1/15360；官方timestamp有21276项，其相对60fps时钟残差为[-0.000333333334,0.000666666667]秒。没有嵌入capture timecode；时间戳与视频还存在一帧计数差。这些元数据不能证明哪一个raw frame生成30fps annotation-frame0。没有读取像素/pose/标签数组或封存数据，也没有选择标签join。',
'另对官方账号7个公开仓库完成目录盘点，新核验download-scripts固定版本01835b08aebe563148bea21a7be7a7674bc432cb的3个文本源码及action-anticipation固定版本7ae5cf6328bc4413a279b2664318dafe2845ce36的10个文本源码，Git blob逐项一致。下载脚本只转移原视频；anticipation消费fps=30的已有特征/标签，没有原RGB→标注抽帧producer。这是有界源码审计，不是所有未公开生产资料不存在的证明。',
'官方旧pose链接正常浏览器实际AccessDenied，没有发送访问申请。另核验27个官方AssemblyHands toolkit文本Git blob；其visualization/video_visualizer.py声明ANNOT_FPS60、RECTIFIED_VIDEO_FPS30，针对不同手姿标注与rectified视频，不能嫁接为原Assembly101动作抽帧生产证明。[固定源码](https://github.com/facebookresearch/assemblyhands-toolkit/blob/8d4b7f1bce3f99874514199aae7504f78ebf3517/visualization/video_visualizer.py)。',
'全50688时点在phase0/1/60两个条件假设下的verb union差异为train1234/34816、validation493/15872，合计1727/50688=3.407%。这不是相位证明，不能忽略或挑一个看似更好的join。',
'24verb readout/NLL、within-activity drift、switch response、different-activity donor和0/1/2/4/8/16恢复均UNAVAILABLE。没有制造供体或用无分层视觉稳定性替代语义门槛。下一项必要证据是权威生产脚本或带原始索引的实际标注帧，可据其与批准开发RGB做精确匹配；现有timestamp及消费者不足以替代。继续只读调查新来源，不能原样重复相位搜索/审计或声称标签已解决。',
'## 来源与最终核验',
'28明确compact JSON/小trainingCSV/日志原生before/local/afterSHA一致，见native7251_7268/transfer_integrity_final.json。另3份小型官方目录/timestamp结构/RGB时钟摘要三SHA一致，见primary_sources/timestamp_metadata_native_transfer_integrity.json。七组冻结源、78cache前后SHA、全部6模型数值来源与fit字节已root重算严格等于正式执行证明。大raw CSV/trace/features/weights/videos/labels数组未下载到Mac。',
'| 证据 | SHA256 |','|---|---|']
for name in ('training_report.json','all_models_completion.json','protocol.json','source_lock_diagnostic.json','visual_health.json','fixed_old_reference_comparison.json','root_health_scalar_replay_7268_20261014T0144.json','root_fixed_old_reference_replay_7268_20261014T0144.json','root_monitor_all6_health_7268_20261014T0144.json'):
    lines.append(f'| native7251_7268/{name} | {sha(DATA/name)} |')
if full:
    lines += ['完整输出哈希如下；当前HPC全部输出字节、大小和SHA已root重新核验等于执行报告，大CSV/trace留HPC；紧凑摘要另外完成原生三SHA传输。','| 已执行输出 | bytes | 执行报告声明SHA256 |','|---|---:|---|']
    for name,value in full['files'].items():lines.append(f'| {name} | {value["bytes"]} | {value["sha256"]} |')
if final:
    lines += ['最终Slurm/全部文件大小SHA/31824候选数/48CI/812592重放标量/456192固定RNG轨迹已原生独立重放证明。',f'最终审计SHA：{sha(DATA/"root_final_audit_7268.json")}。']
else:lines += ['最终Slurm、B/C当前原始标量与轨迹独立复核、最终compact三SHA取回尚待接续；不能用公开stdout代替这些检查。']
lines += ['## 结论与继续执行',
'本次固定增加正则化的试验未达到原采用门槛；所有种子和失败项保留。GPU训练与全部可用视觉评估、最终模型/源码/缓存/原始标量/保留轨迹取证已完成。临时原生取证连接7271用途完成后由agent释放，实际CANCELLED by3850228/0:0（02:16:37–02:35:05UTC），正常门户Completed；没有活跃科学任务或认证待办。标签语义依赖仍未解决，不能把可用视觉计算完整称作全部语义实验完成。原Gate/Motion REJECTED保持，不能由真实分支救回；sealed/fresh9137确认不启动。继续有依据的原始生产来源调查，不原样重训失败trial、放宽门槛或扫描参数。']
blocks=[]
for line in lines:
    if line.startswith('|') and blocks and blocks[-1].startswith('|'):
        blocks[-1]+='\n'+line
    else:blocks.append(line)
(ROOT/'experiment_report.md').write_text('\n\n'.join(blocks)+'\n')
print(json.dumps({'report':str(ROOT/'experiment_report.md'),'sha256':sha(ROOT/'experiment_report.md'),'figure':str(figure),'complete_available_visual_and_final_audit':complete,'health_passed':primary,'fixed_reference_passed':fixed['primary_checks_passed']}))
