import datetime
import hashlib
import json
import stat
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
HERE=BASE/'far_phase1'
NATIVE=HERE/'native_completed7426_7435'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
summary=json.loads((NATIVE/'analysis_outputs/summary.json').read_text())
proof=json.loads((NATIVE/'final_evidence/final_native_audit.json').read_text())
transfer=json.loads((NATIVE/'transfer_integrity.json').read_text())
assert proof['status'].startswith('PASS_INDEPENDENT_FINAL_NATIVE') and len(transfer['files'])==25
doc=HERE/'report/Memory_Retention_Phase1_Report.docx'
assert doc.exists() and all((HERE/f'report/rendered_final/page-{n}.png').exists() for n in [1,2])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
qa={'utc':now,'docx':str(doc),'docx_sha256':sha(doc),'pages':2,'all_pages_visually_inspected':True,
    'blank_page_removed_and_latest_render_verified':True,'figures':2,'black_and_white':True,
    'conclusion_last':True,'Google_Doc_updated':False,'Google_Doc_reason':'Existing focus/write access limitation not re-probed; prior document preserved.'}
(HERE/'report/visual_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
result={'utc':now,'status':'COMPLETE_VERIFIED_SMALL_PILOT_FAR_CURRENT_SUPERIORITY_NOT_SUPPORTED',
        'scope':'Direct FAR to Phase1 pilot only; full FAR reproduction cancelled; original Phase1 GO unchanged',
        'science':{'job':7426,'state':'COMPLETED','exit_code':'0:0','start_utc':'2026-10-14T22:38:12','end_utc':'2026-10-15T00:25:02','wall_seconds':6410,'GPU':'one A100-SXM4-40GB','course':'csci_ga_2271-2026fa','records':96,'steps':6996,'scenes':8,'seeds':[0,1,2]},
        'analysis':{'job':7435,'state':'COMPLETED','exit_code':'0:0','start_utc':'2026-10-15T00:26:43','end_utc':'2026-10-15T00:28:36','no_GPU_or_forward':True},
        'primary':{p:summary['policies'][p]['dreamsim_tail'] for p in summary['policies']},
        'primary_paired':{p:summary['paired_comparisons'][p]['dreamsim_tail'] for p in ['FIFO','Reservoir']},
        'final_native_proof_sha256':sha(NATIVE/'final_evidence/final_native_audit.json'),
        'independent_asset_source_raw_stats_storage_replay':proof['status'],
        'exact_Reservoir_algorithm_R_trace_replay':True,'analysis_own4_source_verified':True,
        'transfer':{'files':25,'bytes':207032,'three_SHA_equal':True,'path':'far_phase1/native_completed7426_7435/transfer_integrity.json'},
        'control7451':{'single_necessary_final_evidence_connection':True,'GPUs':0,'native_start_utc':'2026-10-15T04:53:16','host':'b-9-10','ip':'10.144.0.25','alias':'phase1-far-finalevidence7451','fingerprint':'SHA256:HhvJY47ecla2LnA5WYQxnYOiJHaDxrvhew1Ad2n3ng8','identity_from_authenticated_metadata_Notebook_and_Torch_keyscan_exact':True,'scancel_returncode':0,'SSH_disconnect_during_self_release':True,'normal_portal_terminal':'Completed','post_shutdown_native_actor_exit_not_read':True,'ended_alias_do_not_reconnect':True},
        'new_training_or_GPU_rerun':False,'report':qa}
(HERE/'final_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
header='''## FAR适配Phase1小实验最终结果（2026-10-15）

用户要求的独立小实验已经完成并独立核验。7426原生COMPLETED0:0，2026-10-14 22:38:12至2026-10-15 00:25:02UTC，实际一A100/课程csci_ga_2271-2026fa/4科学线程60G，wall6410秒。8独立场景、ABA/ABCA各4、3seed、4条件共96完整记录/6996目标步。7435独立CPU审计原生COMPLETED0:0，00:26:43至00:28:36UTC。断网只阻挡本机取回，没有重训或重跑。

主DreamSim返程最后四分之一（越低越好）：FIFO0.3326988022，trueReservoir0.1794886462，FAR_current0.3389438471，Archive_reference0.1953896387。FAR相对FIFO改善为-0.006245045，8场景配对bootstrap95%[-0.024792283,0.012272273]；相对Reservoir为-0.159455201，95%[-0.217832298,-0.099818787]。固定优越性标准未满足，不能称原Phase1 GO。Archive非等容量，仅诊断；这不是原FAR全论文或I-JEPA复现。

当前169源文件（science6/upstream159/analysis4）、选中16数据JSON/11模型/27指标缓存全SHA及稳定性再次原生核验，96原始数组有限/配对帧轴与CSV一致，18配对统计和完整Reservoir Algorithm R实际选择独立重放全部通过。最终原生证明SHA5b8628ecf49602070e0b703cfe94588759688a64ce1b606f4dce9b997a513d72。25明确compact文件共207032B原生before/local/afterSHA完全一致；weights/features/arrays/rawNPZ/大trace仍留HPC。

事后ID-only诊断：返程末尾past memory中真实外出观测占比Reservoir89.3%、FAR4.7%、FIFO0%。该诊断支持继续调查旧证据丢失，不是因果证明；各策略generated current允许分化，未改变主指标/选择样本/重跑。科学冻结源未改，sealed/fresh9137未开，旧SPACE与原结果门槛不变。

网络恢复实际核验：批准Mac key exact、正常OOD DNS恢复、existing login成功、Torch master5153和hostname实际验证，pendinghuman=null。唯一必要0GPU取证7451用途完成，native scancel实际return0，SSH随自身释放断开；正常门户最终Completed。未再连接其Ended alias/IP25，未冒称读到关停后的native actor/exit。现有本chat仅保留72Sessions并handoff。当前本任务没有活跃science/metadata/control，不自动新增实验。

最终结果：far_phase1/final_result.json；紧凑完整证据：far_phase1/native_completed7426_7435；两页英文黑白报告及两图：far_phase1/report/Memory_Retention_Phase1_Report.docx，全部页面已渲染目检。Google Doc仍保留旧内容；原focus/403无新恢复证据，未重试、未改权限。实验监督已无继续轮询需要；不得按旧heartbeat重开科学或control。
'''
for rel in ['runs/phase1_v4_diagnostics/runtime_status.json','far_phase1/workflow_state.json','runs/phase1_v6_revision/current_handoff.json']:
    f=BASE/rel;assert not getattr(f.stat(),'st_flags',0)&getattr(stat,'SF_DATALESS',0x40000000)
    d=json.loads(f.read_text());d.update(stage='FAR_PHASE1_PILOT_COMPLETE_VERIFIED_NO_RETENTION_GAIN',updated_utc=now,last_observed_utc=proof['utc'],pending_human_action=None,active_science_jobs=[],active_data_jobs=[],active_data_job=None,pending_metadata_job=None,temporary_connection=None,current_native_route=None,far_native_route=None,current_native_route_available=False,native_current_route_available=False,sole_current_science_job_id=None,current_phase1_GPU_job_id=None,actual_final_states_verified=True,active_science_jobs_currently_unknown=False,current_transfer_integrity='far_phase1/native_completed7426_7435/transfer_integrity.json',final_phase1_result='far_phase1/final_result.json',final_phase1_report=qa,completed_science_job_id=7426,control7451_release=result['control7451'])
    for key in ['sole_science_job','current_sole_science_job','latest_recovery_progress']:
        if isinstance(d.get(key),dict):d[key].update(native_state='COMPLETED',current_native_state='COMPLETED',native_allocation_state='COMPLETED; batch COMPLETED',exit_code='0:0',actual_end_utc='2026-10-15T00:25:02',completed_records=96,completed_steps=6996,completed_target_steps=6996,current_state_verified=True,native_state_as_of_utc=proof['utc'])
    if isinstance(d.get('phase1_analysis_handoff'),dict):d['phase1_analysis_handoff'].update(native_state='COMPLETED',current_native_state='COMPLETED',exit_code='0:0',actual_aggregation_not_started=False,actual_summary_accepted=True,analysis_own_source_verified=True)
    d['next_work']='Fixed pilot and independent native/transfer/statistical/storage audit complete. Report2pages rendered and inspected; no further science, controls or source changes. Full FAR and SPACE remain stopped; original GO unchanged. GoogleDoc not updated due existing access limitation; local report ready. End monitoring instead of repeating completed evidence.'
    f.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
for f in [BASE/'README.md',BASE.parent/'CV_HPC_CONNECTION_GUIDE.md']:
    s=f.read_text();a=s.find('## 当前Phase1核验网络依赖');b=s.find('\n\n## ',a+3)
    assert a>=0 and b>a
    f.write_text(s[:a]+header+'\n\n'+s[b+2:])
f=HERE/'current_handoff.txt';s=f.read_text();marker='继续本聊天用户直接授权的FAR-to-Phase1小实验';tail=s[s.index(marker):] if marker in s else s
f.write_text(header+'\n\n历史约束仍有效，后文运行状态均以本段完成记录为准。\n\n'+tail)
f=BASE/'runs/phase1_v5_gate/overnight_report.md';s=f.read_text();cut=s.find('\n')
f.write_text(s[:cut+1]+'\n'+header+'\n\n'+s[cut+1:])
print('Final result and all current handoffs updated; frozen scientific sources unchanged.')
