"""Compact editable report, generated only from transferred audited results."""
from pathlib import Path
import csv,hashlib,json,sys
import subprocess
from reportlab.graphics.shapes import Drawing,String,Line,Circle
from reportlab.graphics import renderSVG
from reportlab.lib import colors
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(native):
    native=Path(native);transfer=json.loads((native/'transfer_integrity.json').read_text())
    result_dir=native if (native/'summary.json').exists() else native/'output/job_7590'
    summary=json.loads((result_dir/'summary.json').read_text());audit=json.loads((result_dir/'independent_audit.json').read_text())
    assert transfer['native_before_local_native_after_all_match'] and audit['status']=='INDEPENDENT_SCALAR_AUDIT_PASS'
    assert sha(result_dir/'summary.json')==audit['summary_sha256']
    report=BASE/'report';report.mkdir(exist_ok=True)
    drawing=Drawing(504,215)
    for task,origin,label in [('gate',60,'Gate cue'),('motion',315,'Motion start and end')]:
        width=175;lo=-.30;hi=1.
        xp=lambda value:origin+(value-lo)/(hi-lo)*width
        drawing.add(String(origin-44,199,label,fontName='Helvetica-Bold',fontSize=10))
        drawing.add(Line(xp(0),34,xp(0),185,strokeColor=colors.grey,strokeDashArray=[2,2]))
        for tick in (-.25,0,.25,.5,.75,1):
            drawing.add(Line(xp(tick),32,xp(tick),35,strokeColor=colors.black))
            drawing.add(String(xp(tick)-7,20,f'{tick*100:.0f}',fontSize=7))
        selected=[x for x in summary['contrasts'] if x['task']==task and x['split']=='test_id' and x['horizon']==6]
        for i,row in enumerate(selected):
            y=168-i*(43 if task=='gate' else 23)
            lab=str(row['seed']) if task=='gate' else row['role']+' '+str(row['seed'])
            drawing.add(String(origin-45,y-2,lab,fontSize=8))
            for j,metric in enumerate(('feature_mse','task_error')):
                v=row['metrics'][metric]['relative_reduction'];yy=y+(3 if j==0 else -3);color=colors.black if j==0 else colors.HexColor('#888888')
                a,b=xp(v['low']),xp(v['high'])
                drawing.add(Line(a,yy,b,yy,strokeColor=color,strokeWidth=1))
                drawing.add(Line(a,yy-2,a,yy+2,strokeColor=color));drawing.add(Line(b,yy-2,b,yy+2,strokeColor=color))
                drawing.add(Circle(xp(v['mean']),yy,2,fillColor=color,strokeColor=color))
    drawing.add(String(60,5,'Relative error reduction percent   Black feature error   Gray task error',fontSize=8))
    svg=report/'deletion_effects.svg';png=report/'deletion_effects.png';renderSVG.drawToFile(drawing,str(svg))
    node='/Users/linxiangyu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
    sharp='/Users/linxiangyu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp'
    subprocess.run([node,'-e',"require(process.argv[1])(process.argv[2],{density:220}).png().toFile(process.argv[3]).catch(e=>{console.error(e);process.exit(1)})",sharp,str(svg),str(png)],check=True)
    doc=Document();sec=doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    for style in doc.styles:
        if style.element.pPr is not None:
            for border in list(style.element.pPr.findall(qn('w:pBdr'))):style.element.pPr.remove(border)
    sec.top_margin=sec.bottom_margin=Inches(.7);sec.left_margin=sec.right_margin=Inches(.75)
    for name in ('Normal','Title','Heading 1','Heading 2'):
        style=doc.styles[name];style.font.name='Arial';style.font.color.rgb=RGBColor(0,0,0)
        style.font.size=Pt(10 if name=='Normal' else 20 if name=='Title' else 12)
    doc.styles['Normal'].paragraph_format.space_after=Pt(6)
    doc.add_paragraph('Equal capacity memory deletion assay',style='Title')
    doc.add_paragraph('Proposal Phase 1 mechanism evaluation')
    doc.add_paragraph('We test whether deleting different memories produces a reproducible prediction-cost signal that could supervise a bounded retention policy. We reuse six complete frozen predictors and existing development features. No predictor is retrained and no new visual encoding or GPU allocation is used. The entry decision appears at the end.')
    doc.add_heading('Design and scope',level=1)
    doc.add_paragraph('Each parent contains three real historical observations. Three forks delete one entry each, leaving exactly two memories alongside identical recent frames and action continuations. Gate parents contain one cue and two post-cue frames; Motion parents contain two pre-occlusion observations and one shutter frame. Two fixed nuisance-time variants are included. These privileged pools diagnose a mechanism and are not online retention policies.')
    doc.add_paragraph('We evaluate all 128 independent trajectory pairs in each of three existing development splits, both horizons h3 and h6, and seeds 17, 29 and 43. Siblings and nuisance variants are averaged within each pair before 10000 paired bootstrap replicates. The model always reads the same complete 5120-dimensional target representation. Simulator states are used only for evaluation after prediction.')
    doc.add_paragraph('The prospective ID entry rule requires at least 10 percent relative error reduction and a positive confidence lower bound when retaining each critical observation, for both feature and task errors in every seed and horizon. A feature-loss deletion ranking chosen at one horizon must also improve both errors at the other horizon. Long-gap and distractor splits separately delimit scope. Previous predictor-adoption failures are preserved.')
    doc.add_heading('Audited results',level=1)
    doc.add_picture(str(png),width=Inches(6.65))
    doc.add_paragraph('ID h6 effects by seed and evidence role. Points show mean relative reduction and bars show paired 95 percent bootstrap intervals. Positive values favor retaining the critical observation over nuisance content.')
    doc.add_page_break()
    t=doc.add_table(rows=1,cols=5);t.style='Table Grid'
    for c,v in zip(t.rows[0].cells,['Task','Split','Critical contrasts','Cross horizon','Supported']):c.text=v
    for task,values in summary['tasks'].items():
      for split,v in values['scope'].items():
        for c,x in zip(t.add_row().cells,[task,split.replace('test_','').replace('_',' '),f"{v['contrasts_passed']}/{v['contrasts_total']}",f"{v['ranking_passed']}/{v['ranking_total']}",'Yes' if v['supported'] else 'No']):c.text=str(x)
    tr=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(tr)
    doc.add_paragraph('Counts summarize prespecified checks, not the probability that Phase 1 is complete. A supported entry decision applies only to an ID-bounded policy pilot.')
    gate_failed=[x for x in summary['contrasts'] if x['task']=='gate' and x['split']=='test_id' and not x['passed']]
    doc.add_paragraph('Gate remains unstable for seed43: its ID h3 lane-error benefit has a confidence interval containing zero, while ID h6 fails both feature-error and lane-error contrasts. Motion passes the deletion contrasts in all three examined development splits, including long occlusion, even though its earlier absolute prediction-adoption test failed. Ranking useful deletions and solving future prediction to the earlier quality threshold are different requirements.')
    baselines=json.loads((result_dir/'baselines.json').read_text())['results']
    def baseline_mean(name,metric):
        values=[x['metrics'][metric]['mean'] for x in baselines if x['task']=='motion' and x['split']=='test_id' and x['horizon']==6 and x['baseline']==name]
        assert len(values)==3
        return sum(values)/3
    reject=baseline_mean('reject_newest','feature_mse');oracle=baseline_mean('future_feature_oracle','feature_mse')
    doc.add_paragraph(f'Motion provides limited headroom over a simple rule in these privileged pools. At ID h6, mean feature MSE across three seeds is {reject:.6f} for reject-newest and {oracle:.6f} for the future oracle, only {(reject-oracle)/reject:.2%} lower. This descriptive comparison does not establish a learned retention advantage; varied streaming memory states are still needed.')
    doc.add_heading('Limitations and decision boundary',level=1)
    doc.add_paragraph('The existing development data and earlier prediction results have been inspected. Deletion outcomes were new at specification time, but this is not untouched-data confirmation. Candidate pools are privileged, deletion is a single fork rather than a full streaming policy, and future-selected oracle rankings are unavailable online. Same-horizon oracle headroom alone is optimistically selected and cannot establish learnability or an advantage over strong retention baselines. FIFO, reject-newest and uniform-deletion costs are supplied in the accompanying audited data.')
    doc.add_paragraph(f"The audit independently recomputed {audit['independent_scalar_comparisons']} scalar comparisons from {audit['raw_rows']} raw fork/horizon rows and checked equal capacity, pairing and current asset hashes. Native CPU inference took {summary['elapsed_seconds']:.1f} seconds. Added GPU time is zero; cumulative settled GPU use remains 12723 of 28800 seconds.")
    doc.add_heading('Conclusion',level=1)
    ready=[task for task,v in summary['tasks'].items() if v['ID_pilot_ready']]
    if len(ready)==2:
        text='Both tasks support an ID-bounded retention-policy pilot under the prospective deletion assay. This permits a next-stage feasibility study, not a claim that a learned policy works or that the previous full Phase 1 confirmation protocol passed.'
    elif ready:
        text=f"Only {', '.join(ready)} supports an ID-bounded retention-policy pilot. The joint two-task policy stage is not ready. Keep the supported task as a scoped pilot and preserve the unsupported task as an explicit limitation; no additional predictor training was performed."
    else:
        text='The joint retention-policy stage is not ready: neither task satisfies the prospective ID deletion-signal rule. Preserve the negative deletion results and do not claim reliable supervision or launch policy training on this evidence.'
    doc.add_paragraph(text)
    path=report/'Proposal_Phase1_Deletion_Assay_Report.docx';doc.save(path)
    with (report/'all_contrasts.csv').open('w',newline='') as f:
        keys=['task','seed','split','horizon','role','metric','critical_mean','nuisance_mean','relative_reduction','CI_low_difference','CI_high_difference','passed'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in summary['contrasts']:
            for metric,v in row['metrics'].items():w.writerow({**{k:row[k] for k in keys[:5]},'metric':metric,'critical_mean':v['critical_mean'],'nuisance_mean':v['nuisance_mean'],'relative_reduction':v['relative_reduction']['mean'],'CI_low_difference':v['difference']['low'],'CI_high_difference':v['difference']['high'],'passed':v['passed']})
    (report/'report_manifest.json').write_text(json.dumps(dict(report=str(path),sha256=sha(path),decision=summary['decision'],native_summary_sha256=sha(result_dir/'summary.json'),rendered_pages_verified=False),indent=2)+'\n')
    print(path)
if __name__=='__main__':main(sys.argv[1])
