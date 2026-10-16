"""Plain English explanation of Phase1 and its separate deletion-signal check."""
import hashlib,json
from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'report'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    result_path=ROOT/'native7590/output/job_7590/summary.json'
    result=json.loads(result_path.read_text())
    audit=json.loads((result_path.parent/'independent_audit.json').read_text())
    prediction=json.loads((ROOT.parent/'phase1_predictor_repair_v2/report/final_summary.json').read_text())
    assert sha(result_path)==audit['summary_sha256'] and audit['status']=='INDEPENDENT_SCALAR_AUDIT_PASS'
    assert result['tasks']['motion']['ID_pilot_ready'] and not result['tasks']['gate']['ID_pilot_ready']
    assert prediction['models_completed']==6 and not prediction['phase1_go']
    doc=Document();sec=doc.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=Inches(.7)
    sec.left_margin=sec.right_margin=Inches(.8)
    for style in doc.styles:
        if style.element.pPr is not None:
            for b in list(style.element.pPr.findall(qn('w:pBdr'))):style.element.pPr.remove(b)
    for name,size in [('Normal',11),('Title',22),('Heading 1',13),('Heading 2',11)]:
        st=doc.styles[name];st.font.name='Arial';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
    normal=doc.styles['Normal'].paragraph_format
    normal.space_after=Pt(7);normal.line_spacing=1.08
    doc.styles['Heading 1'].paragraph_format.space_before=Pt(12)
    doc.styles['Heading 1'].paragraph_format.space_after=Pt(5)
    foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    run=foot.add_run('Phase One Experiment Report  |  ');run.font.size=Pt(8)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
    doc.add_paragraph('Phase One Experiment Report',style='Title')
    p=doc.add_paragraph('A simple explanation of the goals, experiments and results\n16 October 2026')
    p.paragraph_format.space_after=Pt(12)
    doc.add_heading('1  What was the goal',level=1)
    doc.add_paragraph('Our overall research question is: which past observations should a model keep when its memory is limited? We want to test whether the prediction cost of forgetting an observation can help a future memory policy make better choices.')
    doc.add_paragraph('Phase One prepares the evidence for that research. We need to check whether visual features contain useful historical information and whether giving that information to a predictor improves its forecasts. Training a learned memory policy belongs to a later stage.')
    doc.add_heading('2  What experiments did we run',level=1)
    table=doc.add_table(rows=1,cols=2);table.style='Table Grid'
    for cell,text in zip(table.rows[0].cells,['Task','Why the past matters']):cell.text=text
    for row in [
        ('Gate','A brief earlier cue helps determine which lane an object will take later. The cue has disappeared from the recent images.'),
        ('Motion','An object moves, becomes hidden, and reappears. Its earlier positions help predict where it will return.')]:
        for cell,text in zip(table.add_row().cells,row):cell.text=text
    table.columns[0].width=Inches(.85);table.columns[1].width=Inches(6.05)
    for cell in table.rows[0].cells:
        for run in cell.paragraphs[0].runs:run.bold=True
    doc.add_paragraph('We used a frozen I-JEPA image encoder to turn images into numerical features, then trained small predictors of future features. Predictions were also checked against the true lane or position. Hidden simulator states were used for scoring, never as predictor inputs.')
    doc.add_paragraph('We compared recent images alone, all available history, and two important past observations selected using the task design. Paired episodes had the same recent images and actions, so differences could be linked to their history. We tested ordinary conditions, longer delays and changed visual distractions. The last complete prediction experiment repeated training three times per task with different random starts, giving six predictors. Each test condition contained 128 matched trajectory pairs per task.')
    doc.add_heading('3  What did the deletion check add',level=1)
    doc.add_paragraph('As a bridge to the next stage, we reused those six frozen predictors and their cached features. Each test started with three past observations. We deleted each one in turn, always leaving exactly two. We asked whether deleting important evidence caused more harm than deleting an uninformative frame, and whether a deletion choice useful at one future time was also useful at another.')
    doc.add_paragraph('This was a single-deletion experiment. It used no new predictor training or GPU time, and no learned memory policy was trained.')
    doc.add_page_break()
    doc.add_heading('4  What did we find',level=1)
    for lead,body in [
        ('The visual features contain relevant evidence. ', 'Both tasks passed the checks that asked whether important historical information could be read from the stored features.'),
        ('Prediction gains vary across conditions. ', 'Gate showed strong benefits in most checks, with one unstable training seed. Motion improved in ordinary conditions, but prediction quality was less reliable with long occlusion or changed backgrounds. The last complete predictor version did not pass all of the stricter acceptance checks.'),
        ('The deletion signal is much stronger for Motion. ', 'Motion passed all 36 critical-deletion comparisons across the three test conditions. Its deletion choices also transferred between the two future times in all 18 ranking checks. Gate was less stable: it passed 4 of 6 critical-deletion comparisons in ordinary conditions, 0 of 6 with long delays, and 2 of 6 with changed distractions. The ordinary-condition failures came from one of the three training runs.')]:
        p=doc.add_paragraph();p.add_run(lead).bold=True;p.add_run(body)
    doc.add_paragraph('Each critical-deletion check required a meaningful improvement in both visual-feature error and the lane or position error, with statistical support. These counts describe the checks passed; they are not percentages of project completion.')
    doc.add_heading('5  Why these results matter',level=1)
    doc.add_paragraph('A deletion cost compares the harm of different forgetting choices. It can identify which observation matters even when the model still makes substantial prediction errors. This explains why Motion can have a useful deletion signal while its difficult future predictions remain imperfect.')
    doc.add_paragraph('However, a useful signal does not yet prove that a learned policy will beat simple rules. In the ordinary six-step Motion comparison, simply rejecting the newest frame was within 1 percent of the best future-informed deletion in average visual-feature error. We need more varied memory situations to test whether learning offers a useful advantage.')
    doc.add_heading('6  What remains unproven',level=1)
    doc.add_paragraph('The deletion test used existing development data and deliberately constructed memory sets containing known evidence. It has not established success on untouched data, continuous online forgetting, or a policy that makes good choices using only past information. Independent confirmation on fresh data has not been run.')
    doc.add_heading('Conclusion',level=1)
    doc.add_paragraph('Phase One has produced useful evidence, but complete validation of both tasks has not been established. Motion provides enough support for a small, clearly scoped test of whether a retention policy can learn from deletion costs. Gate does not yet provide the same reliable signal. We should not claim that the full two-task memory method works, or that the previous full acceptance protocol has passed.')
    doc.add_paragraph('No retention policy has been trained. The deletion experiment added zero GPU time, and its CPU resources have been released.')
    OUT.mkdir(exist_ok=True)
    path=OUT/'Phase1_Simple_English_Report.docx';doc.save(path)
    manifest={'file':str(path),'sha256':sha(path),'purpose':'Plain English account of original Phase1 goals, prediction experiments and separate deletion-signal bridge test','source_summary_sha256':sha(result_path),'source_prediction_summary_sha256':sha(ROOT.parent/'phase1_predictor_repair_v2/report/final_summary.json'),'pages_visually_verified':False}
    (OUT/'simple_report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(path)

if __name__=='__main__':main()
