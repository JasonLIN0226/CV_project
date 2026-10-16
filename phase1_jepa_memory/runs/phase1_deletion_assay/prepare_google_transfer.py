"""Read the verified DOCX into a lossless content plan for its specified Google Doc."""
from pathlib import Path
import hashlib
import json
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from PIL import Image, ImageChops

BASE=Path(__file__).resolve().parent/'report/proposal_structured'
SOURCE=BASE/'Phase1_Structured_Academic_Report.docx'
OUT=BASE/'google_transfer'
(OUT/'images').mkdir(parents=True,exist_ok=True)
doc=Document(SOURCE)

def inherited(p,group,key,default=None):
    direct=getattr(getattr(p,group),key) if hasattr(p,group) else None
    if direct is not None:return direct
    s=p.style
    while s:
        v=getattr(getattr(s,group),key)
        if v is not None:return v
        s=s.base_style
    return default

def para(p):
    style=p.style.name
    pf={}
    for key,google,default in [('space_before','spaceAbove',0),('space_after','spaceBelow',6)]:
        v=inherited(p,'paragraph_format',key)
        pf[google]={'magnitude':float(v.pt) if v is not None else default,'unit':'PT'}
    ls=inherited(p,'paragraph_format','line_spacing',1.09)
    pf['lineSpacing']=float(ls)*100 if isinstance(ls,float) else 109
    for key,google,default in [('keep_with_next','keepWithNext',False),('keep_together','keepLinesTogether',False),('page_break_before','pageBreakBefore',False),('widow_control','avoidWidowAndOrphan',True)]:
        pf[google]=bool(inherited(p,'paragraph_format',key,default))
    a=inherited(p,'paragraph_format','alignment')
    pf['alignment']={0:'START',1:'CENTER',2:'END',3:'JUSTIFIED'}.get(a,'START')
    pf['namedStyleType']={'Title':'TITLE','Subtitle':'SUBTITLE','Heading 1':'HEADING_1','Heading 2':'HEADING_2','Heading 3':'HEADING_3'}.get(style,'NORMAL_TEXT')
    size=inherited(p,'font','size')
    fs=float(size.pt) if size else 10.5
    bold=bool(inherited(p,'font','bold',False))
    italic=bool(inherited(p,'font','italic',False))
    runs=[]
    for r in p.runs:
        if not r.text:continue
        runs.append({'text':r.text,'bold':bool(r.bold if r.bold is not None else bold),'italic':bool(r.italic if r.italic is not None else italic),'size':float(r.font.size.pt) if r.font.size else fs})
    return {'kind':'paragraph','text':p.text,'style':style,'paragraphStyle':pf,'size':fs,'bold':bold,'italic':italic,'runs':runs}

blocks=[];tables=[];images=[]
for el in doc.element.body:
    if el.tag.endswith('}p'):
        p=Paragraph(el,doc)
        b=para(p)
        blips=el.xpath('.//a:blip')
        if blips:
            assert len(blips)==1
            part=doc.part.related_parts[blips[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')]
            i=len(images);path=OUT/'images'/f'figure_{i+1:02}.png';path.write_bytes(part.blob)
            im=Image.open(path).convert('RGB');r,g,blue=im.split()
            assert ImageChops.difference(r,g).getbbox() is None and ImageChops.difference(g,blue).getbbox() is None,'Figure contains color'
            extent=el.xpath('.//wp:extent')[0]
            b.update(kind='image',image_id=i,path=str(path),width=float(extent.get('cx'))/12700,height=float(extent.get('cy'))/12700)
            images.append(b)
        blocks.append(b)
    elif el.tag.endswith('}tbl'):
        t=Table(el,doc);i=len(tables)
        b={'kind':'table','table_id':i,'widths':[float(c.width.pt) for c in t.columns],'rows':[]}
        for ri,row in enumerate(t.rows):
            cells=[]
            for ci,cell in enumerate(row.cells):
                c=para(cell.paragraphs[0]);c['text']=cell.text
                c['background']='333333' if ri==0 else ('F4F4F4' if ri%2==0 else 'FFFFFF')
                c['white_text']=ri==0
                cells.append(c)
            b['rows'].append(cells)
        tables.append(b);blocks.append(b)

sec=doc.sections[0]
plan={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'document_id':'1ZCtgzUOKNq6hlM-S1oZU7n7mHoKquX8HOkIPk--3hOs','tab_id':'t.0','blocks':blocks,'tables':tables,'images':images,'documentStyle':{'documentSize':{'width':{'magnitude':float(sec.page_width.pt),'unit':'PT'},'height':{'magnitude':float(sec.page_height.pt),'unit':'PT'}},'marginLeft':{'magnitude':float(sec.left_margin.pt),'unit':'PT'},'marginRight':{'magnitude':float(sec.right_margin.pt),'unit':'PT'},'marginTop':{'magnitude':float(sec.top_margin.pt),'unit':'PT'},'marginBottom':{'magnitude':float(sec.bottom_margin.pt),'unit':'PT'}}}
(OUT/'content_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'tables':len(tables),'figures':len(images),'body_blocks':len(blocks),'all_figures_monochrome':True,'plan':str(OUT/'content_plan.json')}))
