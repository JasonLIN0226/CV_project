"""Draw two monochrome figures for the existing team Google Doc."""
from pathlib import Path
import json
import os
import subprocess
from reportlab.graphics.shapes import Drawing, Rect, Line, Polygon, String, Circle
from reportlab.graphics import renderSVG
from reportlab.lib import colors

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'google_doc_team_summary'
OUT.mkdir(exist_ok=True)
FONTCONFIG=ROOT/'english_detailed_report/fonts/fonts.conf'
NODE='/Users/linxiangyu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
SHARP='/Users/linxiangyu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp/dist/index.mjs'
health=json.loads((ROOT/'native7251_7268/visual_health.json').read_bytes())['full_vs_recent_paired_checks']

def text(d,x,y,s,size=11,anchor='start',bold=False):
    d.add(String(x,y,s,fontName='Times-Bold' if bold else 'Times-Roman',
                 fontSize=size,textAnchor=anchor,fillColor=colors.black))

def arrow(d,x1,y,x2):
    d.add(Line(x1,y,x2-5,y,strokeWidth=.8,strokeColor=colors.black))
    d.add(Polygon([x2,y,x2-6,y+3,x2-6,y-3],fillColor=colors.black,strokeColor=colors.black))

def save(d,name):
    svg=OUT/(name+'.svg');png=OUT/(name+'.png')
    renderSVG.drawToFile(d,str(svg))
    svg.write_text(svg.read_text().replace('Times-Roman','Times New Roman').replace('Times-Bold','Times New Roman'))
    js=f'import sharp from {json.dumps(SHARP)}; await sharp(process.argv[1], {{density:250}}).flatten({{background:"#ffffff"}}).png().toFile(process.argv[2]);'
    subprocess.run([NODE,'--input-type=module','-e',js,str(svg),str(png)],check=True,cwd=OUT,
                   env={**os.environ,'FONTCONFIG_FILE':str(FONTCONFIG)})
    return str(png)

d=Drawing(520,150)
boxes=[(8,['Observed video','past and current']),
       (139,['Visual features','fixed encoder']),
       (270,['Available memory','change what is kept']),
       (401,['Future prediction','same reader'])]
for x,lines in boxes:
    d.add(Rect(x,67,111,48,strokeColor=colors.black,fillColor=colors.white,strokeWidth=.8))
    text(d,x+55.5,95,lines[0],11,'middle',True)
    text(d,x+55.5,78,lines[1],9,'middle')
for i in range(3): arrow(d,boxes[i][0]+111,91,boxes[i+1][0])
text(d,260,132,'Compare memory while keeping the video and reader fixed',11,'middle',True)
text(d,8,43,'A  Change the amount of history to test whether it helps.',10)
text(d,8,26,'B  Change one deletion and keep the next observations the same.',10)
text(d,8,9,'C  Repeat fixed policies online; deleted information cannot return.',10)
workflow=save(d,'experimental_structure')

d=Drawing(520,160)
for col,h in enumerate(('16','64')):
    left=53+262*col; right=241+262*col
    low,high=-.001,.0084
    sx=lambda v:left+(v-low)/(high-low)*(right-left)
    text(d,(left+right)/2,146,'8 seconds' if h=='16' else '32 seconds',12,'middle',True)
    d.add(Line(sx(0),42,sx(0),132,strokeColor=colors.black,strokeWidth=.7,strokeDashArray=[3,3]))
    for i,(key,label) in enumerate((('17','Run 1'),('29','Run 2'),('43','Run 3'),('pooled','Mean'))):
        c=health[h][key]; y=124-i*23
        text(d,left-10,y-3,label,10,'end',key=='pooled')
        lo,hi,point=c['ci95_lower'][0],c['ci95_upper'][0],c['point'][0]
        d.add(Line(sx(lo),y,sx(hi),y,strokeColor=colors.black,strokeWidth=1))
        d.add(Line(sx(lo),y-3,sx(lo),y+3,strokeColor=colors.black,strokeWidth=.7))
        d.add(Line(sx(hi),y-3,sx(hi),y+3,strokeColor=colors.black,strokeWidth=.7))
        d.add(Circle(sx(point),y,3 if key=='pooled' else 2.5,fillColor=colors.black,strokeColor=colors.black))
    d.add(Line(left,38,right,38,strokeColor=colors.black,strokeWidth=.7))
    for value in (0,.004,.008):
        x=sx(value);d.add(Line(x,35,x,38,strokeColor=colors.black,strokeWidth=.7))
        text(d,x,24,f'{value:.3f}',9,'middle')
    text(d,(left+right)/2,8,'MSE reduction  (positive is better)',10,'middle')
effects=save(d,'history_benefit')
print(json.dumps({'workflow':workflow,'effects':effects,'new_inference_or_statistics':False}))
