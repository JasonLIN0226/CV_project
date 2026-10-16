#!/usr/bin/env python3
"""Rebuild the latest Word report using only the published scalar evidence."""
from pathlib import Path
import re, shutil

ROOT=Path(__file__).resolve().parents[1]
builder=ROOT/'phase1_jepa_memory/runs/phase1_deletion_assay/build_structured_report.py'
pdftoppm=shutil.which('pdftoppm')
if not pdftoppm:raise SystemExit('Install Poppler so that pdftoppm is on PATH, then retry.')
code=re.sub(r'^PDFTOPPM = .+$','PDFTOPPM = '+repr(pdftoppm),builder.read_text(),count=1,flags=re.M)
assert 'PDFTOPPM = '+repr(pdftoppm) in code
exec(compile(code,str(builder),'exec'),{'__name__':'__main__','__file__':str(builder)})
out=ROOT/'phase1_jepa_memory/runs/phase1_deletion_assay/report/proposal_structured/Phase1_Structured_Academic_Report.docx'
shutil.copy2(out,ROOT/'reports/Phase1_Report.docx')
print('Updated reports/Phase1_Report.docx. No model or training execution was performed.')
