"""Evaluation bookkeeping repair; frozen implementation remains unchanged."""
import hashlib
from pathlib import Path
import sys

root = Path('/scratch/xl4358/clean_future_gate_20261022/code/CV_project')
source = root / 'experiments/gate_clean_future/evaluate_validation.py'
text = source.read_text()
import json
lock = json.loads((source.parent / 'source_lock.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == lock['files'][source.name]
anchor = '                    for item in pair_items:\n                        row = item["row"]'
replacement = '''                    for item in pair_items:
                        other = pair_items[1] if item is pair_items[0] else pair_items[0]
                        pair_accuracy, pair_margin, separation = pair_diagnostic(
                            item["prediction"][horizon_index],
                            item["target"][horizon_index],
                            other["target"][horizon_index],
                        )
                        row = item["row"]'''
assert text.count(anchor) == 1
sys.path.insert(0, str(source.parent))
exec(compile(text.replace(anchor, replacement), str(source), 'exec'),
     {'__name__': '__main__', '__file__': str(source)})
