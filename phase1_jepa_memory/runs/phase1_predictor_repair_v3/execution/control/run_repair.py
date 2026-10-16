import argparse
import json
from pathlib import Path

from phase1.utils import write_json

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['train', 'evaluate', 'verify'])
parser.add_argument('--control', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--training-root')
args = parser.parse_args()
try:
    if args.stage == 'train':
        from phase1.repair_trial import run_training
        result = run_training(args.control, args.output)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['status'] == 'ALL_SIX_MODELS_COMPLETE_EVALUATION_PENDING' else 3)
    if args.stage == 'evaluate':
        from phase1.repair_evaluation import run_evaluation
        result = run_evaluation(args.control, args.training_root, args.output)
        print(json.dumps({k: v for k, v in result.items() if k != 'tasks'}), flush=True)
    else:
        from phase1.proposal_trial import verify_control
        p, lock, configs = verify_control(args.control)
        print(json.dumps({'source_protocol_verified': True, 'tasks': p['tasks'],
                          'files': len(lock['files']), 'phase1_go': False}), flush=True)
except Exception as error:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'failure.json', {'stage': args.stage, 'type': type(error).__name__,
               'error': str(error), 'phase1_go': False, 'refuse_automatic_retry': True})
    raise
