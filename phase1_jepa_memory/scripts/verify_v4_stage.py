#!/usr/bin/env python3
"""Verify the already audited prerequisite files before the new allocated stage."""
from pathlib import Path
from phase1.utils import read_json, file_hash

root=Path('/scratch/xl4358/phase1_v4_diagnostics')
c=read_json('configs/phase1_v4_stage_authorization.json')
checks={'motion_cross_policy_sha256':root/'motion_cross_policy.json',
        'motion_review_sha256':root/'motion_cross_policy_independent_review.json',
        'gate_training_authorization_sha256':root/'gate_training_authorization.json',
        'motion_mechanism_protocol_sha256':Path('configs/phase1_v4_motion_mechanism_protocol.json'),
        'gate_protocol_sha256':Path('configs/phase1_v4_gate_protocol.json')}
if c['protocol_version']!=4 or any(file_hash(path)!=c[key] for key,path in checks.items()):
    raise ValueError('Actual prerequisite evidence/protocol changed')
grid=read_json(checks['motion_cross_policy_sha256'])
if grid['status']!='DIAGNOSTIC_ONLY_NOT_GO' or grid['frozen_weights'] is not True or grid['new_training'] is not False or grid['source_unchanged'] is not True:
    raise ValueError('Actual frozen Motion grid is invalid')
print('Audited prerequisite files match; no Phase1 GO or Motion training claim.',flush=True)
