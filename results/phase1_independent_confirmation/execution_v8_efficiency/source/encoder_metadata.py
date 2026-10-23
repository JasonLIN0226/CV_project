"""Strict JSON wire comparison; normalize representation, never values."""
import json

def canonical_metadata(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))

def matches(actual,frozen):
    return canonical_metadata(actual)==frozen
