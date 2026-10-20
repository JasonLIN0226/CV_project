"""Cached checkpoint config/processor only; model construction/forward forbidden."""
from pathlib import Path
import torch
from phase1.encoders import checkpoint_components
from phase1.utils import write_json
from encoder_metadata import canonical_metadata

def verify(root,c,protocol):
 mc,processor,revision,_=checkpoint_components(c['encoder']);mc.dtype=getattr(torch,c['encoder']['dtype'])
 frozen=protocol['encoder_metadata'];actual=dict(frozen,model_config=mc.to_dict(),processor=processor.to_dict(),resolved_revision=revision)
 assert actual!=frozen and canonical_metadata(actual)==frozen
 write_json(Path(root)/'encoder_wire_metadata_CPU_check.json',dict(passed=True,
  cached_config_processor_revision_exact_after_JSON_roundtrip=True,no_CPU_model_construct_or_forward=True,
  GPU_actual_weight_count_and_full_metadata_still_checked_before_first_frame=True))
