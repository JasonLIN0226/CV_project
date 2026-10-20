"""Distinguish completed batch parents from a released interactive assay."""
def validate_parent_provenance(batch_text, interactive_text, receipt, audit, expected_summary):
    assert set(batch_text.strip().splitlines()) == {f'{j}|COMPLETED|0:0' for j in (7574,7575,7576)}
    assert interactive_text.strip() == '7590|CANCELLED by 3850228|0:0'
    assert receipt['job_id']==7590 and receipt['status']=='COMPLETE_WITH_INDEPENDENT_AUDIT'
    assert receipt['assay_returncode']==receipt['independent_returncode']==0
    assert audit['status']=='INDEPENDENT_SCALAR_AUDIT_PASS' and audit['raw_rows']==55296
    assert audit['independent_scalar_comparisons']==180 and audit['all_equal_capacity_and_pair_axis'] and audit['assets_stable']
    assert audit['summary_sha256']==expected_summary
    return True
