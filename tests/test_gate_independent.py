import pytest

from bounded_memory.data.io import file_hash, read_json, write_json
from bounded_memory.environments.gate.holdout_protocol import (
    COHORT_DIR,
    GENERATION_SEED,
    SPLITS,
    audit_identities,
    historical_records,
    paired_route_inference,
)
from bounded_memory.environments.gate.protocol import HORIZONS, SEEDS
from bounded_memory.evaluation.gate_independent import compare_validation, holdout_inputs
from bounded_memory.methods.deletion_cost.training import ARMS


def validation_fixture():
    records = [{"id": str(i), "pair": str(i // 2)} for i in range(192)]
    actual, archived = [], []
    for r in records:
        for p in ARMS:
            for s in SEEDS:
                for h in HORIZONS:
                    common = dict(
                        condition=p,
                        split="validation",
                        episode=r["id"],
                        pair=r["pair"],
                        seed=s,
                        horizon=h,
                        normalized_mse=0.1,
                        selected_indices="0;2",
                    )
                    truth = int(r["id"]) % 2
                    actual.append(
                        dict(common, state_accuracy=1, true_route_class=truth, predicted_route_class=truth)
                    )
                    archived.append(dict(common, state_accuracy=1))
    return actual, archived, records


def test_replay_detects_wrong_routes_and_frames_even_with_identical_pooled_accuracy():
    rows, ref, manifest = validation_fixture()
    ref[0]["state_accuracy"] = 0
    rows[1]["state_accuracy"] = 0
    rows[1]["predicted_route_class"] = 1 - rows[1]["true_route_class"]
    rows[1]["selected_indices"] = "0;3"
    result = compare_validation(rows, list(reversed(ref)), manifest)
    assert result[ARMS[0]]["route_mismatches"] == 2
    assert result[ARMS[0]]["selection_mismatches"] == 1
    assert result[ARMS[1]]["route_mismatches"] == 0


def test_replay_rejects_missing_and_duplicate_rows():
    rows, ref, records = validation_fixture()
    with pytest.raises(ValueError, match="2,304"):
        compare_validation(rows, ref[:-1], records)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_validation(rows, ref[:-1] + ref[:1], records)


def test_independent_mode_rejects_validation_manifest_before_loading_features(tmp_path):
    path = tmp_path / "dataset/manifest.json"
    write_json(path, {"records": [{"split": "validation"}]})
    with pytest.raises(ValueError, match="rejects train/validation"):
        holdout_inputs(None, tmp_path, [], {"manifest_sha256": file_hash(path)})


def test_paired_inference_keeps_ordinary_and_long_gap_separate():
    # Synthetic score rows only; no renderer, encoded features, models or test assets.

    records, rows = [], []
    for split in SPLITS:
        for i in range(64):
            pair = f"fixture-{split}-{i}"
            for sibling in (0, 1):
                episode = f"{pair}-{sibling}"
                records.append(dict(id=episode, pair=pair, split=split))
                for p in ARMS:
                    for seed in SEEDS:
                        for h in HORIZONS:
                            correct = (p == "deletion_cost_student") == (split == "test_id")
                            rows.append(
                                dict(
                                    episode=episode,
                                    pair=pair,
                                    split=split,
                                    condition=p,
                                    state_accuracy=int(correct),
                                    seed=seed,
                                    horizon=h,
                                )
                            )
    result = paired_route_inference(rows, records)
    assert result["conditions"]["test_id"]["ci_97_5_percent"] == [1.0, 1.0]
    assert result["conditions"]["test_long_gap"]["ci_97_5_percent"] == [-1.0, -1.0]
    assert result["positive_on_both_conditions"] is False


def test_registered_cohort_is_disjoint_from_prior_scenes():
    config = read_json(COHORT_DIR / "cohort.json")
    assert config["dataset_seed"] == GENERATION_SEED
    manifest = COHORT_DIR / "manifest.json"
    assert file_hash(manifest) == config["manifest_sha256"]
    records = read_json(manifest)["records"]
    prior = historical_records([], [COHORT_DIR / "prior_ids.csv.gz"])
    audit = audit_identities(records, prior, GENERATION_SEED, config)
    assert audit["episodes"] == 256
    with pytest.raises(ValueError, match="previously exposed"):
        audit_identities(records, [*prior, records[0]], GENERATION_SEED, config)
