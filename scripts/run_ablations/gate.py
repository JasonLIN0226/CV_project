"""Gate ablation of our method for one predictor seed: the feature-attention-proxy twin.

    python scripts/run_ablations/gate.py --config configs/gate/team_bundle.yaml --seed 17 --device cpu --out <dir> \
        [--students <folder with student_<seed>.pt>]

Same scorer, same initial weights and minibatches as ours; only the training label differs
(similarity to the realised future in place of the predictor's error after each deletion). Both
arms live in one student checkpoint. Writes <out>/attention_proxy_seed<seed>.json.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluate_gate", Path(__file__).resolve().parents[1] / "evaluate/gate.py")
evaluate_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_gate)

if __name__ == "__main__":
    evaluate_gate.main(arm="attention_proxy_student", name="attention_proxy", doc=__doc__)
