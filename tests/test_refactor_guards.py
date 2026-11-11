"""Guards for the src/bounded_memory layout.

The code was moved out of experiments/, which no longer exists in the working tree. "Old" below
always means the file at the base commit, read with `git show fe69074:<path>`.

1. The historical record (docs, results, reports, handoff) is byte-identical to the base commit.
2. Code that was extracted "unchanged" really is: each function or class has the same source text
   as the one it came from; moved files differ only in their import block.
3. The package and the scripts import nothing from results/ or experiments/.
"""
import ast
import functools
import hashlib
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/bounded_memory"
BASE = "fe69074"          # the commit the refactor started from
PROTECTED = ["docs", "results", "reports", "handoff"]
P1 = "experiments/predictor/control/src/phase1"
V2 = "experiments/gate_supervision_comparison_design_v2"
MB = "experiments/memory_baselines"
PIPE = "results/literature_diagnosis/qualified-reader-retention-20261025/code/pipe.py"
RTS = "experiments/phase2/run_terminal_structures.py"
RRA = "experiments/phase2/run_reader_aggregation.py"
VERBATIM = [   # (file in src/bounded_memory, source file, names; "Class.method" for methods)
    ("data/io.py", f"{P1}/utils.py", ["digest", "file_hash", "read_json", "atomic", "write_json", "write_npz", "save_torch", "stable_seed"]),
    ("predictors/dynamics.py", f"{P1}/model.py", ["project", "DynamicsPredictor"]),
    ("predictors/inputs.py", f"{P1}/inputs.py", ["selected_evidence", "memory_indices", "collate"]),
    ("environments/gate/assets.py", f"{P1}/cache.py", ["normalize"]),
    ("evaluation/metrics/gate.py", "experiments/gate_clean_future/frozen_control/probes.py", ["LinearReadout"]),
    ("evaluation/heuristics/gate.py", "experiments/gate_retention_comparison_v1/online.py", ["update", "replay"]),
    ("methods/deletion_cost/scorer.py", f"{V2}/core.py", ["EvictionScorer", "paired_initialization", "OnlineState", "choose", "score_update"]),
    ("methods/deletion_cost/retention.py", f"{V2}/core.py", ["rollout", "behavior_update"]),
    ("methods/deletion_cost/retention.py", f"{V2}/states.py", ["deterministic_seed"]),
    ("methods/deletion_cost/teacher.py", f"{V2}/teachers.py", ["OfflineTargets", "deletion_distribution", "predictor_samples", "deletion_energy", "teacher_pair"]),
    ("methods/deletion_cost/ablations/attention_proxy.py", f"{V2}/teachers.py", ["attention_energy"]),
    ("methods/deletion_cost/training.py", f"{V2}/fit.py", ["minibatch_indices", "fit_pair"]),
    ("methods/pafu_kv/repo_adaptation.py", "experiments/phase2/literature_baselines.py",
     ["_integer", "_immutable", "make_student_inputs", "balanced_attention_targets", "SalienceStudent", "SalienceTrainingRecord", "train_salience_student"]),
    ("methods/pafu_kv/gate.py", f"{MB}/pafu.py", ["teacher_salience", "write_features", "SalienceHead", "train", "select", "teacher_select"]),
    ("methods/pacc/compressor.py", f"{MB}/pacc.py", ["Compressor", "_schedule"]),
    ("methods/pacc/gate.py", f"{MB}/pacc.py", ["_history", "_teacher", "_student", "_recurrent_memory", "train", "predict"]),
    ("methods/framemorrow/gate.py", f"{MB}/fm.py", ["_condition", "_tuple", "_collate", "_student_kwargs", "_loss", "train", "select_published", "select_online", "select_teacher"]),
    ("methods/framemorrow/synergy.py", f"{MB}/synergy_run.py", ["fm_rows", "fm_collate", "fm_student", "framemorrow", "framemorrow_teacher"]),
    ("predictors/readers.py", "experiments/phase2/reader_aggregation.py",
     ["parameter_count", "_output", "_masked_softmax", "AggregationPredictor", "UniformInterventionPredictor", "_wide_width", "build_reader", "describe_reader", "reader_attention"]),
    ("environments/synergy/assets.py", PIPE, ["Store.capsule", "Store.student_features", "Store.target_residual", "Store.actual_target_residual",
                                               "batch_tensors", "coefficients", "pooled_attention", "teacher_attention", "qform"]),
    ("environments/gate/protocol.py", f"{MB}/gate_data.py", ["item_batch", "with_memory_tokens", "pooling_attention", "predict_selection"]),
    ("evaluation/metrics/gate.py", f"{MB}/gate_data.py", ["metric_rows", "summarise"]),
    ("evaluation/metrics/synergy.py", f"{MB}/synergy_data.py", ["family_mean", "score"]),
    ("evaluation/oracles/gate.py", f"{MB}/gate_policies.py", ["recent_only", "oracle_keyframe"]),
    ("environments/synergy/dataset.py", RTS, ["read", "write", "sha", "emit", "RGBRegistry", "fifo_parent_trace", "grouped", "Data",
                                              "causal_capsule", "reader_sample", "group_loss"]),
    ("environments/synergy/train_readers.py", RRA, ["initialize_model"]),
    ("encoders/ijepa.py", "experiments/phase2/encode_fresh_gate_diagnostic.py", ["digest"]),
]
VERBATIM_BUT_IMPORT_LINES = [   # the same, ignoring the import statements inside the function body
    ("environments/synergy/dataset.py", RTS, ["make_banks", "generate", "encode"]),
    ("environments/synergy/train_readers.py", RRA, ["train_one"]),
]
MOVED_IMPORT_BLOCK_ONLY = [   # whole files: every statement the same, except imports and search-path set-up
    ("environments/gate/design.py", "experiments/retention_position_noise_audit/design.py"),
    ("environments/gate/structured_design.py", "experiments/retention_structured_noise_audit/structured_design.py"),
    ("environments/gate/generate.py", "experiments/gate_clean_future/materialize.py"),
    ("environments/gate/features.py", "experiments/gate_clean_future/fit_features.py"),
    ("environments/gate/train_predictor.py", "experiments/retention_structured_noise_audit/train_reference.py"),
]
SEARCH_PATH_NAMES = {"HERE", "REPO", "STRUCTURED", "CONTROL", "CONTROL_SOURCE", "SOURCE_DIR"}
MOTION = ["envs_v2.py", "generate_v2.py", "step2_metrics.py", "step34_core.py", "step34_data.py", "step34_students.py",
          "step34_metrics.py", "audit_v2.py"]
PUSHT = ["pusht.py", "pusht_cog_cue.py", "pusht_cog_coherent.py", "pusht_cog_matched.py", "pusht_cog_natural.py", "pusht_cog_marker.py",
         "prepare_pusht_cog_marker_data.py", "prepare_pusht_cog_v1_data.py", "prepare_pusht_cog_v2_data.py",
         "prepare_pusht_cog_v4_data.py", "prepare_pusht_planning_data.py", "diagnose_pusht_cog_pose_decoding.py",
         "dev_pusht_cog_matched.py", "dev_pusht_cog_natural.py", "audit_pusht_training_data.py"]
ONE_LINE_CHANGED = [   # moved with a single documented edit: where the frozen predictor source is found
    ("environments/motion_v2/reader_v2.py", "experiments/motion_v2/reader_v2.py", "_REFERENCE_ROOT"),
    ("environments/pusht/probe_pusht_cog_features.py", "experiments/phase2/probe_pusht_cog_features.py", "PHASE1_SRC"),
]
MOVED_AS_IS = ("predictors/phase1", "environments/motion_v2", "environments/pusht")   # original files, original import style
IDENTICAL_FILES = [   # byte-identical copies
    ("methods/deletion_cost/terminal_students.py", "experiments/phase2/fit_terminal_students.py"),
    ("methods/framemorrow/reference/model.py", f"{MB}/framemorrow_ref/model.py"),
    ("methods/framemorrow/reference/losses.py", f"{MB}/framemorrow_ref/losses.py"),
    ("methods/framemorrow/reference/LICENSE", f"{MB}/framemorrow_ref/LICENSE"),
    ("methods/framemorrow/reference/README.md", f"{MB}/framemorrow_ref/README.md"),
] + [(f"environments/motion_v2/{n}", f"experiments/motion_v2/{n}") for n in MOTION] \
  + [(f"environments/motion_v2/pipeline/{p.name}", f"experiments/motion_v2/{p.name}")
     for p in sorted((SRC / "environments/motion_v2/pipeline").iterdir()) if p.name not in ("README.md", "__pycache__")] \
  + [(f"environments/pusht/{n}", f"experiments/phase2/{n}") for n in PUSHT] \
  + [(f"predictors/phase1/{p.name}", f"{P1}/{p.name}") for p in sorted((SRC / "predictors/phase1").glob("*.py"))]
IDENTICAL_CONFIGS = [   # (file in the repository, file at the base commit)
    ("configs/gate/dataset_config_template.json", "results/gate_retention_comparison_20261029/dataset_config_template.json"),
    ("configs/synergy/reader_aggregation_20261025.json", "experiments/phase2/configs/reader_aggregation_20261025.json"),
]


def git(*args, text=True):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=text)


@functools.lru_cache(maxsize=None)
def old_bytes(path):
    """A file as it was at the base commit."""
    if git("cat-file", "-e", BASE).returncode != 0:
        pytest.skip("not a git checkout with the base commit")
    result = git("show", f"{BASE}:{path}", text=False)
    assert result.returncode == 0, f"{path} does not exist at {BASE}"
    return result.stdout


def old_text(path):
    return old_bytes(path).decode()


def definitions(text):
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            out[node.name] = textwrap.dedent("\n".join(text.splitlines()[start - 1:node.end_lineno]))
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        first = min([child.lineno] + [d.lineno for d in child.decorator_list])
                        out[f"{node.name}.{child.name}"] = textwrap.dedent("\n".join(text.splitlines()[first - 1:child.end_lineno]))
    return out


def without_import_lines(source):
    return "\n".join(line for line in source.splitlines() if not line.strip().startswith(("from ", "import ")))


def statements(text):
    """Every top-level statement except imports and search-path set-up, as syntax trees."""
    kept = []
    for node in ast.parse(text).body:
        source = ast.get_source_segment(text, node)
        if isinstance(node, (ast.Import, ast.ImportFrom)) or "sys.path" in source:
            continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in SEARCH_PATH_NAMES for t in node.targets):
            continue
        kept.append(ast.dump(node))
    return kept


def test_the_historical_record_is_unchanged():
    if git("cat-file", "-e", BASE).returncode != 0:
        pytest.skip("not a git checkout with the base commit")
    changed = git("diff", "--name-status", BASE, "--", *PROTECTED).stdout.strip()
    assert changed == "", f"protected files differ from {BASE}:\n{changed}"


@pytest.mark.parametrize("new, old, names", VERBATIM, ids=[f"{n}<-{Path(o).name}" for n, o, _ in VERBATIM])
def test_extracted_code_has_the_source_text_it_came_from(new, old, names):
    mine, theirs = definitions((SRC / new).read_text()), definitions(old_text(old))
    for name in names:
        assert name in mine and name in theirs, name
        assert mine[name] == theirs[name], f"{new}:{name} differs from {old}"


@pytest.mark.parametrize("new, old, names", VERBATIM_BUT_IMPORT_LINES, ids=[f"{n}<-{Path(o).name}" for n, o, _ in VERBATIM_BUT_IMPORT_LINES])
def test_extracted_code_differs_only_in_its_import_lines(new, old, names):
    mine, theirs = definitions((SRC / new).read_text()), definitions(old_text(old))
    for name in names:
        assert without_import_lines(mine[name]) == without_import_lines(theirs[name]), f"{new}:{name} differs from {old}"


@pytest.mark.parametrize("new, old", MOVED_IMPORT_BLOCK_ONLY, ids=[n for n, _ in MOVED_IMPORT_BLOCK_ONLY])
def test_moved_files_differ_only_in_their_import_block(new, old):
    assert statements((SRC / new).read_text()) == statements(old_text(old))


@pytest.mark.parametrize("new, old", IDENTICAL_FILES, ids=[n for n, _ in IDENTICAL_FILES])
def test_copied_files_are_byte_identical(new, old):
    assert hashlib.sha256((SRC / new).read_bytes()).hexdigest() == hashlib.sha256(old_bytes(old)).hexdigest()


@pytest.mark.parametrize("new, old", IDENTICAL_CONFIGS, ids=[n for n, _ in IDENTICAL_CONFIGS])
def test_frozen_configs_are_byte_identical(new, old):
    assert (ROOT / new).read_bytes() == old_bytes(old)


@pytest.mark.parametrize("new, old, marker", ONE_LINE_CHANGED, ids=[n for n, _, _ in ONE_LINE_CHANGED])
def test_moved_files_differ_only_where_the_predictor_source_is_located(new, old, marker):
    mine, theirs = (SRC / new).read_text().splitlines(), old_text(old).splitlines()
    start = next(i for i, line in enumerate(theirs) if line.startswith(marker))
    assert mine[:start] == theirs[:start]
    assert mine[start].startswith("#") or mine[start].startswith(marker)
    tail = min(len(mine), len(theirs)) - start - 6          # everything after the edited constant is unchanged
    assert mine[-tail:] == theirs[-tail:]


def test_vendored_predictor_source_matches_the_checksums_motion_v2_registered():
    registered = ast.literal_eval(ast.get_source_segment(
        (SRC / "environments/motion_v2/reader_v2.py").read_text(),
        next(n for n in ast.parse((SRC / "environments/motion_v2/reader_v2.py").read_text()).body
             if isinstance(n, ast.Assign) and n.targets[0].id == "_REFERENCE_HASHES").value))
    for name, expected in registered.items():
        assert hashlib.sha256((SRC / f"predictors/phase1/{name}.py").read_bytes()).hexdigest() == expected, name


def test_gate_rebuild_uses_the_probe_functions_of_the_clean_future_source():
    """The published Gate predictors were trained with a later probes.py than the vendored one.
    The two differ only in probe(), which the Gate rebuild never calls."""
    mine = definitions((SRC / "predictors/phase1/probes.py").read_text())
    theirs = definitions(old_text("experiments/gate_clean_future/frozen_control/probes.py"))
    assert set(mine) == set(theirs)
    assert [name for name in mine if mine[name] != theirs[name]] == ["probe"]
    called = {n.id for n in ast.walk(ast.parse((SRC / "environments/gate/features.py").read_text())) if isinstance(n, ast.Name)}
    assert "probe" not in called


def test_pinned_encoder_is_the_frozen_one():
    old = old_text("experiments/phase2/encode_fresh_gate_diagnostic.py")
    for node in ast.parse((SRC / "encoders/ijepa.py").read_text()).body:
        if isinstance(node, ast.Assign):
            assert ast.get_source_segment((SRC / "encoders/ijepa.py").read_text(), node) in old, node.targets[0].id


def test_synergy_generator_is_the_frozen_generator():
    mine = (SRC / "environments/synergy/generate.py").read_text().split("\n", 1)[1]
    assert mine == old_text("experiments/phase2/terminal_structures.py")


def test_package_and_scripts_import_nothing_from_results_or_experiments():
    offenders = []
    for path in [*SRC.rglob("*.py"), *(ROOT / "scripts").rglob("*.py"), *(ROOT / "torch").glob("*.py")]:
        inside = path.is_relative_to(SRC)
        if inside and any(str(path.relative_to(SRC)).startswith(folder) for folder in MOVED_AS_IS) and path.name != "evidence.py":
            continue      # moved unchanged; they find their siblings by file name, as the originals did
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            offenders += [f"{path.relative_to(ROOT)}: {m}" for m in modules if m.split(".")[0] in ("results", "experiments", "pipe", "phase1")]
        if inside and ("sys.path" in text or "results/literature_diagnosis" in text.split('"""')[2 if text.startswith('"""') else 0]):
            offenders.append(f"{path.relative_to(ROOT)}: path manipulation or results/ path in code")
    assert not offenders, offenders


def test_reproduction_runs_no_code_outside_the_new_layout():
    """Every task in the manifest starts a script under scripts/ (or pytest), from the code snapshot."""
    import yaml
    tasks = yaml.safe_load((ROOT / "configs/reproduction.yaml").read_text())["tasks"]
    for name, task in tasks.items():
        script = task["script"]
        assert script.startswith(("scripts/", "-m pytest ")), f"{name}: {script}"
        assert script.startswith("-m pytest ") or (ROOT / script).is_file(), f"{name}: {script} is missing"
        assert task.get("cwd") != "repo" or script.startswith("-m pytest "), f"{name} runs from the checkout"
