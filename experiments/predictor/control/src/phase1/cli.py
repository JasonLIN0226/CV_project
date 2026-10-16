import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from .config import load_config, prepare
from .utils import environment, write_json, seed_all


def preflight(c, check_checkpoint=False):
    from .utils import device_of
    info = environment(c["device"])
    errors, notes = [], []
    try:
        device = device_of(c)
        if device.type == "cuda":
            import torch
            x = torch.ones((2, 2), device=device)
            info["cuda_compute_passed"] = bool(torch.equal(x @ x, 2*x))
            if not info["cuda_compute_passed"]:
                errors.append("CUDA arithmetic check failed")
    except RuntimeError as e:
        errors.append(str(e))
    root = Path(c["output"])
    root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=root):
            pass
    except OSError as e:
        errors.append("Output not writable: "+str(e))
    info["disk_free_bytes"] = shutil.disk_usage(root).free
    info["config_valid"] = True
    if c["encoder"]["backend"] == "ijepa":
        try:
            from transformers import IJepaModel, IJepaConfig, AutoImageProcessor
            info["ijepa_import"] = True
        except ImportError as e:
            errors.append("I-JEPA import failed: "+str(e))
        ec = c["encoder"]
        try:
            if Path(ec["repository"]).is_dir():
                cached = Path(ec["repository"])/"model.safetensors"
                info["checkpoint_single_weight_cached"] = cached.is_file()
            else:
                from huggingface_hub import try_to_load_from_cache
                cached = try_to_load_from_cache(ec["repository"], "model.safetensors", cache_dir=ec["cache_dir"], revision=ec["revision"])
                info["checkpoint_single_weight_cached"] = isinstance(cached, str)
        except Exception as e:
            errors.append("Checkpoint cache check failed: "+str(e))
        if check_checkpoint:
            try:
                from .encoders import checkpoint_components
                mc, processor, revision, _ = checkpoint_components(ec)
                info["resolved_revision"] = revision
                info["processor"] = processor.to_dict()
                if not ec["local_files_only"] and not Path(ec["repository"]).is_dir():
                    from huggingface_hub import get_hf_file_metadata, hf_hub_url
                    head = get_hf_file_metadata(hf_hub_url(ec["repository"], "model.safetensors", revision=ec["revision"]))
                    info["checkpoint_head_bytes"] = head.size
            except Exception as e:
                errors.append("Checkpoint metadata/network check failed: "+str(e))
        else:
            notes.append("Checkpoint remote access not checked; --check-checkpoint checks small config/processor and weight HEAD only.")
    else:
        info["ijepa_import"] = "not required for CPU smoke"
        notes.append("SMOKE ONLY — NOT JEPA; large checkpoint not loaded.")
    info.update(errors=errors, notes=notes, passed=not errors)
    write_json(root/"preflight.json", info)
    print(f"preflight: {'PASS' if not errors else 'FAIL'}; device={c['device']}; free={info['disk_free_bytes']/2**30:.1f} GiB", flush=True)
    for line in errors+notes:
        print(line, flush=True)
    return info


def encoder_check(c):
    import numpy as np
    from .encoders import build_encoder
    from .envs import scene, episode
    from .utils import device_of, write_npz
    s = scene(c, "gate", "train", 0)
    rgb = np.stack([episode(s, k)["rgb"][s["start"]] for k in [0, 1]])
    encoder = build_encoder(c, device_of(c))
    z = np.concatenate([encoder.encode(rgb[i:i+c["encoder"]["batch_size"]]) for i in range(0, len(rgb), c["encoder"]["batch_size"])])
    result = {"mode": c["mode"], "metadata": encoder.metadata, "shape": list(z.shape), "finite": bool(np.isfinite(z).all()),
              "pair_feature_mse": float(np.mean((z[0]-z[1])**2)), "meaning": "adapter/load diagnostic only, not a trained prediction experiment"}
    if hasattr(encoder, "model"):
        result.update(eval_mode=not encoder.model.training,
                      all_parameters_frozen=all(not p.requires_grad for p in encoder.model.parameters()),
                      parameter_gradients_absent=all(p.grad is None for p in encoder.model.parameters()))
        if not all(result[k] for k in ["eval_mode", "all_parameters_frozen", "parameter_gradients_absent"]):
            raise ValueError("Encoder must be frozen in eval/inference mode")
    if not result["finite"] or z.ndim != 2 or z.shape[0] != 2:
        raise ValueError("Encoder load diagnostic produced invalid features")
    write_json(Path(c["output"])/"encoder_check.json", result)
    write_npz(Path(c["output"])/"encoder_check_features.npz", z=z)
    print(f"encoder check: {result['shape']}, finite={result['finite']}; no research conclusion", flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description="Weeks 1-2 paired visual memory diagnostics; no retention policy")
    subs = p.add_subparsers(dest="command", required=True)
    helptexts = {
        "preflight": "Check environment; no weight download/training",
        "generate": "Generate deterministic paired RGB environments",
        "validate-data": "Abort on structural leakage; export contact sheets",
        "encode": "Cache frozen frame features (official backend may download multi-GB checkpoint)",
        "encoder-check": "Load encoder and infer only two images (official backend uses GPU/weights)",
        "probe": "Fit train-only linear evidence probes and independent state readouts",
        "train": "Train shared cached-feature predictor, automatic epoch-boundary resume",
        "evaluate": "Evaluate all conditions with identical checkpoint for each seed",
        "report": "Plot results and apply preregistered GO/REVISE/INCONCLUSIVE rules",
        "run-all": "Generate -> audit -> encode -> probe -> train -> evaluate -> report"}
    helptexts.update({"prepare-features": "Generate -> audit -> frozen encode; GPU stage",
                      "finish": "Fit train-only bases/probes -> resume train -> evaluate -> report; cached-feature stage",
                      "v3-develop": "Run one matched task-local comparison using validated v2 frozen features; CPU only",
                      "v3-development-report": "Combine actual development decisions; never a scientific GO",
                      "v3-confirm-features": "Encode fresh seed9137 data ONLY after both development gates pass",
                      "v3-confirm": "Train and evaluate fresh confirmation ONLY after both development gates pass"})
    for name, helptext in helptexts.items():
        sub = subs.add_parser(name, help=helptext, description=helptext)
        sub.add_argument("--config", default="configs/smoke.yaml", help="YAML config path")
        sub.add_argument("--output", help="Override output directory; mismatches require a fresh directory")
        sub.add_argument("--device", choices=["cpu", "cuda"], help="Explicit CPU/CUDA override")
        sub.add_argument("--seed", type=int, help="Override predictor seeds with a single seed (pilot only conclusion)")
        if name == "preflight":
            sub.add_argument("--check-checkpoint", action="store_true", help="Small metadata download + weight HEAD; no large weights")
        if name == "encode":
            sub.add_argument("--max-episodes", type=int, help="Limit NEW episodes for resumable encoder diagnostics")
        if name in {"train", "run-all", "finish", "v3-develop"}:
            sub.add_argument("--stop-after-epoch", type=int, help="Controlled early stop for resume tests")
        if name == "v3-develop":
            sub.add_argument("--source-output", required=True, help="Original audited v2 run; remains unchanged")
            sub.add_argument("--task", choices=["gate", "motion"], required=True)
        if name in {"v3-confirm-features", "v3-confirm"}:
            sub.add_argument("--development-output", required=True, help="Both actual comparisons must be SUPPORTED")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    c = load_config(args.config, args.output, args.device, args.seed)
    if args.command.startswith("v3-"):
        from .experiment import development, development_report, confirmation
        seed_all(c["training"]["seeds"][0], c["threads"])
        root = Path(c["output"])
        root.mkdir(parents=True, exist_ok=True)
        write_json(root/"environment.json", environment(c["device"]))
        if args.command == "v3-development-report":
            result = development_report(root)
        elif args.command == "v3-develop":
            if c["device"] != "cpu":
                raise ValueError("Cached-feature development must use allocated course CPU, no GPU")
            result = development(c, args.source_output, args.task, args.stop_after_epoch)
        else:
            result = confirmation(c, args.development_output, args.command == "v3-confirm-features")
        print(f"{args.command}: {result}", flush=True)
        return 143 if result.get("interrupted") else 0
    root = prepare(c)
    seed_all(c["training"]["seeds"][0], c["threads"])
    env = environment(c["device"])
    write_json(root/"environment.json", env)
    if args.command == "preflight":
        if not preflight(c, args.check_checkpoint)["passed"]:
            return 2
        return 0
    from .data import generate, validate_data
    from .cache import encode, fit_normalization
    from .probes import probe
    from .training import train
    from .evaluation import evaluate
    from .reporting import report
    if args.command in {"encode", "encoder-check", "run-all", "prepare-features"} and c["mode"] != "smoke":
        print("REAL I-JEPA: initial checkpoint load may download several GB; encoder and predictor stages use requested device. No Slurm job submitted by this command.", flush=True)
    commands = {"generate": lambda: generate(c), "validate-data": lambda: validate_data(c),
        "encode": lambda: encode(c, args.max_episodes), "encoder-check": lambda: encoder_check(c),
        "probe": lambda: probe(c), "train": lambda: train(c, args.stop_after_epoch),
        "evaluate": lambda: evaluate(c), "report": lambda: report(c)}
    if args.command in {"run-all", "prepare-features", "finish"}:
        if not preflight(c)["passed"]:
            return 2
        if args.command != "finish":
            generate(c)
            validate_data(c)
            encode(c)
        if args.command == "prepare-features":
            print(f"Frozen feature stage complete; output={root}", flush=True)
            return 0
        fit_normalization(c)
        probe(c)
        result = train(c, args.stop_after_epoch)
        if result["interrupted"]:
            return 143
        if args.stop_after_epoch is not None and args.stop_after_epoch < c["training"]["epochs"]:
            print("Stopped at requested epoch; rerun identical command without --stop-after-epoch to resume.")
            return 0
        evaluate(c)
        result = report(c)
    else:
        result = commands[args.command]()
        if args.command == "train" and result["interrupted"]:
            return 143
    print(f"{args.command}: done; output={root}", flush=True)
    if args.command in {"report", "run-all", "finish"}:
        print(result, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
