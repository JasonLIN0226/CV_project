"""Read-only relocation of the project's verified development backup.

Original absolute-path manifests remain byte-for-byte unchanged. Only explicitly
requested train/validation arrays are opened. This is an offline audit/training
adapter, never a retention-policy input or a way to restore evicted memories.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from .train_predictor_pilot import ROOT, sha256
from phase1.model import DynamicsPredictor
from phase1.utils import seed_all


BACKUP_INDEX_SHA256 = "4980c4ffe3ad7357e4c80e23d5fe6d919610add866acb66f646e0e35d2620da8"


def state_hash(model):
    import hashlib
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode() + b"\0" + str(array.dtype).encode() + repr(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


class PortableSnapshot:
    """Load the fixed, locally generated Oct-20 archive, with lazy hash checks."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        index_path = self.root / "backup-index.json"
        if sha256(index_path) != BACKUP_INDEX_SHA256:
            raise ValueError("Backup index differs from the independently verified development export")
        self.index = json.loads(index_path.read_text())["files"]
        self.loaded = {}
        self.array_access = []
        self._features = {}
        self._bases = {}
        self.run_relative = "runs/cpu-predictor-pilot"
        self.config = self.read_json(self.run_relative + "/training_config.json")
        self.data_config = copy.deepcopy(self.config)
        self.data_config["output"] = str(self.root / "data")
        self.manifest = self.read_json("data/dataset/manifest.json")
        self.all_records = self.manifest["records"]
        self.records = [r for r in self.all_records if r["split"] in ("train", "validation")]
        self.by_id = {r["id"]: copy.deepcopy(r) for r in self.records}
        self.feature_index = self.read_json("data/features/index.json")
        self.ix = self.feature_index
        self.norm = self.read_npz("data/features/normalization.npz")
        norm_meta = self.read_json("data/features/normalization.json")
        train_ids = {r["id"] for r in self.records if r["split"] == "train"}
        if (set(self.norm["train_ids"].tolist()) != train_ids
                or str(self.norm["signature"]) != self.ix["signature"]
                or norm_meta["fit_split"] != "train"
                or norm_meta["sha256"] != self.loaded["data/features/normalization.npz"]):
            raise ValueError("Train-only normalization provenance mismatch")
        run = self.read_json(self.run_relative + "/run_manifest.json")
        result = self.read_json(self.run_relative + "/training_result.json")
        if run["status"] != "completed" or run["training_result_sha256"] != self.loaded[self.run_relative + "/training_result.json"]:
            raise ValueError("Original predictor run is incomplete or corrupted")
        for name in ("model.py", "inputs.py", "training.py"):
            relative = "experiments/predictor/control/src/phase1/" + name
            if sha256(ROOT / relative) != run["source_sha256"][relative]:
                raise ValueError("Retained predictor source differs: " + relative)
        self.result = result

    def verified_path(self, relative):
        relative = str(relative)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or relative not in self.index:
            raise ValueError("Path is absent from the verified backup index")
        expected = self.index[relative]
        actual = sha256(path)
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Backup file checksum/size mismatch: " + relative)
        self.loaded[relative] = actual
        return path

    def read_json(self, relative):
        return json.loads(self.verified_path(relative).read_text())

    def read_npz(self, relative, keys=None):
        # Refuse opening test arrays, including diagnostic and observation files.
        if any(part.startswith("test_") for part in Path(relative).parts):
            raise ValueError("This development diagnostic does not open test arrays")
        path = self.verified_path(relative)
        with np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k].copy() for k in (archive.files if keys is None else keys)}
        self.array_access.append(str(relative))
        return values

    def _record(self, rec):
        if rec.get("split") not in ("train", "validation") or self.by_id.get(rec.get("id")) != rec:
            raise ValueError("Require an unchanged train/validation observation record")
        return rec

    def features(self, rec):
        rec = self._record(rec)
        if rec["id"] not in self._features:
            info = self.ix["episodes"][rec["id"]]
            relative = "data/features/" + info["path"]
            values = self.read_npz(relative)
            z = values["z"]
            if (self.loaded[relative] != info["sha256"] or list(z.shape) != info["shape"]
                    or z.dtype != np.float32 or not np.isfinite(z).all()
                    or str(values["signature"]) != self.ix["signature"]
                    or str(values["encoder_hash"]) != self.ix["encoder_hash"]):
                raise ValueError("Feature identity mismatch")
            self._features[rec["id"]] = ((z-self.norm["mean"])/self.norm["std"]).astype(np.float32)
        return self._features[rec["id"]].copy()

    def actions(self, rec):
        rec = self._record(rec)
        return self.read_npz("data/dataset/" + rec["obs"], ("actions",))["actions"]

    def evidence(self, rec):
        """Privileged diagnostic timestamps only; never use to train retention."""
        rec = self._record(rec)
        return self.read_npz("data/dataset/" + rec["eval"], ("evidence",))["evidence"]

    def basis(self, task):
        if task not in ("gate", "motion"):
            raise ValueError("Unknown task")
        if task not in self._bases:
            basis = self.read_npz(f"{self.run_relative}/predictor/{task}/basis.npz")
            ids = {r["id"] for r in self.records if r["task"] == task and r["split"] == "train"}
            if set(basis["train_ids"].tolist()) != ids:
                raise ValueError("Basis must fit only this task's original training episodes")
            self._bases[task] = basis
        return {k: v.copy() for k, v in self._bases[task].items()}

    def checkpoint(self, task, kind="frozen"):
        if task not in ("gate", "motion") or kind not in ("frozen", "best", "latest"):
            raise ValueError("Unknown task/checkpoint")
        relative = f"{self.run_relative}/predictor/{task}/seed_17/{kind}.pt"
        path = self.verified_path(relative)
        # best/latest include NumPy RNG state. These exact bytes were produced
        # by this project's previous training run and verified against its export.
        value = torch.load(path, map_location="cpu", weights_only=(kind == "frozen"))
        signature = value["signature"]
        if (signature["task"] != task or signature["seed"] != 17
                or signature["cache_signature"] != self.ix["signature"]
                or signature["normalization_sha256"] != self.loaded["data/features/normalization.npz"]):
            raise ValueError("Checkpoint/cache binding mismatch")
        basis = self.basis(task)
        if signature["basis_sha256"] != self.loaded[f"{self.run_relative}/predictor/{task}/basis.npz"]:
            raise ValueError("Checkpoint/basis binding mismatch")
        for key in ("input_mean", "projection", "target_mean", "output_basis"):
            if not np.array_equal(value["model"][key].numpy(), basis[key]):
                raise ValueError("Checkpoint fitted transform differs: " + key)
        return value

    def model(self, task, kind="frozen"):
        if kind not in ("initial", "frozen", "best", "latest"):
            raise ValueError("Unknown checkpoint kind")
        basis = self.basis(task)
        seed_all(17, 4)
        model = DynamicsPredictor(len(self.norm["mean"]), self.config["model"], (3, 6), basis).cpu()
        meta = {"task": task, "kind": kind, "seed": 17}
        if kind != "initial":
            ck = self.checkpoint(task, kind)
            model.load_state_dict(ck["model"])
            meta["epoch"] = ck.get("epoch", ck.get("selected_epoch"))
        else:
            meta.update(epoch=0, reconstructed=True, exact_reconstruction_requires_score_replay=True)
        model.eval().requires_grad_(False)
        meta["state_sha256"] = state_hash(model)
        return model, meta

    def training_log(self, task):
        if task not in ("gate", "motion"):
            raise ValueError("Unknown task")
        return self.read_json(f"{self.run_relative}/predictor/{task}/seed_17/training.json")

    def audit(self):
        # Recheck consumed bytes at the end; input mutation invalidates a run.
        if sha256(self.root / "backup-index.json") != BACKUP_INDEX_SHA256:
            raise ValueError("Backup index mutated during analysis")
        for relative, expected in list(self.loaded.items()):
            if sha256(self.root / relative) != expected:
                raise ValueError("Original artifact mutated during analysis: " + relative)
        return {"backup_index_sha256": BACKUP_INDEX_SHA256, "loaded_file_sha256": dict(self.loaded),
                "array_files_opened": sorted(set(self.array_access)),
                "test_arrays_opened": 0, "originals_unchanged": True}
