"""Synthetic software checks only; no real data, model fitting or science."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from latent_dataset import PredictorTrainingDataset, collate_predictor, LatentCorpus


class CausalDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = []
        for split in ["train", "val"]:
            name = f"{split}.pt"
            row = {"video_id": split, "path": name, "split": split, "valid_length": 100, "subject_id": split}
            rows.append(row)
            torch.save({"video_id": split, "valid_length": 100,
                        "latents": torch.arange(100).float()[:, None].expand(-1, 3).clone()}, self.root / name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"test_features_accessed": False, "d_in": 3, "videos": rows}))

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, split="val", dropout=0.15):
        return PredictorTrainingDataset(manifest_path=self.manifest, split=split,
            local_context_len=8, memory_slots=17, horizons=[1,4,16,64], memory_dropout=dropout, seed=0)

    def test_every_anchor_has_complete_future(self):
        d = self.dataset()
        self.assertEqual(len(d), 29)
        self.assertEqual(d[len(d)-1].targets[:,0].tolist(), [36,39,51,99])

    def test_past_never_overlaps_recent_context(self):
        b = self.dataset()[24]
        self.assertTrue((b.memory[b.memory_valid,0] < b.local[0,0]).all())
        self.assertTrue((b.memory_age[b.memory_valid] >= 8).all())
        self.assertLessEqual(int(b.memory_valid.sum()), 16)
        self.assertFalse(b.memory_valid[-1])

    def test_validation_is_fixed_and_no_dropout(self):
        a, b = self.dataset(dropout=0.99)[24], self.dataset(dropout=0)[24]
        self.assertTrue(torch.equal(a.memory, b.memory))
        self.assertTrue(torch.equal(a.memory_valid, b.memory_valid))
        self.assertEqual(int(a.memory_valid.sum()),16)

    def test_future_mutation_cannot_change_model_inputs(self):
        d = self.dataset(); before=d[24]
        x=torch.load(self.root/'val.pt',weights_only=True)
        x['latents'][32:]=123456
        torch.save(x,self.root/'val.pt')
        after=self.dataset()[24]
        for name in ['local','memory','memory_age','local_age','memory_valid']:
            self.assertTrue(torch.equal(getattr(before,name),getattr(after,name)))
        self.assertFalse(torch.equal(before.targets,after.targets))

    def test_test_manifest_is_refused(self):
        m=json.loads(self.manifest.read_text());m['test_features_accessed']=True
        self.manifest.write_text(json.dumps(m))
        with self.assertRaises(ValueError): self.dataset()

    def test_collation_matches_author_batch_contract(self):
        d=self.dataset();b=collate_predictor([d[0],d[20]]).to('cpu')
        self.assertEqual(tuple(b.local.shape),(2,8,3))
        self.assertEqual(tuple(b.memory.shape),(2,17,3))
        self.assertEqual(tuple(b.targets.shape),(2,4,3))


if __name__ == '__main__': unittest.main()
