"""environments/motion_v2/fresh.py: the copied generator reproduces the registered scenes; fresh scenes are new."""
import numpy as np
import pytest

from bounded_memory.environments.motion_v2 import envs_v2, fresh

CONFIG = {"generator_version": envs_v2.GENERATOR_VERSION, "data": envs_v2.REGISTERED_DATA}


def _same(a, b):
    assert set(a) == set(b)
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("split,index", [("train", 0), ("train", 7), ("train", 511), ("validation", 0), ("validation", 95)])
def test_copies_reproduce_registered_scenes_and_episodes(split, index):
    registered = envs_v2.scene(CONFIG, "motion", split, index)
    copied = fresh.draw(envs_v2.REGISTERED_DATA, "motion", split, index)
    _same(registered, copied)
    for label in (0, 1):
        a, b = envs_v2.episode(registered, label), fresh.render_episode(copied, label)
        _same(a["observations"], b["observations"])
        _same(a["evaluation_only"], b["evaluation_only"])


def test_registered_ids_are_generate_v2s():
    pair, ids = fresh.pair_ids("train", 3)
    assert pair == envs_v2.digest([20261030, "motion", "train", 3])[:24]
    assert ids == [envs_v2.digest([pair, slot, "opaque-id"])[:24] for slot in range(2)]


def test_fresh_scenes_are_new_and_never_held_out():
    seeds = {fresh.scene(i)["seed"] for i in range(600)}
    assert len(seeds) == 600 and not seeds & fresh.registered_scene_seeds()
    for name in ("train", "validation", "test", "confirmation", "Test_2"):
        with pytest.raises(ValueError):
            fresh.scene(0, name)


def test_generate_writes_resumable_dataset(tmp_path):
    records = fresh.generate(tmp_path, 2, log=lambda *_: None)
    assert len(records) == 4 and {r["split"] for r in records} == {fresh.FRESH_SPLIT}
    first = (tmp_path / "dataset" / records[0]["obs"]).stat().st_mtime_ns
    again = fresh.generate(tmp_path, 2, log=lambda *_: None)
    assert again == records and (tmp_path / "dataset" / records[0]["obs"]).stat().st_mtime_ns == first
    with np.load(tmp_path / "dataset" / records[0]["obs"]) as f:
        assert set(f.files) == {"rgb", "actions", "query"} and f["rgb"].shape == (records[0]["frames"], 96, 96, 3)


def test_fresh_episodes_feed_the_deletion_cost_pipeline(tmp_path):
    """generate -> encode (stand-in encoder) -> episode -> pair table, behaviour states and scorer inputs."""
    from types import SimpleNamespace

    import phase1.model as phase1_model
    from bounded_memory.environments.motion_v2 import deletion_cost_v2 as dc
    from motion_v2_fakes import D, FakeAssets

    class StandInEncoder:
        def __init__(self, config, device):
            self.metadata = {"resolved_revision": config["revision"]}
            self.rng = np.random.default_rng(0)

        def encode(self, rgb):
            assert rgb.shape[1:] == (96, 96, 3) and len(rgb) <= 2
            return (self.rng.normal(size=(len(rgb), D)) / 30).astype(np.float32)

    assets = FakeAssets()
    assets.config = {"encoder": {"revision": "pinned", "batch_size": 2}}
    assets.store = SimpleNamespace(norm={"mean": np.zeros(D, np.float32), "std": np.ones(D, np.float32)})
    modules = {"model": phase1_model, "encoders": SimpleNamespace(FrozenIJepa=StandInEncoder)}
    assets.reader = SimpleNamespace(load_reference=modules.__getitem__)
    records = fresh.generate(tmp_path, 2, log=lambda *_: None)
    index = fresh.encode(assets, tmp_path, records, "cpu", log=lambda *_: None)
    assert set(index) == {r["id"] for r in records}
    ep = fresh.episode(assets, tmp_path, records[0])
    q = ep["rec"]["query"]
    assert ep["z"].shape == (records[0]["frames"], D) and ep["prefix"].shape == (records[0]["frames"], 2)
    assert np.allclose(ep["p"], phase1_model.project(ep["z"], assets.basis))
    assert set(ep["meta"]) == set(fresh.EVAL_KEYS) and ep["meta"]["sufficient"].any()
    table = dc.pair_table(assets, ep)
    assert table.shape == (2, q - 3, q - 3) and np.isfinite(table[:, 0, 1]).all()
    states = dc.behaviour_states(assets, ep, 0)
    items, context, prior = dc.observe(assets, ep, *states[0])
    assert items.shape == (3, dc.ITEM_DIM) and context.shape == (dc.CONTEXT_DIM,) and prior.shape == (3,)
