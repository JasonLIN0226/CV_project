"""Original repair train operations with exact resident input gathering only."""
import time
import numpy as np
import torch
from phase1.training import TRAIN_CONDITIONS, regression_loss
from phase1.common_anchor import recent_only_teacher_batch, common_anchor_loss


def train_epoch(c, store, model, optimizer, pairs, seed, epoch, device, task, stop, deadline):
    model.train()
    rng = np.random.default_rng(seed + 100003 * epoch)
    order = rng.permutation(len(pairs))
    counts = {k: 0 for k in TRAIN_CONDITIONS}
    totals = {k: np.zeros(4, dtype=np.float64) for k in TRAIN_CONDITIONS}
    steps = 0
    for start in range(0, len(order), 32):
        selected = [pairs[i] for i in order[start:start + 32]]
        rows = [r for pair in selected for r in pair]
        for condition in rng.permutation(TRAIN_CONDITIONS):
            sample_seeds = [int(rng.integers(2 ** 32)) for _ in selected]
            batch, target, _ = store.packed_train.get(selected, str(condition), sample_seeds)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss, mse, paired = regression_loss(prediction, target, 2., .5)
            anchor = prediction.new_zeros(())
            if task == 'gate' and condition != 'recent_only':
                with torch.no_grad():
                    teacher = model.reference(recent_only_teacher_batch(batch))
                anchor = common_anchor_loss(prediction, teacher, rows, condition=str(condition))
                loss = loss + anchor
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            n = len(rows)
            counts[str(condition)] += n
            totals[str(condition)] += np.asarray([float(x.detach()) for x in (loss, mse, paired, anchor)]) * n
            if stop['requested'] or time.perf_counter() >= deadline:
                return None
    if counts != {k: 1024 for k in TRAIN_CONDITIONS} or steps != 64:
        raise ValueError('Incomplete balanced epoch')
    store.packed_train.verify_unchanged()
    return {'steps': steps, 'counts': counts,
            'means': {k: (totals[k] / counts[k]).tolist() for k in TRAIN_CONDITIONS}}
