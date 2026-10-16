"""Disclosed causal benchmark reconstruction for the incomplete SPACE release.

Policies receive current observations, never annotation or future payloads.
The evaluator uses labels only for scoring and prespecified interventions.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score

from space.controller import SpaceController
from space.memory import (MemoryState, counter_reservoir_action, enumerate_evictions,
                          execute_action, stable_video_seed)
from space.slow_basis import predictive_state_signatures
from space.utility import RetrievalDistribution, UtilityBank, build_action_features

METHODS = ('recent', 'fifo', 'reservoir', 'space', 'utility_only', 'state_only')
HORIZONS = (1, 4, 16, 64)
OFFSETS = (0, 1, 2, 4, 8, 16)


class CachedRetrieval:
    """Cache invariant bank normalization; retain released retrieval algebra."""
    def __init__(self, bank: UtilityBank, state_neighbors=32, action_neighbors=32):
        self.features = bank._normalize(bank.features)
        self.states = F.normalize(self.features.mean(dim=1), dim=-1)
        self.costs = bank.relative_costs
        self.ns = min(state_neighbors, len(self.states))
        self.na = min(action_neighbors, self.ns * self.features.shape[1])

    def retrieve(self, query):
        # One query set, exactly as the public method. Immutable bank tensors
        # are normalized once, avoiding repeated 0.8 GB copies per decision.
        q = UtilityBank._normalize(query)
        qs = F.normalize(q.mean(dim=0), dim=0)
        ids = (qs @ self.states.T).topk(self.ns).indices
        bank = self.features[ids].flatten(0, 1)
        costs = self.costs[ids].flatten()
        near = (q @ bank.T).topk(self.na, dim=-1)
        values = costs[near.indices]
        se = values.std(dim=-1, unbiased=True) / math.sqrt(self.na) if self.na > 1 else torch.zeros_like(values.mean(-1))
        return RetrievalDistribution(values.mean(-1), se, near.values.mean(-1))


class OnlineMethods:
    """Six matched policies sharing one immutable predictor, basis and bank."""
    def __init__(self, predictor, basis, bank, config, *, seed, video_id):
        self.predictor, self.basis, self.bank, self.config = predictor, basis, bank, config
        self.device = next(predictor.parameters()).device
        self.seed, self.video_id = int(seed), str(video_id)
        self.counter_seed = stable_video_seed(video_id, seed)
        self.seen = 0
        self.memories = {m: MemoryState.empty(config.capacity, predictor.d_in, self.device) for m in METHODS}
        self.origins = {m: [-1] * config.capacity for m in METHODS}
        self.controllers = {
            'space': SpaceController(config.controller),
            'state_only': SpaceController(replace(config.controller, tau_utility=-float('inf'), tau_support=-float('inf'))),
        }
        self.last_predictions = None
        self.overrides = Counter()

    def clone(self):
        out = OnlineMethods(self.predictor, self.basis, self.bank, self.config, seed=self.seed, video_id=self.video_id)
        out.seen = self.seen
        out.memories = {m: v.clone() for m, v in self.memories.items()}
        out.origins = {m: list(v) for m, v in self.origins.items()}
        for m in out.controllers:
            out.controllers[m].basin = self.controllers[m].basin.clone()
        out.overrides = self.overrides.copy()
        out.last_predictions = None if self.last_predictions is None else self.last_predictions.clone()
        return out

    @torch.no_grad()
    def _predict(self, contexts, memories, valid, ages):
        return self.predictor(contexts, memories, valid, memory_age=ages).predictions.float()

    @torch.no_grad()
    def hold_predictions(self, local, current_time):
        mem = torch.stack([self.memories[m].values for m in METHODS])
        valid = torch.stack([self.memories[m].valid for m in METHODS])
        ages = torch.stack([(current_time-self.memories[m].timestamps).float().clamp_min(0) for m in METHODS])
        if self.predictor.memory_slots == self.config.capacity + 1:
            mem = F.pad(mem, (0, 0, 0, 1)); valid = F.pad(valid, (0, 1)); ages = F.pad(ages, (0, 1))
        self.last_predictions = self._predict(local.expand(len(METHODS), -1, -1), mem, valid, ages)
        return self.last_predictions

    def _apply(self, method, candidates, action, incoming_origin):
        before = self.origins[method] + [int(incoming_origin)]
        self.origins[method] = [v for i, v in enumerate(before) if i != action]
        self.memories[method] = execute_action(candidates, action)

    @torch.no_grad()
    def step(self, event, event_time, local, current_time, origin):
        event_id = self.seen; self.seen += 1
        adaptive = []
        selected = {}
        for m in METHODS[1:]:
            state = self.memories[m]
            if not state.full:
                slot = state.active_count
                self.memories[m] = state.insert_first_free(event, event_time, event_id)
                self.origins[m][slot] = int(origin)
                continue
            c = enumerate_evictions(state, event, event_time, event_id)
            ra = counter_reservoir_action(c, current_time=current_time, base_seed=self.counter_seed)
            if m == 'fifo':
                action = min(range(c.actions), key=lambda i: (int(c.timestamps[i]), i))
                self._apply(m, c, action, origin)
            elif m == 'reservoir':
                self._apply(m, c, ra, origin)
            else:
                adaptive.append((m, c, ra))
        if adaptive:
            memories = torch.cat([c.memories for _, c, _ in adaptive])
            ages = torch.cat([c.ages(current_time)[1] for _, c, _ in adaptive])
            valid = torch.ones(memories.shape[:2], dtype=torch.bool, device=self.device)
            if self.predictor.memory_slots == self.config.capacity + 1:
                memories = F.pad(memories, (0, 0, 0, 1)); valid = F.pad(valid, (0, 1)); ages = F.pad(ages, (0, 1))
            predictions = self._predict(local.expand(len(memories), -1, -1), memories, valid, ages)
            n = self.config.capacity + 1
            for j, (m, c, ra) in enumerate(adaptive):
                pred = predictions[j*n:(j+1)*n]
                signatures = predictive_state_signatures(self.predictor.project_inputs(pred), self.basis)
                if m == 'state_only':
                    z = torch.zeros(n, device=self.device)
                    distribution = RetrievalDistribution(z, z, torch.ones_like(z))
                else:
                    features = build_action_features(self.predictor, pred, c.items, local, c.ages(current_time)[0])
                    distribution = self.bank.retrieve(features)
                if m == 'utility_only':
                    allowed = [ra]
                    for a in range(n):
                        _, _, lower = distribution.conservative_advantage(ra, a, uncertainty_multiplier=self.config.utility.uncertainty_multiplier)
                        support = min(float(distribution.mean_similarity[ra]), float(distribution.mean_similarity[a]))
                        if lower >= self.config.controller.tau_utility and support >= self.config.controller.tau_support:
                            allowed.append(a)
                    action = min(allowed, key=lambda a: (float(distribution.mean_cost[a]), a))
                else:
                    decision = self.controllers[m].decide(signatures, distribution, reservoir_action=ra,
                                                        uncertainty_multiplier=self.config.utility.uncertainty_multiplier)
                    action = decision.action
                    self.controllers[m].commit(signatures, decision)
                    self.overrides[m] += int(decision.override)
                    self.overrides[m+'_boundary'] += int(decision.boundary_release)
                self._apply(m, c, action, origin)
                selected[m] = pred[action]
        # Reuse the selected candidate forecasts. Only the three simple policies
        # require another predictor call after their memory update.
        if len(selected) == 3:
            methods = METHODS[:3]
            mem = torch.stack([self.memories[m].values for m in methods])
            val = torch.stack([self.memories[m].valid for m in methods])
            ages = torch.stack([(current_time-self.memories[m].timestamps).float().clamp_min(0) for m in methods])
            if self.predictor.memory_slots == self.config.capacity+1:
                mem=F.pad(mem,(0,0,0,1)); val=F.pad(val,(0,1)); ages=F.pad(ages,(0,1))
            base = self._predict(local.expand(3,-1,-1),mem,val,ages)
            self.last_predictions = torch.stack([*base, *[selected[m] for m in METHODS[3:]]])
        else:
            self.hold_predictions(local, current_time)
        return self.last_predictions

    def corrupt(self, donor, indices):
        for m in METHODS[1:]:
            if not self.memories[m].full or not donor.memories[m].full:
                raise ValueError('corruption requires actual full clean/donor memories')
            self.memories[m].values[indices] = donor.memories[m].values[indices]
            for i in indices:
                self.origins[m][i] = donor.origins[m][i]
        # Timestamp/event IDs and basin/counter metadata intentionally remain
        # unchanged. Evaluator-only origin IDs track replaced payload content.
        self.last_predictions = None


def ranking_scores(labels, probabilities):
    y = np.asarray(labels, dtype=np.uint8).reshape(-1, 24)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1, 24)
    if not len(y) or p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError('complete finite matched label/probability rows required')
    ap = [float(average_precision_score(y[:, k], p[:, k])) if y[:, k].any() else 0. for k in range(24)]
    foreground = y.any(axis=1)
    lrap = float(label_ranking_average_precision_score(y, p))
    top = np.argsort(-p, axis=1, kind='stable')[:, :5]
    hits = np.take_along_axis(y, top, axis=1).sum(axis=1)
    recall = np.divide(hits, y.sum(axis=1), out=np.ones(len(y), dtype=float), where=y.sum(axis=1)>0)
    return {'mAP':float(np.mean(ap)), 'LRAP':lrap, 'Recall_at_5':float(recall.mean()),
            'background_fraction':float((~foreground).mean()),
            'foreground_LRAP':float(label_ranking_average_precision_score(y[foreground],p[foreground])) if foreground.any() else None,
            'foreground_Recall_at_5':float(recall[foreground].mean()) if foreground.any() else None,
            'class_AP':ap, 'known_targets':len(y)}


def recovery_anchors(labels):
    """Eligibility is frozen independently of prediction outcomes."""
    y=np.asarray(labels,dtype=bool); candidates=[]
    first_for_key={}
    for t in range(23, len(y)-64-16):
        key=tuple(np.flatnonzero(y[t]))
        if not key:continue
        # Only earlier full states enter the donor pool.
        earlier=[v for k,v in first_for_key.items() if k != key]
        if earlier and np.all(y[t:t+17] == y[t]) and t+16+64<len(y):
            candidates.append((t,min(earlier)))
        first_for_key.setdefault(key,t)
    if not candidates:return []
    indexes=np.unique(np.rint(np.linspace(0,len(candidates)-1,min(4,len(candidates)))).astype(int))
    return [candidates[i] for i in indexes]


def memory_distance(clean, perturbed, method):
    if method=='recent':return float('nan')
    a=Counter(clean.origins[method]); b=Counter(perturbed.origins[method])
    return 1.-sum((a & b).values())/clean.config.capacity


def normalized_recovery(distances):
    d=np.asarray(distances,dtype=float)
    if d.shape!=(6,) or not np.isfinite(d).all():raise ValueError('all six recovery offsets required')
    rec=(d[0]-d)/max(float(d[0]),1e-8)
    return {'initial_distance':float(d[0]),'final_distance':float(d[-1]),
            'denominator_floored':bool(d[0]<1e-8),'recovery':rec.tolist(),
            'normalized_auc':float(np.trapz(rec[1:],x=np.asarray(OFFSETS[1:]))/15.)}


def participant_pair_inference(gains):
    values=np.asarray(gains,dtype=np.float64)
    if len(values)!=10 or not np.isfinite(values).all():raise ValueError('all ten validation participants required')
    rng=np.random.default_rng(0)
    means=values[rng.integers(0,10,size=(10000,10))].mean(axis=1)
    signs=1-2*((np.arange(1<<10)[:,None] >> np.arange(10)) & 1)
    null=(signs*values).mean(axis=1)
    return {'mean_gain':float(values.mean()),'CI95':np.quantile(means,[.025,.975]).tolist(),
            'exact_one_sided_sign_flip_p':float((null >= values.mean()-1e-15).mean()),
            'participants_positive':int((values>0).sum()),'participants':10,
            'bootstrap_replicates':10000,'bootstrap_seed':0}
