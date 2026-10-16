"""Causal payload ownership; only IDs survive eviction for audit purposes."""
import random


class OnlineMemory:
    def __init__(self, policy, seed, capacity=16, rank=None):
        self.policy, self.capacity, self.rank = policy, capacity, rank
        self.rng = random.Random(seed)
        self.past, self.current = [], None
        self.seen, self.ids, self.deleted, self.trace = 0, set(), set(), []

    def observe(self, entry):
        assert entry['id'] not in self.ids, 'Arrival ID reused'
        self.ids.add(entry['id'])
        if self.current is not None:
            incoming = self.current
            self.seen += 1
            candidates = self.past + [incoming]
            if self.policy == 'Archive_reference' or len(candidates) <= self.capacity:
                self.past = candidates
            elif self.policy == 'FIFO':
                self.past = candidates[-self.capacity:]
            elif self.policy == 'Reservoir':
                j = self.rng.randrange(self.seen)
                if j < self.capacity:
                    self.past[j] = incoming
            elif self.policy == 'FAR_current':
                chosen = self.rank(candidates, entry)
                assert len(chosen) == len(set(chosen)) == self.capacity
                assert all(0 <= i < len(candidates) for i in chosen)
                self.past = [candidates[i] for i in chosen]
            else:
                raise ValueError(self.policy)
            kept = {x['id'] for x in self.past}
            removed = {x['id'] for x in candidates} - kept
            assert not kept & self.deleted, 'Evicted payload re-entered'
            self.deleted.update(removed)
            self.trace.append({'arrived': entry['id'], 'offered': incoming['id'],
                               'kept': sorted(kept), 'deleted': sorted(removed)})
        self.current = entry
        if self.policy != 'Archive_reference':
            assert len(self.past) <= self.capacity
        assert len({x['id'] for x in self.past}) == len(self.past)

    def read(self):
        return sorted(self.past, key=lambda x: x['frame']) + [self.current]
