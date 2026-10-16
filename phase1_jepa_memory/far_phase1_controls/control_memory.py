"""Strict causal explicit-memory controls; metadata traces cannot provide payloads."""
class ControlMemory:
    def __init__(self, policy):
        assert policy in ('Current_only','Recent_only')
        self.policy=policy; self.capacity=0 if policy=='Current_only' else 16
        self.current=None; self.past=[]; self.seen=set(); self.deleted=set(); self.trace=[]; self.read_trace=[]
    def observe(self, entry):
        assert entry['id'] not in self.seen
        if self.current is not None:
            candidates=self.past+[self.current]
            retained=[] if self.capacity==0 else candidates[-self.capacity:]
            kept={x['id'] for x in retained}
            removed={x['id'] for x in candidates}-kept
            assert not kept & self.deleted
            self.trace.append({'arrived':entry['id'],'offered':self.current['id'],'kept':[x['id'] for x in retained],'deleted':sorted(removed)})
            self.deleted.update(removed); self.past=retained
        self.seen.add(entry['id']);self.current=entry
        assert len(self.past)<=self.capacity
    def read(self):
        assert self.current is not None
        recent=[] if self.capacity==0 else self.past[-11:]
        return recent+[self.current]
