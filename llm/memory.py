"""Shared memory: the L3 analog of the pheromone field.

Entries are keyed by (cell, direction) — that pair is the STATE VARIABLE, the
L3 analog of scent on an edge. Weight aggregates on it; content (the natural
-language text) is evidence attached to it.

  reinforcement : agents bump the weight of (cell, direction) and APPEND their
                  text to the entry's history — the first writer's wording no
                  longer silently absorbs later writers' endorsements (the
                  "old text gets reinforced" artifact is removed)
  evaporation   : every EPISODE, all weights *= lam; entries below `floor` die.
                  lam is therefore a PER-EPISODE retention; with n_agents per
                  round, per-round retention = lam ** n_agents. (Cross-layer
                  mapping to L2's per-tick lam is qualitative, not numeric.)
  retrieval     : top-k by weight at the current cell. top_k is a candidate-set
                  truncation mechanism; we do NOT claim top_k = beta -> inf.
                  The model's effective follow sharpness is measured separately
                  by calib_beta.py (beta_eff).

Design law embodied here: reinforcement and evaporation act on the SAME state
variable (entry weight). A retrieval-time recency bias is NOT evaporation.
"""
import json
from pathlib import Path


class SharedMemory:
    def __init__(self, lam=1.0, top_k=5, floor=0.05):
        self.lam = lam
        self.top_k = top_k
        self.floor = floor
        # entry: dict(cell, direction, weight, n_reinforce,
        #             texts=[dict(text, writer, round)], )
        self.entries = []

    def retrieve(self, cell, k=None):
        k = k or self.top_k
        hits = [e for e in self.entries if e['cell'] == list(cell)]
        hits.sort(key=lambda e: -e['weight'])
        out = []
        for e in hits[:k]:
            out.append(dict(cell=e['cell'], direction=e['direction'],
                            weight=e['weight'],
                            text=e['texts'][-1]['text'],
                            n_writers=len({t['writer'] for t in e['texts']}),
                            n_reinforce=e['n_reinforce']))
        return out

    def reinforce(self, cell, direction, text, writer, round_id, amount=1.0):
        for e in self.entries:
            if e['cell'] == list(cell) and e['direction'] == direction:
                e['weight'] += amount
                e['n_reinforce'] += 1
                e['texts'].append(dict(text=text, writer=writer, round=round_id))
                return
        self.entries.append(dict(cell=list(cell), direction=direction,
                                 weight=amount, n_reinforce=1,
                                 texts=[dict(text=text, writer=writer,
                                             round=round_id)]))

    def evaporate(self):
        """Apply ONE episode's worth of evaporation: weights *= lam (per-episode)."""
        for e in self.entries:
            e['weight'] *= self.lam
        self.entries = [e for e in self.entries if e['weight'] >= self.floor]

    def half_life_episodes(self):
        """Memory half-life in episodes: log(0.5)/log(lam). inf when lam >= 1."""
        import math
        if self.lam >= 1.0:
            return float('inf')
        return math.log(0.5) / math.log(self.lam)

    def save(self, path):
        Path(path).write_text(json.dumps(self.entries, indent=1))

    @classmethod
    def load(cls, path, **kw):
        m = cls(**kw)
        entries = json.loads(Path(path).read_text())
        # backward compatibility: old entries had a single 'text' field
        for e in entries:
            if 'texts' not in e:
                e['texts'] = [dict(text=e.pop('text'), writer=e.pop('writer', -1),
                                   round=e.pop('round', -1))]
                e.setdefault('n_reinforce', 1)
        m.entries = entries
        return m
