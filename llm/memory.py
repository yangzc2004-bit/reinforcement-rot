"""Shared memory: the L3 analog of the pheromone field.

Entries are keyed by maze cell (the state variable). Each entry carries a piece
of advice (suggested direction + free text) and a weight.

  reinforcement : agents write entries / bump weights (dep = deposit)
  evaporation   : every round, all weights *= lam; entries below `floor` die
  retrieval     : top-k by weight at the current cell — the sharp cutoff is the
                  LLM analog of beta -> infinity in the follow rule

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
        self.entries = []  # list of dict(cell, direction, text, weight, writer, round)

    def retrieve(self, cell, k=None):
        k = k or self.top_k
        hits = [e for e in self.entries if e['cell'] == list(cell)]
        hits.sort(key=lambda e: -e['weight'])
        return hits[:k]

    def reinforce(self, cell, direction, text, writer, round_id, amount=1.0):
        for e in self.entries:
            if e['cell'] == list(cell) and e['direction'] == direction:
                e['weight'] += amount
                return
        self.entries.append(dict(cell=list(cell), direction=direction, text=text,
                                 weight=amount, writer=writer, round=round_id))

    def evaporate(self):
        for e in self.entries:
            e['weight'] *= self.lam
        self.entries = [e for e in self.entries if e['weight'] >= self.floor]

    def save(self, path):
        Path(path).write_text(json.dumps(self.entries, indent=1))

    @classmethod
    def load(cls, path, **kw):
        m = cls(**kw)
        m.entries = json.loads(Path(path).read_text())
        return m
