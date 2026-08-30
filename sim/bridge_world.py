"""L2 literal-pheromone simulation core: Deneubourg double-bridge + ring world.

Reinforcement = pheromone deposition on edges (per-tick, per-agent).
Evaporation   = global decay of all edge scents by lambda every `evap_every` ticks.
Follow rule   = choice prob ∝ (scent + eps)^beta, with weak outbound merit bias
                (beta=2, Deneubourg-style superlinear response).
The ring is a closed loop attached to the fork via a neck: agents entering it
circulate; each lap at the junction J they may exit (exit competes with staying
by scent). The ring is the physical embodiment of the ant mill.
"""
import numpy as np
from collections import deque


class BridgeWorld:
    """Graph world: NEST -> fork A -> {lane0 (L0 edges), lane1 (L0+2*delta edges)} -> B -> FOOD.
    A also connects via a 2-edge neck to a ring of R nodes; ring exits back to NEST."""

    def __init__(self, delta=3, L0=20, R=12):
        self.edges = []
        def add(u, v):
            self.edges.append((u, v)); return len(self.edges) - 1
        self.NEST, self.A, self.B, self.FOOD = 0, 1, 2, 3
        add(self.NEST, self.A)
        nid = 4
        def chain(u, k):
            prev, els = u, []
            nonlocal nid
            for _ in range(k - 1):
                els.append(add(prev, nid)); prev = nid; nid += 1
            return prev, els
        last, self.lane0_e = chain(self.A, L0)
        add(last, self.B); self.lane0_e.append(len(self.edges) - 1)
        last, self.lane1_e = chain(self.A, L0 + 2 * delta)
        add(last, self.B); self.lane1_e.append(len(self.edges) - 1)
        add(self.B, self.FOOD)
        self.J = nid; nid += 1
        add(self.A, nid); self.neck_in = [len(self.edges) - 1]
        add(nid, self.J); self.neck_in.append(len(self.edges) - 1); nid += 1
        self.ring_nodes = [self.J]; prev = self.J; self.ring_e = []
        for _ in range(R - 1):
            self.ring_e.append(add(prev, nid)); self.ring_nodes.append(nid); prev = nid; nid += 1
        self.ring_e.append(add(prev, self.J))
        add(self.J, nid); self.neck_out = [len(self.edges) - 1]
        add(nid, self.NEST); self.neck_out.append(len(self.edges) - 1)
        self.n_nodes = nid
        self.adj = {}
        for eid, (u, v) in enumerate(self.edges):
            self.adj.setdefault(u, []).append((eid, v))
            self.adj.setdefault(v, []).append((eid, u))
        self.n_edges = len(self.edges)
        self.lane0_set = set(self.lane0_e)
        self.lane1_set = set(self.lane1_e)
        self.ring_nset = set(self.ring_nodes)


def bfs_dist(w, target):
    dist = {target: 0}; q = deque([target])
    while q:
        u = q.popleft()
        for _, v in w.adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def run_bridge(delta=3, lam=1.0, N=8, T=8000, seed=0, eps=0.5, dep=1.0,
               w_merit=1.2, beta=2.0, ring_R=12, evap_every=10):
    """One agent-based simulation run. Returns traffic shares, mill fraction, category."""
    rng = np.random.default_rng(seed)
    w = BridgeWorld(delta=delta, R=ring_R)
    d_food = bfs_dist(w, w.FOOD)
    scent = np.zeros(w.n_edges)
    pos = [w.NEST] * N
    came_e = [None] * N
    homing = [False] * N
    last_lane = [0] * N
    succ = []
    ring_hits = np.zeros(N)
    lam_chunk = lam ** evap_every
    for t in range(T):
        for i in range(N):
            u = pos[i]
            cands = [(e, v) for (e, v) in w.adj[u] if e != came_e[i]] or [(e, v) for (e, v) in w.adj[u]]
            weights = []
            for e, v in cands:
                wgt = (scent[e] + eps) ** beta
                if not homing[i]:  # weak merit bias, outbound only (homing = pure scent)
                    dd = d_food[v] - d_food[u]
                    wgt *= w_merit if dd < 0 else (1.0 if dd == 0 else 1.0 / w_merit)
                weights.append(wgt)
            weights = np.array(weights); weights /= weights.sum()
            k = rng.choice(len(cands), p=weights)
            e, v = cands[k]
            came_e[i], pos[i] = e, v
            scent[e] += dep
            if e in w.lane0_set: last_lane[i] = 0
            elif e in w.lane1_set: last_lane[i] = 1
            if v == w.FOOD and not homing[i]:
                homing[i] = True; succ.append((t, last_lane[i]))
            elif v == w.NEST and homing[i]:
                homing[i] = False
            if t >= T - 500 and v in w.ring_nset:
                ring_hits[i] += 1
        if t % evap_every == 0:
            scent *= lam_chunk
    win = [l for (tt, l) in succ if tt >= T - 500]
    milling = float((ring_hits / 500 > 0.6).mean())
    s0, s1 = win.count(0), win.count(1)
    tot = s0 + s1
    sh0 = s0 / tot if tot else 0.0
    sh1 = s1 / tot if tot else 0.0
    if milling >= 0.5: cat = 'mill'
    elif sh0 >= 0.7: cat = 'optimal'
    elif sh1 >= 0.7: cat = 'detour'
    else: cat = 'mixed'
    return dict(milling=milling, sh0=sh0, sh1=sh1, trips=len(succ), cat=cat)
