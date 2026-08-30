"""L1 mean-field theory: two lanes + ring capture ODE.

State: s0, s1 (lane scent stocks), sr (ring scent stock), sx (exit-neck scent), nr (ring population).

    ds_i = F * p_i * tau_i - kappa * s_i          (lane scent: deposition vs evaporation)
    dsr  = nr - kappa * sr                        (ring: captives deposit every tick)
    dsx  = (escape + F * a_frac) - kappa * sx     (exit neck: escapees + ambient wanderers)
    dnr  = F * p_neck - nr * q_exit / tau_r       (capture vs escape)

UNITS (fixed 2026-08): lambda is the PER-TICK scent retention, identical to the
sim (bridge_world.py: scent *= lam**10 every 10 ticks, i.e. lam per tick).
The ODE integrates per tick, so kappa = -ln(lam) per tick. An earlier revision
used kappa = -ln(lam)/10, which silently made theory evaporation 10x weaker
than the sim at the same lambda; the qualitative picture below survives the
correction, but all absolute lifetimes shrink (see __main__ table).

Choice: p_i ∝ (s_i / L_i + eps)^beta * merit_i;  q_exit analogous at ring junction J.

Key findings from this model (corrected kappa):
- Deterministic mean-field gives STABILITY (attractor map), not FORMATION (needs fluctuations).
- Mill = metastable attractor with a critical-mass separatrix: below it, collapse feedback
  (fewer agents -> weaker scent -> faster escape); above, self-sustaining.
- Mill lifetime tau_mill(lambda, beta): ~100 ticks or less for beta=1.5 regardless of lambda
  (mills cannot live); grows with lambda for beta >= 2 and explodes for beta = 2.5.
- Sim's mill band = formation window: lower edge = mill viability (lifetime > horizon),
  upper edge = race (lane lock-in strangles neck inflow before ring reaches critical mass).
"""
import numpy as np


def meanfield_run(lam=0.95, delta=2, N=8, L0=20, R=12, eps=0.5, dep=1.0,
                  w=1.2, beta=2.0, T=30000, a_frac=0.08, init=None):
    """Deterministic mean-field integration (Euler, per-tick). Returns (p1, nr/N)."""
    kappa = -np.log(lam)  # lam = per-tick retention, one ODE step = one tick
    L1 = L0 + 2 * delta
    tau0, tau1, tau_r = 2 * L0, 2 * L1, float(R)
    st = dict(s0=eps * 0.1, s1=eps * 0.1, sr=eps * 0.1, sx=eps * 0.1, nr=0.0)
    if init:
        st.update(init)
    s0, s1, sr, sx, nr = st['s0'], st['s1'], st['sr'], st['sx'], st['nr']
    for t in range(T):
        n_tr = max(N - nr, 0.0); F = n_tr / 50.0
        w0 = (s0 / L0 + eps) ** beta * w
        w1 = (s1 / L1 + eps) ** beta / w
        wn = eps ** beta / w
        Z = w0 + w1 + wn
        p0, p1, pn = w0 / Z, w1 / Z, wn / Z
        ring_edge = sr / R
        q_exit = ((sx + eps) ** beta * w) / (((sx + eps) ** beta * w) + ((ring_edge + eps) ** beta / w))
        escape = nr * q_exit / tau_r
        s0 += F * p0 * tau0 * dep - kappa * s0
        s1 += F * p1 * tau1 * dep - kappa * s1
        sr += nr * dep - kappa * sr
        sx += (escape + F * a_frac) * dep - kappa * sx
        s0, s1, sr, sx = max(s0, 0), max(s1, 0), max(sr, 0), max(sx, 0)
        nr = min(max(nr + F * pn - escape, 0.0), float(N))
    p1 = w1 / (w0 + w1 + eps ** beta / w)
    return p1, nr / N


def mill_lifetime(lam, beta=2.0, delta=2, N=8, nr0=4.0, sr0=30.0, T_max=200000):
    """Lifetime of a formed mill: ticks until ring population drops below 0.5."""
    kappa = -np.log(lam)  # lam = per-tick retention (matches sim semantics)
    L0, R, eps, dep, w = 20, 12, 0.5, 1.0, 1.2
    L1 = L0 + 2 * delta
    tau0, tau1, tau_r = 2 * L0, 2 * L1, float(R)
    s0, s1, sx = eps * 0.1, eps * 0.1, 1.0
    sr, nr = sr0, nr0
    for t in range(T_max):
        n_tr = max(N - nr, 0.0); F = n_tr / 50.0
        w0 = (s0 / L0 + eps) ** beta * w
        w1 = (s1 / L1 + eps) ** beta / w
        wn = eps ** beta / w
        Z = w0 + w1 + wn; pn = wn / Z
        ring_edge = sr / R
        q_exit = ((sx + eps) ** beta * w) / (((sx + eps) ** beta * w) + ((ring_edge + eps) ** beta / w))
        escape = nr * q_exit / tau_r
        s0 += F * (w0 / Z) * tau0 * dep - kappa * s0
        s1 += F * (w1 / Z) * tau1 * dep - kappa * s1
        sr += nr * dep - kappa * sr
        sx += (escape + F * 0.08) * dep - kappa * sx
        s0, s1, sr, sx = max(s0, 0), max(s1, 0), max(sr, 0), max(sx, 0)
        nr = min(max(nr + F * pn - escape, 0.0), float(N))
        if nr < 0.5:
            return t
    return T_max


if __name__ == '__main__':
    # Mill lifetime vs lambda for several beta — the paper's core theory curve
    lams = [0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
    for beta in [1.5, 2.0, 2.5]:
        taus = [mill_lifetime(l, beta=beta) for l in lams]
        print(f'beta={beta}: ' + ' '.join(f'{t:>7d}' for t in taus))
    print('lambda: ' + ' '.join(f'{l:>7.2f}' for l in lams))
