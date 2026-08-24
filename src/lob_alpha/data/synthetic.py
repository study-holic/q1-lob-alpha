"""Synthetic order book simulator with a known ground truth.

This exists for one reason: to check that the research machinery finds a signal
when one is there, and finds nothing when nothing is there. It is not a claim
about how markets work.

Three regimes of ground truth:

* ``null``   : book state carries no information about future mid moves.
* ``ofi``    : the next mid move is driven by realised order flow imbalance
               with a known coefficient ``beta``.
* ``regime`` : the same OFI relationship, but active only when the spread is
               wide, so regime conditioning has something to discover.

Construction order matters. At each event the book updates first, the OFI
increment implied by that update is computed from the observed book, and only
then does the *next* mid move react to it. Nothing at time t is a function of
anything after t, so a look-ahead bug in the pipeline shows up as an
impossibly good result rather than a plausible one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_book(
    n_events: int = 40_000,
    n_sessions: int = 1,
    instrument: str = "SYN1",
    mode: str = "ofi",
    beta: float = 0.12,
    tick: float = 0.01,
    start_price: float = 100.0,
    base_size: float = 250.0,
    seed: int = 0,
    start: str = "2026-01-05 08:00:00",
    mean_interarrival_ms: float = 50.0,
    return_labels: bool = False,
):
    """Simulate a level-1 book in event time. Returns the canonical schema.

    ``n_sessions`` splits ``n_events`` across that many trading days, each on
    its own date and separated by an overnight price gap. Anything less than a
    multi-session panel cannot exercise the session boundary logic, and a
    single continuous stream is exactly the shape of data that hides an
    overnight-return bug.

    ``return_labels`` additionally returns the true latent regime state. It is
    kept out of the frame by default so that no downstream stage can read the
    answer off the data; regimes must be inferred from observable book state,
    exactly as on real data.
    """
    if mode not in {"null", "ofi", "regime"}:
        raise ValueError(f"unknown mode: {mode}")
    if n_sessions > 1:
        frames, labels, price = [], [], start_price
        per_session = n_events // n_sessions
        gap_rng = np.random.default_rng(seed + 99991)
        for k in range(n_sessions):
            day = (pd.Timestamp(start) + pd.Timedelta(days=k)).strftime("%Y-%m-%d 08:00:00")
            result = simulate_book(
                n_events=per_session, n_sessions=1, instrument=instrument, mode=mode,
                beta=beta, tick=tick, start_price=price, base_size=base_size,
                seed=seed + 7919 * k, start=day, mean_interarrival_ms=mean_interarrival_ms,
                return_labels=True,
            )
            frame, label = result
            frames.append(frame)
            labels.append(label)
            # Overnight gap: the price moves while nobody is looking, which is
            # precisely why a forward return must never span the boundary.
            price = float(frame["ask_price"].iloc[-1]) * float(np.exp(gap_rng.normal(0.0, 0.004)))
        out = pd.concat(frames, ignore_index=True)
        return (out, np.concatenate(labels)) if return_labels else out
    rng = np.random.default_rng(seed)

    bid_ticks = np.empty(n_events, dtype=np.int64)
    ask_ticks = np.empty(n_events, dtype=np.int64)
    bid_size = np.empty(n_events)
    ask_size = np.empty(n_events)
    regime_state = np.zeros(n_events, dtype=np.int8)

    b = int(round(start_price / tick))
    a = b + 1
    qb = base_size
    qa = base_size
    ofi_state = 0.0  # exponentially decayed order flow pressure
    state = 0  # latent regime, persistent by construction
    decay = 0.90
    switch_prob = 1.0 / 2000.0

    for t in range(n_events):
        # --- latent regime: persistent, so it is discoverable from the book --
        if rng.random() < switch_prob:
            state = 1 - state

        # --- queue dynamics: arrivals, cancellations, consumption -----------
        qb = max(1.0, qb * rng.uniform(0.75, 1.25) + rng.normal(0.0, 0.08 * base_size))
        qa = max(1.0, qa * rng.uniform(0.75, 1.25) + rng.normal(0.0, 0.08 * base_size))

        # --- price dynamics -------------------------------------------------
        # Baseline: symmetric random tick moves. Predictable part: pressure
        # accumulated from *past* order flow, scaled by the active beta.
        if mode == "ofi":
            active_beta = beta
        elif mode == "regime":
            active_beta = beta if state == 1 else 0.0
        else:
            active_beta = 0.0

        drift = active_beta * np.tanh(ofi_state / (3.0 * base_size))
        u = rng.random()
        p_up = 0.5 * (0.10 + drift)
        p_dn = 0.5 * (0.10 - drift)
        if u < p_up:
            b += 1
            qb, qa = max(1.0, qa * rng.uniform(0.4, 0.9)), base_size * rng.uniform(0.6, 1.4)
        elif u < p_up + p_dn:
            b -= 1
            qa, qb = max(1.0, qb * rng.uniform(0.4, 0.9)), base_size * rng.uniform(0.6, 1.4)

        # --- spread process ---------------------------------------------------
        # Widening is symmetric about the mid: the book goes from (b, b+1) to
        # (b-k, b+1+k), so the spread is 1 + 2k ticks and the mid never moves
        # because of a spread change. That matters. If widening shifted the
        # mid, the simulator would manufacture bid-ask bounce, every signal
        # touching the ask would inherit a mechanical negative IC, and the null
        # control would stop being a null.
        #
        # In regime mode the informative state is also the illiquid one, so the
        # regime is inferable from observable book state rather than being a
        # hidden label only the simulator knows.
        if mode == "regime":
            k = int(rng.integers(1, 3)) if state == 1 else 0
        else:
            k = int(rng.integers(1, 3)) if rng.random() < 0.15 else 0
        a = b + 1 + k
        bid = b - k

        bid_ticks[t] = bid
        ask_ticks[t] = a
        bid_size[t] = qb
        ask_size[t] = qa
        regime_state[t] = state

        # --- order flow pressure implied by the book we just wrote ----------
        if t == 0:
            increment = 0.0
        else:
            increment = _ofi_increment(
                bid_ticks[t - 1], bid_size[t - 1], bid_ticks[t], bid_size[t],
                ask_ticks[t - 1], ask_size[t - 1], ask_ticks[t], ask_size[t],
            )
        ofi_state = decay * ofi_state + increment

    dt_ms = rng.exponential(mean_interarrival_ms, n_events)
    timestamp = pd.Timestamp(start) + pd.to_timedelta(np.cumsum(dt_ms), unit="ms")

    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "instrument": instrument,
            "bid_price": bid_ticks * tick,
            "bid_size": bid_size,
            "ask_price": ask_ticks * tick,
            "ask_size": ask_size,
        }
    )
    return (out, regime_state) if return_labels else out


def _ofi_increment(pb0, qb0, pb1, qb1, pa0, qa0, pa1, qa1) -> float:
    """Scalar version of the Cont, Kukanov and Stoikov OFI increment."""
    e_b = (qb1 if pb1 >= pb0 else 0.0) - (qb0 if pb1 <= pb0 else 0.0)
    e_a = (qa1 if pa1 <= pa0 else 0.0) - (qa0 if pa1 >= pa0 else 0.0)
    return e_b - e_a


def simulate_panel(
    n_instruments: int = 4,
    n_events: int = 40_000,
    mode: str = "ofi",
    beta: float = 0.12,
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """A panel of independent instruments, concatenated into one frame."""
    frames = []
    for i in range(n_instruments):
        frames.append(
            simulate_book(
                n_events=n_events,
                instrument=f"SYN{i + 1}",
                mode=mode,
                beta=beta,
                seed=seed + 1000 * i,
                **kwargs,
            )
        )
    return pd.concat(frames, ignore_index=True)
