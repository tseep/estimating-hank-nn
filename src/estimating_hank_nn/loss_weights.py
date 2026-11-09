"""Per-residual loss-weight schemes for neural-network training.

The *type* of the scheme selects the behaviour; both expose a uniform interface, so a training
loop stays scheme-agnostic::

    scheme  = config.weights            # ConstantWeights | ReLoBRaLo
    weights = scheme.initial(n_residuals)
    state   = None
    for i in ...:
        ...                              # compute per-residual losses
        weights, state = scheme.update(weights, losses, i, warmup_iter, state)

``losses`` is the list of current per-residual loss values; ``state`` carries any transient
bookkeeping (``None`` on the first call). The schemes are plain dataclasses — the transient
state lives in ``state``, not on the scheme — so they pickle cleanly into a saved config.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ConstantWeights:
    """Fixed per-residual loss weights (``None`` -> all ones)."""

    weights: Optional[list] = None

    def initial(self, n):
        return list(self.weights) if self.weights is not None else [1.0] * n

    def update(self, weights, losses, iteration, warmup_iter, state):
        return weights, state


@dataclass
class ReLoBRaLo:
    """ReLoBRaLo adaptive loss weights (Bischof & Kraus, 2021).

    Re-balances the residual terms from their loss history: each update softmax-compares the
    current per-residual losses against the initial (``loss_init``) and previous-step
    (``loss_prev``) references, blended by ``alpha`` (history vs. current) and ``rho``
    (look-back retention — the paper's Bernoulli look-back applied in expectation, so
    deterministic). ``T`` is the softmax temperature; weights are renormalized to sum to ``n``.
    """

    T: float = 1.0
    alpha: float = 0.999
    rho: float = 0.999
    update_every: int = 1

    def initial(self, n):
        return [1.0] * n

    def update(self, weights, losses, iteration, warmup_iter, state):
        """One bookkeeping step; ``state`` is ``(loss_init, loss_prev)``. Returns ``(weights, state)``."""
        n = len(weights)
        current = list(losses)
        loss_init, loss_prev = state if state is not None else (None, None)
        if loss_init is None:                    # first call: anchor the reference losses
            loss_init = list(current)
        elif iteration == warmup_iter:           # shock introduced: re-anchor and reset weights
            loss_init = list(current)
            weights = [1.0] * n
        elif iteration % self.update_every == 0:
            weights = self._blend(weights, current, loss_init, loss_prev, n)
        return weights, (loss_init, current)     # current is next call's loss_prev

    def _blend(self, weights, current, loss_init, loss_prev, n):
        eps = 1e-12
        r_step = torch.tensor([current[k] / (self.T * loss_prev[k] + eps) for k in range(n)])
        w_step = (torch.softmax(r_step, dim=0) * n).tolist()
        r_base = torch.tensor([current[k] / (self.T * loss_init[k] + eps) for k in range(n)])
        w_base = (torch.softmax(r_base, dim=0) * n).tolist()
        return [
            self.alpha * (self.rho * weights[k] + (1 - self.rho) * w_base[k])
            + (1 - self.alpha) * w_step[k]
            for k in range(n)
        ]
