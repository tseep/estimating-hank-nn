"""One-asset HANK model (monetary-policy shock) solved with neural networks.

`OneAssetHANK` maps (idiosyncratic state, wealth distribution, aggregate state, parameters)
to household labor `n`, inflation `Pi`, wage `W`, and the deterministic-steady-state
levels `Rn_dss`/`Y_dss`. Four networks: `dss_hh_network`/`dss_network` (deterministic
steady-state distribution, DSS phase), `hh_network`/`agg_network` (full stochastic model).

Notation (kept consistent with the SSJ reference in `main_hankmp_ssj.py`):

    Concept                         symbol
    ---------------------------------------------------------------
    gross / net inflation           Pi / pi              (Pi = 1 + pi)
    gross / net real rate           Rr / r               (Rr = Rn_lag/Pi = 1 + r)
    gross nominal policy rate       Rn = Rn_dss + phipi*(Pi-1) + mps   (Taylor rule)
    lagged gross nominal rate       Rn_lag                (agg state; return on bonds)
    gross nominal SS rate           par.Rn_ss             (= 1 + rstar of the SSJ side)
    Taylor inflation coefficient    phipi
    agg TFP (log) / level           z / exp(z)           (exp(z) = SSJ `Z`; dormant here)
    prod-weighted labor             NE = mean(exp(s)*n)  (SSJ `NE`, pinned to 1 at SS)
    unweighted labor                N  = mean(n)         (SSJ `N`)
    gov debt = SS assets            par.B                (SSJ `B`)
    individual assets               b                    (SSJ `a_grid`)
    idiosyncratic log productivity  s, level exp(s)      (SSJ `e_grid`)

The monetary block is a nominal additive Taylor rule with a lagged-rate state (no ELB);
the SSJ side uses the Fisher real-rate rule. With Rn_dss = 1 + rstar_ss and mps = rstar
deviation the two rules are algebraically identical (verified to machine precision, incl.
the realized real-rate path), so the NN `mps` shock maps 1:1 to the SSJ `rstar` shock.
"""

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo
from tqdm import trange

# Compiled networks are evaluated under both grad-enabled (training) and torch.no_grad()
# (simulation) contexts; each transition can trigger a dynamo recompile, so raise the cache.
torch._dynamo.config.cache_size_limit = 64

from estimating_hank_nn.structures import Parameters, State, Ranges, Shocks
from estimating_hank_nn.networks import NormalizeLayer
from estimating_hank_nn.helpers import seed_all, fisher_burmeister
from estimating_hank_nn.loss_weights import ConstantWeights


def _orig(network):
    """Underlying module of a (possibly) torch.compiled network (for clean state_dicts)."""
    return getattr(network, "_orig_mod", network)


@dataclass
class TrainingConfig:
    """Hyperparameters for ``OneAssetHANK.train_dss_model`` / ``train_model``."""

    # --- Loop structure ---
    iteration: int = 10_000
    internal: int = 25
    steps: int = 5
    batch: int = 50
    mc: int = 10
    par_draw_after: int = 10

    # --- Optimizer ---
    lr: float = 1e-4
    eta_min: float = 1e-7
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lr_schedule: str = "cosine"  # cosine (LinearLR warmup -> CosineAnnealing) | steplr (legacy StepLR, no warmup)

    # --- Loss weighting (the type of `weights` selects the scheme) ---
    weights: object = field(default_factory=ConstantWeights)  # ConstantWeights | ReLoBRaLo
    adjusted_residuals: bool = False
    detach_next: bool = True   # False backprops through the next-period expectation (exact forward-looking gradient, deeper graph)
    warmup_frac: float = 0.05

    # --- Infrastructure ---
    device: str = "cpu"
    print_after: int = 10
    seed: Optional[int] = None

    def __str__(self):
        lines = ["TrainingConfig:"]
        for f in self.__dataclass_fields__:
            lines.append(f"  {f}: {getattr(self, f)}")
        return "\n".join(lines)


class BaseNetwork(torch.nn.Module):
    """Grouped MLP: per-(group, variable) normalize + linear projection, summed, then a
    shared trunk. Input groups are e.g. ``state_ind``, ``state_dist``, ``state_agg``, ``par``."""

    def __init__(self, n_inputs, n_outputs, norm_state=None, norm_par=None,
                 hidden=128, layers=5, activation=torch.nn.CELU()):
        super().__init__()
        self.norm_state = norm_state or {}
        self.norm_par = norm_par

        self.normalizers = torch.nn.ModuleDict()
        self.projections = torch.nn.ModuleDict()

        if "par" in n_inputs:
            dim = n_inputs["par"]
            if self.norm_par is not None:
                self.normalizers["par"] = NormalizeLayer(self.norm_par.low_tensor(), self.norm_par.high_tensor())
            self.projections["par"] = torch.nn.Linear(dim, hidden)

        for group, group_dim in n_inputs.items():
            if group == "par":
                continue
            self.normalizers[group] = torch.nn.ModuleDict()
            self.projections[group] = torch.nn.ModuleDict()
            for var, dim in group_dim.items():
                low, high = self.norm_state.get(group, {}).get(var, (-1.0, 1.0))
                self.normalizers[group][var] = NormalizeLayer(torch.ones(dim) * low, torch.ones(dim) * high)
                self.projections[group][var] = torch.nn.Linear(dim, hidden)

        middle_layers = []
        for _ in range(layers - 1):
            middle_layers.extend([torch.nn.Linear(hidden, hidden), activation])
        self.middle = torch.nn.Sequential(*middle_layers)
        self.output = torch.nn.Linear(hidden, n_outputs)

    def forward(self, **inputs):
        projs = []
        for name, layer in self.projections.items():
            if name == "par":
                x = inputs.get("par")
                if x is not None:
                    x = self.normalizers["par"](x)
                    projs.append(layer(x))
            else:
                group_input = inputs.get(name, None)
                if group_input is not None:
                    for var, var_layer in layer.items():
                        x = group_input.get(var)
                        if x is not None:
                            x = self.normalizers[name][var](x)
                            projs.append(var_layer(x))
        x = sum(projs)
        x = self.middle(x)
        return self.output(x)


class OneAssetHANK(object):
    N_RESIDUALS = 6  # number of residual equations
    RESIDUAL_NAMES = {                           # positional, matching residuals / dss_residuals
        "full": ["res_fb", "res_nkpc", "res_bond", "res_product", "res_bond_next", "res_product_next"],
        "dss": ["res_fb", "res_y", "res_bond", "res_product", "res_bond_next", "res_product_next"],
    }

    def __init__(self, parameters, ranges, shocks_ind, shocks_agg, *,
                 N_agents=100, N_dist=100, hidden=128, layers=5,
                 single_dist_draw=False, clearing=False, known_dss=False, state_range_mult=1.0,
                 compile_networks=True):
        self.range = Ranges(parameters, ranges)
        self.shock_ind = Shocks(shocks_ind)
        self.shocks_agg = Shocks(shocks_agg)
        self.par = Parameters(parameters)

        self.par_draw = None
        self.ss = None
        self.state_ind = None
        self.state_dist = None
        self.state_agg = None

        # Dimensions
        self.N_agents = N_agents
        self.N_dist = N_dist
        self.N_par = len(self.par)

        # Methodology toggles
        self.single_dist_draw = single_dist_draw
        self.clearing = clearing
        self.known_dss = known_dss   # DSS + full use exact SS values (ss.Rn/ss.Y) not dss_policy, so only the hh nets train

        # Loss / training state
        self.dss_loss_dict = None
        self.loss_dict = None
        self.dss_training_conf = None
        self.training_conf = None
        self.batch = None
        self.state_aux = None        # particle-filter lagged Y (for Y_growth), set in prepare

        # mps input-normalization half-width = state_range_mult * sigma_mps. At the default 1.0 the
        # CRN impact shock (= -sigma_mps) maps to the full-scale input -1; a wider range compresses
        # the mps signal into the net and attenuates the learned shock response. Deliberately
        # independent of rho_mps (an ergodic-spread width grows with rho_hi and re-widens the range).
        sigma_mps = float(parameters["sigma_mps"])
        mps_hw = state_range_mult * sigma_mps

        # Network input dimensions and state-normalization ranges
        n_state_ind = {"b": 1, "s": 1}
        n_state_dist = {"b": self.N_dist, "s": self.N_dist}
        n_state_agg = {"z": 1, "mps": 1, "Rn_lag": 1}

        norm_state_ind = {"b": (0.0, 25.0), "s": (-1.0, 1.0)}
        norm_state_dist = {"b": (0.0, 25.0), "s": (-1.0, 1.0)}
        norm_state_agg = {"z": (-0.01, 0.01), "mps": (-mps_hw, mps_hw), "Rn_lag": (0.95, 1.05)}

        self.hh_network_conf = {
            "n_inputs": {"state_ind": n_state_ind, "state_dist": n_state_dist,
                         "state_agg": n_state_agg, "par": self.N_par},
            "norm_state": {"state_ind": norm_state_ind, "state_dist": norm_state_dist,
                           "state_agg": norm_state_agg},
            "norm_par": self.range,
            "n_outputs": 1,
            "hidden": hidden,
            "layers": layers,
        }
        self.agg_network_conf = {
            "n_inputs": {"state_dist": n_state_dist, "state_agg": n_state_agg, "par": self.N_par},
            "norm_state": {"state_dist": norm_state_dist, "state_agg": norm_state_agg},
            "norm_par": self.range,
            "n_outputs": 2,
            "hidden": hidden,
            "layers": layers,
        }
        self.dss_network_conf = {
            "n_inputs": {"par": self.N_par},
            "norm_par": self.range,
            "n_outputs": 2,
            "hidden": hidden,
            "layers": layers,
        }

        self.dss_hh_network = BaseNetwork(**self.hh_network_conf)
        self.hh_network = BaseNetwork(**self.hh_network_conf)
        self.agg_network = BaseNetwork(**self.agg_network_conf)
        self.dss_network = BaseNetwork(**self.dss_network_conf)
        if compile_networks:
            self.dss_hh_network = torch.compile(self.dss_hh_network, dynamic=True)
            self.hh_network = torch.compile(self.hh_network, dynamic=True)
            self.agg_network = torch.compile(self.agg_network, dynamic=True)
            self.dss_network = torch.compile(self.dss_network, dynamic=True)

    @property
    def networks(self):
        return {"dss_hh": self.dss_hh_network, "hh": self.hh_network,
                "agg": self.agg_network, "dss": self.dss_network}

    def to(self, device):
        self.par.to(device)
        for attr in ["par_draw", "ss", "state_ind", "state_dist", "state_agg"]:
            value = getattr(self, attr, None)
            if value is not None:
                value.to(device)
        for net in self.networks.values():
            net.to(device)

    # --- Serialization (state_dict checkpoints, like rankzlb) ---

    def save(self, path, name="model"):
        """Save the four sub-networks' state_dicts (compiled prefix stripped) + loss/config."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.to("cpu")
        checkpoint = {
            "networks": {k: _orig(net).state_dict() for k, net in self.networks.items()},
            "dss_loss_dict": self.dss_loss_dict,
            "loss_dict": self.loss_dict,
            # A loaded checkpoint holds plain dicts, so only convert the configurations of this run.
            "dss_training_conf": asdict(self.dss_training_conf) if is_dataclass(self.dss_training_conf) else self.dss_training_conf,
            "training_conf": asdict(self.training_conf) if is_dataclass(self.training_conf) else self.training_conf,
        }
        torch.save(checkpoint, f"{path}/{name}.pt")

    def load(self, path):
        """Load a state_dict checkpoint into the (freshly constructed) networks."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for k, net in self.networks.items():
            _orig(net).load_state_dict(checkpoint["networks"][k])
        self.dss_loss_dict = checkpoint.get("dss_loss_dict")
        self.loss_dict = checkpoint.get("loss_dict")
        self.dss_training_conf = checkpoint.get("dss_training_conf")
        self.training_conf = checkpoint.get("training_conf")

    def warm_start_full(self):
        """Initialize the full-model household network from the trained DSS one."""
        _orig(self.hh_network).load_state_dict(_orig(self.dss_hh_network).state_dict())

    # --- Steady state and state initialization ---

    def steady_state(self, par=None):
        """Analytic deterministic steady state (zero inflation, NE = Y = 1): a ``State`` of aggregate levels."""
        if par is None:
            par = self.par_draw

        B = par.B
        NE = torch.full_like(B, 1.0)
        Y = torch.full_like(B, 1.0)
        Pi = torch.full_like(B, 1.0)
        z = torch.log(Y / NE)                       # log TFP; 0 at SS
        Psi = (par.theta - 1.0) / par.theta
        W = Psi * torch.exp(z)
        C = Y - 0.5 * par.varphi * torch.log(Pi) ** 2 * Y
        Rn = torch.ones_like(B) * par.Rn_ss           # gross nominal SS rate
        T = B * Rn / Pi - B
        tau = T
        chi = par.chi

        ss = {"B": B, "NE": NE, "Y": Y, "Pi": Pi, "z": z, "Psi": Psi,
              "W": W, "C": C, "Rn": Rn, "T": T, "tau": tau, "chi": chi}
        return State(ss)

    def initialize_state(self, batch, par=None, device="cpu"):
        """Seed the ind/dist/agg state panels at the steady state (bonds = B, exogenous states at 0)."""
        if par is None:
            par = self.par_draw
        ss = self.steady_state(par=par)
        noise_b = 0.0

        b_ind = torch.ones((batch, self.N_agents, 1), device=device) * ss.B \
            + torch.randn((batch, self.N_agents, 1), device=device) * noise_b * ss.B
        s_ind = torch.zeros((batch, self.N_agents, 1), device=device)
        self.state_ind = State({"b": b_ind, "s": s_ind})

        b_dist = torch.ones((batch, self.N_dist, 1), device=device) * ss.B \
            + torch.randn((batch, self.N_dist, 1), device=device) * noise_b * ss.B
        s_dist = torch.zeros((batch, self.N_dist, 1), device=device)
        self.state_dist = State({"b": b_dist, "s": s_dist})

        z = torch.zeros((batch, 1, 1), device=device)
        mps = torch.zeros((batch, 1, 1), device=device)
        self.state_agg = State({"z": z, "mps": mps, "Rn_lag": ss.Rn})

    def draw_parameters(self, shape, device="cpu"):
        return self.range.sample(shape, device=device)

    def draw_shocks(self, batch, mc=None, antithetic=True, device="cpu"):
        """Sample shock innovations.

        When ``single_dist_draw`` is set, the wealth-distribution block collapses to a
        single MC draw (concentration of measure: cross-section residuals self-average
        over N_dist), while the per-agent Euler block and aggregate keep the full fanout.
        """
        if mc is None:
            shape_ind = (batch, self.N_agents, 1)
            shape_dist = (batch, self.N_dist, 1)
            shape_agg = (batch, 1, 1)
        else:
            shape_ind = (mc, batch, self.N_agents, 1)
            shape_dist = (1 if self.single_dist_draw else mc, batch, self.N_dist, 1)
            shape_agg = (mc, batch, 1, 1)

        # Antithetic halving is meaningless (and yields a zero-element tensor) on a length-1 dim.
        dist_antithetic = antithetic and shape_dist[0] > 1
        e_ind = self.shock_ind.sample(shape_ind, antithetic=antithetic, device=device)
        e_dist = self.shock_ind.sample(shape_dist, antithetic=dist_antithetic, device=device)
        e_agg = self.shocks_agg.sample(shape_agg, antithetic=antithetic, device=device)
        return e_ind, e_dist, e_agg

    # --- Income / clearing helpers (shared by residuals and step) ---

    def _ar1_next(self, s, shock_s, par, scale_ind):
        """AR(1) update of log productivity, renormalized so the cross-section mean of
        ``exp(s)`` is 1 (post-hoc normalization)."""
        s_next = par.rho_s * s + shock_s * par.sigma_s * scale_ind
        return torch.log(torch.exp(s_next) / torch.mean(torch.exp(s_next), dim=-2, keepdim=True))

    @staticmethod
    def _bond_tax(B, nominal_rate, Pi):
        """Government interest cost on outstanding bonds, net of principal."""
        return B * nominal_rate / Pi - B

    @staticmethod
    def _realized_real_rate(Rn_lag, Pi):
        """Gross return realized this period on bonds selected last period."""
        return Rn_lag / Pi

    def _apply_clearing(self, b_ind, b_dist, par):
        """Bounds-aware cross-section shift bringing each population's bond mean to ``par.B``.

        We must move the mean by ``delta = mean(b) - par.B`` without pushing anyone past
        ``[b_min, b_max]``. Each agent's *headroom* is its distance to the bound on the side we
        shift toward: if the mean is too high (``delta > 0``) we lower bonds, so headroom is
        ``b - b_min``; if too low, it is ``b_max - b``. Distributing the shift in proportion to
        headroom (i) keeps everyone inside the bounds, (ii) loads most of the adjustment onto
        agents far from the bound, whose policies are smoother — so it distorts the cross-section
        less than moving constrained agents — and (iii) since the weights average to 1, lands the
        mean exactly on ``par.B``.

        Simulator-only (no_grad); the training residual still penalizes non-clearing on the
        un-shifted ``b_dist``.
        """
        if not self.clearing:
            return b_ind, b_dist
        delta_ind = torch.mean(b_ind, dim=-2, keepdim=True) - par.B
        delta_dist = torch.mean(b_dist, dim=-2, keepdim=True) - par.B
        # Distance to the bound on the side we're shifting toward (clamped to be non-negative).
        headroom_ind = torch.where(delta_ind > 0, b_ind - par.b_min, par.b_max - b_ind).clamp(min=0.0)
        headroom_dist = torch.where(delta_dist > 0, b_dist - par.b_min, par.b_max - b_dist).clamp(min=0.0)
        mean_hr_ind = torch.mean(headroom_ind, dim=-2, keepdim=True).clamp(min=1e-12)
        mean_hr_dist = torch.mean(headroom_dist, dim=-2, keepdim=True).clamp(min=1e-12)
        # Subtract delta weighted by headroom/mean(headroom) — weights average to 1, so the mean → B.
        b_ind = b_ind - delta_ind * (headroom_ind / mean_hr_ind)
        b_dist = b_dist - delta_dist * (headroom_dist / mean_hr_dist)
        return b_ind, b_dist

    # --- Network input preparation ---

    def _prepare_par_input(self, par, mc=None):
        """Flatten parameters to a network-input row; broadcast over a leading MC axis when ``mc`` is set."""
        input_par = par.cat()
        if mc is not None:
            input_par_shape = list(input_par.shape)
            input_par_shape.insert(0, mc)
            input_par = input_par.unsqueeze(0).expand(input_par_shape)
        return input_par

    def _permute_dist(self, state, mc=None):
        """Move the wealth-distribution agent axis to the channel dimension for the network."""
        return state.permute(dims=(0, 1, 3, 2) if mc is not None else (0, 2, 1))

    # --- Policy networks (states/par passed explicitly; ``mc`` adds the MC-sample axis) ---

    def dss_hh_policy(self, state_ind, state_dist, state_agg, par, mc=None):
        """Household labor n in (0.01, 2.0) from the DSS hh network (distribution sorted by assets)."""
        state_dist = self._permute_dist(state_dist.sort("b", dim=-2), mc)
        par = self._prepare_par_input(par, mc)
        out = self.dss_hh_network(state_ind=state_ind, state_dist=state_dist, state_agg=state_agg, par=par)
        n_low, n_high = 0.01, 2.0
        return n_low + torch.sigmoid(out[..., 0:1]) * (n_high - n_low)

    def hh_policy(self, state_ind, state_dist, state_agg, par, mc=None):
        """Household labor n in (0.01, 2.0) from the full-model hh network (distribution sorted by assets)."""
        state_dist = self._permute_dist(state_dist.sort("b", dim=-2), mc)
        par = self._prepare_par_input(par, mc)
        out = self.hh_network(state_ind=state_ind, state_dist=state_dist, state_agg=state_agg, par=par)
        n_low, n_high = 0.01, 2.0
        return n_low + torch.sigmoid(out[..., 0:1]) * (n_high - n_low)

    def agg_policy(self, state_dist, state_agg, par, mc=None):
        """Aggregate policies from the agg network: inflation Pi in (0.9, 1.1) and wage W in (0.5, 1.5)."""
        state_dist = self._permute_dist(state_dist.sort("b", dim=-2), mc)
        par = self._prepare_par_input(par, mc)
        out = self.agg_network(state_dist=state_dist, state_agg=state_agg, par=par)
        Pi = 0.9 + torch.sigmoid(out[..., 0:1]) * (1.1 - 0.9)
        W = 0.5 + torch.sigmoid(out[..., 1:2]) * (1.5 - 0.5)
        return Pi, W

    def dss_policy(self, par, mc=None):
        """Deterministic steady-state levels from the dss network: Rn_dss, Y_dss in (0.8, 1.2)."""
        par = self._prepare_par_input(par, mc)
        out = self.dss_network(par=par)
        Rn_dss = 0.8 + torch.sigmoid(out[..., 0:1]) * (1.2 - 0.8)
        Y_dss = 0.8 + torch.sigmoid(out[..., 1:2]) * (1.2 - 0.8)
        return Rn_dss, Y_dss

    # --- Residuals (deterministic steady-state distribution) ---

    def dss_residuals(self, e, mc, detach_next=True, scale_ind=1, scale_agg=0, adjusted_residuals=False):
        par = self.par_draw
        ss = self.ss
        state_ind, state_dist, state_agg = self.state_ind, self.state_dist, self.state_agg
        e_ind, e_dist, e_agg = e

        # Current period: policies -> aggregates -> household budgets
        n_ind = self.dss_hh_policy(state_ind=state_ind, state_dist=state_dist, state_agg=state_agg, par=par)
        n_dist = self.dss_hh_policy(state_ind=state_dist, state_dist=state_dist, state_agg=state_agg, par=par)
        if self.known_dss:
            Rn_dss, Y_dss = ss.Rn, ss.Y
        else:
            Rn_dss, Y_dss = self.dss_policy(par=par)

        Rn_lag = Rn_dss          # gross nominal rate on bonds (DSS: notional = SS level)
        Rn = Rn_dss              # realized gross nominal rate (DSS: no inflation gap, no mps)
        Pi = ss.Pi
        W = ss.W
        NE = torch.mean(torch.exp(state_dist.s) * n_dist, dim=-2, keepdim=True)
        Y = torch.exp(ss.z) * NE
        T = self._bond_tax(par.B, Rn_lag, Pi)
        tau = T / torch.mean(torch.exp(state_dist.s), dim=-2, keepdim=True)

        c_ind = ((W * torch.exp(state_ind.s)) / (ss.chi * n_ind ** par.eta)) ** (1 / par.gamma)
        c_dist = ((W * torch.exp(state_dist.s)) / (ss.chi * n_dist ** par.eta)) ** (1 / par.gamma)
        div = Y - W * NE - 0.5 * par.varphi * torch.log(Pi) ** 2 * Y
        div_ind = div * torch.exp(state_ind.s)
        div_dist = div * torch.exp(state_dist.s)
        wealth_ind = W * torch.exp(state_ind.s) * n_ind + Rn_lag / Pi * state_ind.b + div_ind - tau * torch.exp(state_ind.s)
        wealth_dist = W * torch.exp(state_dist.s) * n_dist + Rn_lag / Pi * state_dist.b + div_dist - tau * torch.exp(state_dist.s)
        b_ind = wealth_ind - c_ind
        b_dist = wealth_dist - c_dist
        C = torch.mean(c_dist, dim=-2, keepdim=True)

        # Next period
        shape_ind = e_ind.s.shape
        shape_dist = e_dist.s.shape
        shape_agg = e_agg.z.shape

        s_ind_next = self._ar1_next(state_ind.s, e_ind.s, par, scale_ind)
        s_dist_next = self._ar1_next(state_dist.s, e_dist.s, par, scale_ind)
        z = (1 - par.rho_z) * ss.z + par.rho_z * state_agg.z + e_agg.z * par.sigma_z * scale_agg
        mps = par.rho_mps * state_agg.mps + e_agg.mps * par.sigma_mps * scale_agg

        state_ind_next = State({"b": b_ind.expand(shape_ind), "s": s_ind_next})
        state_dist_next = State({"b": b_dist.expand(shape_dist), "s": s_dist_next})
        state_agg_next = State({"z": z, "mps": mps, "Rn_lag": Rn.expand(shape_agg)})

        with torch.set_grad_enabled(not detach_next):
            n_ind_next = self.dss_hh_policy(mc=mc, state_ind=state_ind_next, state_dist=state_dist_next, state_agg=state_agg_next, par=par)
            n_dist_next = self.dss_hh_policy(mc=mc, state_ind=state_dist_next, state_dist=state_dist_next, state_agg=state_agg_next, par=par)

        NE_next = torch.mean(torch.exp(state_dist_next.s) * n_dist_next, dim=-2, keepdim=True)
        Pi_next = ss.Pi
        W_next = ss.W
        Y_next = torch.exp(ss.z) * NE_next
        T_next = self._bond_tax(par.B, state_agg_next.Rn_lag, Pi_next)
        tau_next = T_next / torch.mean(torch.exp(state_dist_next.s), dim=-2, keepdim=True)
        div_next = Y_next - W_next * NE_next - 0.5 * par.varphi * torch.log(Pi_next) ** 2 * Y_next
        div_dist_next = div_next * torch.exp(state_dist_next.s)
        wealth_dist_next = W_next * torch.exp(state_dist_next.s) * n_dist_next + Rn / Pi_next * state_dist_next.b + div_dist_next - tau_next * torch.exp(state_dist_next.s)
        c_ind_next = ((W_next * torch.exp(state_ind_next.s)) / (ss.chi * n_ind_next ** par.eta)) ** (1 / par.gamma)
        c_dist_next = ((W_next * torch.exp(state_dist_next.s)) / (ss.chi * n_dist_next ** par.eta)) ** (1 / par.gamma)
        b_dist_next = wealth_dist_next - c_dist_next
        C_next = torch.mean(c_dist_next, dim=-2, keepdim=True)

        q = par.beta * (c_ind_next / c_ind) ** (-par.gamma)
        lambda_comp = torch.mean(q * Rn / Pi_next, dim=0)

        # Equilibrium residuals (Euler/FB, output, bond clearing, goods clearing)
        res_fb = torch.where(
            b_ind < 0.8 * par.b_max,
            fisher_burmeister(b_ind - par.b_min, 1 - lambda_comp),
            fisher_burmeister(par.b_max - b_ind, lambda_comp - 1),  # corrected sign (upper bound)
        )
        res_y = Y - Y_dss
        res_bond = par.B - torch.mean(b_dist, dim=-2, keepdim=True)
        res_bond_next = par.B - torch.mean(b_dist_next, dim=-2, keepdim=True)
        res_product = Y - (C + 0.5 * par.varphi * torch.log(Pi) ** 2 * Y)
        res_product_next = Y_next - (C_next + 0.5 * par.varphi * torch.log(Pi_next) ** 2 * Y_next)

        if adjusted_residuals:
            return [res_fb, res_y / Y, res_bond / par.B, res_product / Y, res_bond_next / par.B, res_product_next / Y_next]
        return [res_fb, res_y, res_bond, res_product, res_bond_next, res_product_next]

    # --- Residuals (full stochastic model) ---

    def residuals(self, e, mc, detach_next=True, scale_ind=1, scale_agg=1, adjusted_residuals=False):
        par = self.par_draw
        ss = self.ss
        state_ind, state_dist, state_agg = self.state_ind, self.state_dist, self.state_agg
        e_ind, e_dist, e_agg = e

        # Current period: policies -> aggregates -> household budgets
        if self.known_dss:
            Rn_dss, Y_dss = ss.Rn, ss.Y
        else:
            Rn_dss, Y_dss = self.dss_policy(par=par)
        n_ind = self.hh_policy(state_ind=state_ind, state_dist=state_dist, state_agg=state_agg, par=par)
        n_dist = self.hh_policy(state_ind=state_dist, state_dist=state_dist, state_agg=state_agg, par=par)
        Pi, W = self.agg_policy(state_dist=state_dist, state_agg=state_agg, par=par)

        Rn_lag = state_agg.Rn_lag
        NE = torch.mean(torch.exp(state_dist.s) * n_dist, dim=-2, keepdim=True)
        Y = torch.exp(state_agg.z) * NE
        T = self._bond_tax(par.B, Rn_lag, Pi)
        tau = T / torch.mean(torch.exp(state_dist.s), dim=-2, keepdim=True)
        Psi = W / torch.exp(state_agg.z)
        Rn = Rn_dss + par.phipi * (Pi - 1) + state_agg.mps   # Taylor rule (ELB stripped)

        c_ind = ((W * torch.exp(state_ind.s)) / (ss.chi * n_ind ** par.eta)) ** (1 / par.gamma)
        c_dist = ((W * torch.exp(state_dist.s)) / (ss.chi * n_dist ** par.eta)) ** (1 / par.gamma)
        div = Y - W * NE - 0.5 * par.varphi * torch.log(Pi) ** 2 * Y
        div_ind = div * torch.exp(state_ind.s)
        div_dist = div * torch.exp(state_dist.s)
        wealth_ind = W * torch.exp(state_ind.s) * n_ind + Rn_lag / Pi * state_ind.b + div_ind - tau * torch.exp(state_ind.s)
        wealth_dist = W * torch.exp(state_dist.s) * n_dist + Rn_lag / Pi * state_dist.b + div_dist - tau * torch.exp(state_dist.s)
        b_ind = wealth_ind - c_ind
        b_dist = wealth_dist - c_dist
        C = torch.mean(c_dist, dim=-2, keepdim=True)

        # Next period
        shape_ind = e_ind.s.shape
        shape_dist = e_dist.s.shape
        shape_agg = e_agg.z.shape

        s_ind_next = self._ar1_next(state_ind.s, e_ind.s, par, scale_ind)
        s_dist_next = self._ar1_next(state_dist.s, e_dist.s, par, scale_ind)
        z = (1 - par.rho_z) * ss.z + par.rho_z * state_agg.z + e_agg.z * par.sigma_z * scale_agg
        mps = par.rho_mps * state_agg.mps + e_agg.mps * par.sigma_mps * scale_agg

        state_ind_next = State({"b": b_ind.expand(shape_ind), "s": s_ind_next})
        state_dist_next = State({"b": b_dist.expand(shape_dist), "s": s_dist_next})
        state_agg_next = State({"z": z, "mps": mps, "Rn_lag": Rn.expand(shape_agg)})

        with torch.set_grad_enabled(not detach_next):
            n_ind_next = self.hh_policy(mc=mc, state_ind=state_ind_next, state_dist=state_dist_next, state_agg=state_agg_next, par=par)
            n_dist_next = self.hh_policy(mc=mc, state_ind=state_dist_next, state_dist=state_dist_next, state_agg=state_agg_next, par=par)
            Pi_next, W_next = self.agg_policy(mc=mc, state_dist=state_dist_next, state_agg=state_agg_next, par=par)

        NE_next = torch.mean(torch.exp(state_dist_next.s) * n_dist_next, dim=-2, keepdim=True)
        Y_next = torch.exp(state_agg_next.z) * NE_next
        T_next = self._bond_tax(par.B, state_agg_next.Rn_lag, Pi_next)
        tau_next = T_next / torch.mean(torch.exp(state_dist_next.s), dim=-2, keepdim=True)
        div_next = Y_next - W_next * NE_next - 0.5 * par.varphi * torch.log(Pi_next) ** 2 * Y_next
        div_dist_next = div_next * torch.exp(state_dist_next.s)
        wealth_dist_next = W_next * torch.exp(state_dist_next.s) * n_dist_next + Rn / Pi_next * state_dist_next.b + div_dist_next - tau_next * torch.exp(state_dist_next.s)
        c_ind_next = ((W_next * torch.exp(state_ind_next.s)) / (ss.chi * n_ind_next ** par.eta)) ** (1 / par.gamma)
        c_dist_next = ((W_next * torch.exp(state_dist_next.s)) / (ss.chi * n_dist_next ** par.eta)) ** (1 / par.gamma)
        b_dist_next = wealth_dist_next - c_dist_next
        C_next = torch.mean(c_dist_next, dim=-2, keepdim=True)

        q = par.beta * (c_ind_next / c_ind) ** (-par.gamma)
        Efp = torch.mean((Pi_next / Rn) * torch.log(Pi_next) * (Y_next / Y), dim=0)
        lambda_comp = torch.mean(q * Rn / Pi_next, dim=0)

        # Equilibrium residuals (Euler/FB, NKPC, bond clearing, goods clearing)
        res_fb = torch.where(
            b_ind < 0.8 * par.b_max,
            fisher_burmeister(b_ind - par.b_min, 1 - lambda_comp),
            fisher_burmeister(par.b_max - b_ind, lambda_comp - 1),  # corrected sign (upper bound)
        )
        res_nkpc = par.varphi * torch.log(Pi) - ((1 - par.theta) + par.theta * Psi + par.varphi * Efp)
        res_bond = par.B - torch.mean(b_dist, dim=-2, keepdim=True)
        res_bond_next = par.B - torch.mean(b_dist_next, dim=-2, keepdim=True)
        res_product = Y - (C + 0.5 * par.varphi * torch.log(Pi) ** 2 * Y)
        res_product_next = Y_next - (C_next + 0.5 * par.varphi * torch.log(Pi_next) ** 2 * Y_next)

        if adjusted_residuals:
            return [res_fb, res_nkpc / (1 - par.theta), res_bond / par.B, res_product / Y, res_bond_next / par.B, res_product_next / Y_next]
        return [res_fb, res_nkpc, res_bond, res_product, res_bond_next, res_product_next]

    # --- One-step simulation ---

    @torch.no_grad()
    def dss_step(self, e, scale_ind=1, scale_agg=0, scale_exploration=0.0, advance_first=True):
        """DSS counterpart of ``step``: same period economics with aggregate shocks off, fixed prices
        ``Pi=ss.Pi`` / ``W=ss.W`` and the deterministic rate ``Rn_dss``. ``advance_first`` matches ``step``
        (True = advance-then-observe; False = residual convention, for training). Mutates ``self.state_*``
        and returns the observables dict."""
        par = self.par_draw
        ss = self.ss
        e_ind, e_dist, e_agg = e

        s_ind = self._ar1_next(self.state_ind.s, e_ind.s, par, scale_ind)
        s_dist = self._ar1_next(self.state_dist.s, e_dist.s, par, scale_ind)
        z = (1 - par.rho_z) * ss.z + par.rho_z * self.state_agg.z + e_agg.z * par.sigma_z * scale_agg
        mps = par.rho_mps * self.state_agg.mps + e_agg.mps * par.sigma_mps * scale_agg
        if advance_first:                  # aow: observe the freshly-advanced exogenous state
            state_ind = State({"b": self.state_ind.b, "s": s_ind})
            state_dist = State({"b": self.state_dist.b, "s": s_dist})
            state_agg = State({"z": z, "mps": mps, "Rn_lag": self.state_agg.Rn_lag})
        else:                              # residual convention (training): observe the raw stored state
            state_ind, state_dist, state_agg = self.state_ind, self.state_dist, self.state_agg

        n_ind = self.dss_hh_policy(state_ind=state_ind, state_dist=state_dist, state_agg=state_agg, par=par)
        n_dist = self.dss_hh_policy(state_ind=state_dist, state_dist=state_dist, state_agg=state_agg, par=par)
        if self.known_dss:
            Rn_dss, Y_dss = ss.Rn, ss.Y
        else:
            Rn_dss, Y_dss = self.dss_policy(par=par)
        n_ind = n_ind + scale_exploration * torch.std(n_dist, dim=-2, keepdim=True) * torch.randn_like(n_ind)

        Rn_lag = Rn_dss
        Pi = ss.Pi
        W = ss.W
        NE = torch.mean(torch.exp(state_dist.s) * n_dist, dim=-2, keepdim=True)
        Y = torch.exp(state_agg.z) * NE
        T = self._bond_tax(par.B, Rn_lag, Pi)
        tau = T / torch.mean(torch.exp(state_dist.s), dim=-2, keepdim=True)
        Rn = Rn_dss

        c_ind = ((W * torch.exp(state_ind.s)) / (ss.chi * n_ind ** par.eta)) ** (1 / par.gamma)
        c_dist = ((W * torch.exp(state_dist.s)) / (ss.chi * n_dist ** par.eta)) ** (1 / par.gamma)
        div = Y - W * NE - 0.5 * par.varphi * torch.log(Pi) ** 2 * Y
        div_ind = div * torch.exp(state_ind.s)
        div_dist = div * torch.exp(state_dist.s)
        wealth_ind = W * torch.exp(state_ind.s) * n_ind + Rn_lag / Pi * state_ind.b + div_ind - tau * torch.exp(state_ind.s)
        wealth_dist = W * torch.exp(state_dist.s) * n_dist + Rn_lag / Pi * state_dist.b + div_dist - tau * torch.exp(state_dist.s)
        b_ind = wealth_ind - c_ind
        b_dist = wealth_dist - c_dist
        b_ind, b_dist = self._apply_clearing(b_ind, b_dist, par)
        C = torch.mean(c_dist, dim=-2, keepdim=True)

        self.state_ind = State({"b": b_ind, "s": s_ind})
        self.state_dist = State({"b": b_dist, "s": s_dist})
        self.state_agg = State({"z": z, "mps": mps, "Rn_lag": Rn_dss})

        return {
            "Y": Y, "C": C, "Rn": Rn, "Rr": self._realized_real_rate(Rn_lag, Pi), "Pi": Pi, "W": W,
            "NE": NE, "N": torch.mean(n_dist, dim=-2, keepdim=True),
            "T": T, "tau": tau, "z": state_agg.z, "mps": state_agg.mps, "Rn_lag": Rn_lag,
            "res_bond": par.B - torch.mean(b_dist, dim=-2, keepdim=True),
            "res_product": Y - (C + 0.5 * par.varphi * torch.log(Pi) ** 2 * Y),
        }

    @torch.no_grad()
    def step(self, e, scale_ind=1, scale_agg=1, scale_exploration=0.0, advance_first=True):
        """Advance one period and realize it. ``advance_first=True`` (default; IRF/PF/analysis) is
        advance-then-observe: roll the exogenous states first, evaluate the freshly-advanced state.
        ``advance_first=False`` (training) is the residual convention: evaluate the raw current state and
        advance only the saved next state, keeping the simulated panel consistent with ``residuals``.
        Either way assets ``b`` are predetermined, the saved exogenous states are advanced, and ``Rn``
        becomes the next ``Rn_lag``. Mutates ``self.state_*`` and returns the observables dict."""
        par = self.par_draw
        ss = self.ss
        e_ind, e_dist, e_agg = e

        # Advance the exogenous/Markov states (b and Rn_lag carry); these are stored as the next state
        # below regardless of convention.
        s_ind = self._ar1_next(self.state_ind.s, e_ind.s, par, scale_ind)
        s_dist = self._ar1_next(self.state_dist.s, e_dist.s, par, scale_ind)
        z = (1 - par.rho_z) * ss.z + par.rho_z * self.state_agg.z + e_agg.z * par.sigma_z * scale_agg
        mps = par.rho_mps * self.state_agg.mps + e_agg.mps * par.sigma_mps * scale_agg
        Rn_lag = self.state_agg.Rn_lag
        # advance_first=True (aow; PF/sim): observe the freshly-advanced exogenous state. False (training):
        # observe the raw stored state, so the panel stays consistent with residuals(); the advance above
        # then applies only to what is stored below.
        if advance_first:
            state_ind = State({"b": self.state_ind.b, "s": s_ind})
            state_dist = State({"b": self.state_dist.b, "s": s_dist})
            state_agg = State({"z": z, "mps": mps, "Rn_lag": Rn_lag})
        else:
            state_ind, state_dist, state_agg = self.state_ind, self.state_dist, self.state_agg

        n_ind = self.hh_policy(state_ind=state_ind, state_dist=state_dist, state_agg=state_agg, par=par)
        n_dist = self.hh_policy(state_ind=state_dist, state_dist=state_dist, state_agg=state_agg, par=par)
        if self.known_dss:
            Rn_dss, Y_dss = ss.Rn, ss.Y
        else:
            Rn_dss, Y_dss = self.dss_policy(par=par)
        Pi, W = self.agg_policy(state_dist=state_dist, state_agg=state_agg, par=par)
        n_ind = n_ind + scale_exploration * torch.std(n_dist, dim=-2, keepdim=True) * torch.randn_like(n_ind)

        NE = torch.mean(torch.exp(state_dist.s) * n_dist, dim=-2, keepdim=True)
        Y = torch.exp(state_agg.z) * NE
        T = self._bond_tax(par.B, state_agg.Rn_lag, Pi)
        tau = T / torch.mean(torch.exp(state_dist.s), dim=-2, keepdim=True)
        Rn = Rn_dss + par.phipi * (Pi - 1) + state_agg.mps

        c_ind = ((W * torch.exp(state_ind.s)) / (ss.chi * n_ind ** par.eta)) ** (1 / par.gamma)
        c_dist = ((W * torch.exp(state_dist.s)) / (ss.chi * n_dist ** par.eta)) ** (1 / par.gamma)
        div = Y - W * NE - 0.5 * par.varphi * torch.log(Pi) ** 2 * Y
        div_ind = div * torch.exp(state_ind.s)
        div_dist = div * torch.exp(state_dist.s)
        wealth_ind = W * torch.exp(state_ind.s) * n_ind + Rn_lag / Pi * state_ind.b + div_ind - tau * torch.exp(state_ind.s)
        wealth_dist = W * torch.exp(state_dist.s) * n_dist + Rn_lag / Pi * state_dist.b + div_dist - tau * torch.exp(state_dist.s)
        b_ind = wealth_ind - c_ind
        b_dist = wealth_dist - c_dist
        b_ind, b_dist = self._apply_clearing(b_ind, b_dist, par)
        C = torch.mean(c_dist, dim=-2, keepdim=True)

        self.state_ind = State({"b": b_ind, "s": s_ind})
        self.state_dist = State({"b": b_dist, "s": s_dist})
        self.state_agg = State({"z": z, "mps": mps, "Rn_lag": Rn})

        return {
            "Y": Y, "C": C, "Rn": Rn, "Rr": self._realized_real_rate(Rn_lag, Pi), "Pi": Pi, "W": W,
            "NE": NE, "N": torch.mean(n_dist, dim=-2, keepdim=True),
            "T": T, "tau": tau, "z": state_agg.z, "mps": state_agg.mps, "Rn_lag": Rn_lag,
            "res_bond": par.B - torch.mean(b_dist, dim=-2, keepdim=True),
            "res_product": Y - (C + 0.5 * par.varphi * torch.log(Pi) ** 2 * Y),
        }

    def dss_steps(self, batch, device, steps, scale_ind=1, scale_agg=0, scale_exploration=0.0, advance_first=True):
        """Advance the panel ``steps`` periods of DSS dynamics (fresh shocks each step)."""
        for _ in range(steps):
            e = self.draw_shocks(batch, mc=None, antithetic=False, device=device)
            self.dss_step(e, scale_ind=scale_ind, scale_agg=scale_agg, scale_exploration=scale_exploration, advance_first=advance_first)

    def steps(self, batch, device, steps, scale_ind=1, scale_agg=1, scale_exploration=0.0, advance_first=True):
        """Advance the panel ``steps`` periods of full-model dynamics (fresh shocks each step)."""
        for _ in range(steps):
            e = self.draw_shocks(batch, mc=None, antithetic=False, device=device)
            self.step(e, scale_ind=scale_ind, scale_agg=scale_agg, scale_exploration=scale_exploration, advance_first=advance_first)

    # --- Particle-filter contract (estimation; mirrors rankzlb, see particle_filter.py) ---

    PF_OBS = ("Y_growth", "Pi", "Rn")  # macro aggregates observed by the filter

    def initialize_state_aux(self, batch, device="cpu"):
        """Lagged aggregate output for the Y_growth observable, at the steady state."""
        return State({"Y": torch.full((batch, 1, 1), float(self.ss.Y.reshape(-1)[0]), device=device)})

    def prepare(self, par_draw, batch, device="cpu"):
        """PF setup: reshape par_draw to (batch, 1, 1), solve the steady state, draw a fresh state and
        the lagged-Y aux. Arguments are passed explicitly (the PF does not rely on method defaults)."""
        self.to(device)
        self.par_draw = Parameters({k: v.reshape(batch, 1, 1) for k, v in par_draw.items()})
        self.ss = self.steady_state(par=self.par_draw)
        self.initialize_state(batch=batch, par=self.par_draw, device=device)
        self.state_aux = self.initialize_state_aux(batch=batch, device=device)

    @torch.no_grad()
    def forecast(self, shocks=None):
        """One realize-and-measure step for the filter (advance-then-observe): ``step`` advances and
        evaluates the freshly-propagated state, then this transforms its raw output into the filter's
        observables (output growth %, inflation & rate in annual %) and rolls the Y lag. ``shocks``
        (name -> tensor) supplies CRN for the aggregate innovations; idiosyncratic draws are fresh."""
        e = self.draw_shocks(self.state_agg.z.shape[0], mc=None, antithetic=False,
                             device=self.state_agg.z.device)
        if shocks is not None:
            _, _, e_agg = e
            for k, v in shocks.items():
                e_agg[k] = v.reshape(e_agg[k].shape)
        raw = self.step(e, scale_ind=1.0, scale_agg=1.0)
        Y_growth = 100.0 * (raw["Y"] / self.state_aux.Y - 1.0)
        self.state_aux = State({"Y": raw["Y"]})
        return {"Y_growth": Y_growth.squeeze(-1),
                "Pi": (400.0 * (raw["Pi"] - 1.0)).squeeze(-1),
                "Rn": (400.0 * (raw["Rn"] - 1.0)).squeeze(-1)}

    def resample(self, idx):
        """Resample particles along the batch (particle) dim across all state containers."""
        self.state_ind = self.state_ind.reindex(idx)
        self.state_dist = self.state_dist.reindex(idx)
        self.state_agg = self.state_agg.reindex(idx)
        if self.state_aux is not None:
            self.state_aux = self.state_aux.reindex(idx)

    # --- Loss and training ---

    def loss(self, residuals, criterion, names, weights=None):
        """Weighted sum of per-residual MSEs (the scalar to backprop) plus the unweighted per-residual components for logging."""
        if weights is None:
            weights = [1.0] * len(residuals)
        loss = [weights[i] * criterion(r, target=torch.zeros_like(r)) for i, r in enumerate(residuals)]
        with torch.no_grad():
            loss_components = {name: torch.mean(r ** 2) for name, r in zip(names, residuals)}
        return sum(loss), loss_components

    def initialize(self, batch, device):
        """Set up a training batch: fix ``par_draw`` at the calibration, draw a fresh state panel and steady state, move to device."""
        self.batch = batch
        self.par_draw = self.par.view(1, 1, 1).expand([batch, 1, 1]).to(device)
        self.initialize_state(batch=batch, device=device)
        self.ss = self.steady_state()
        self.to(device)

    @staticmethod
    def adjust_slowly(start, target, iteration, ramp_range=range(0, 1000, 1)):
        """Linear ramp from ``start`` to ``target`` across ``ramp_range`` (the shock-introduction warmup), clamped outside it."""
        slope = (target - start) / len(ramp_range)
        if ramp_range.start <= iteration <= ramp_range.stop:
            return start + slope * (iteration - ramp_range.start)
        if iteration < ramp_range.start:
            return start
        return target

    def _train(self, phase, config):
        """Shared training loop. ``phase`` in {"dss", "full"}."""
        seed_all(config.seed)
        device = config.device

        if phase == "dss":
            trained = [self.dss_hh_network, self.dss_network]
            for net in [self.agg_network, self.hh_network]:
                net.eval()
            residual_fn, step_fn = self.dss_residuals, self.dss_steps
        else:
            trained = [self.hh_network, self.agg_network]
            self.dss_network.eval()
            residual_fn, step_fn = self.residuals, self.steps
        for net in trained:
            net.train()
            net.to(device)
        self.initialize(config.batch, device)

        # Loss weights — the scheme (ConstantWeights | ReLoBRaLo) drives init and per-step update.
        scheme = config.weights
        weights = scheme.initial(self.N_RESIDUALS)
        weight_state = None
        warmup_iter = max(int(config.iteration * config.warmup_frac), 50)

        optimizer = torch.optim.AdamW(
            [{"params": net.parameters()} for net in trained], lr=config.lr, weight_decay=config.weight_decay
        )
        if config.lr_schedule == "steplr":
            # Legacy recipe: plain StepLR (gamma 0.1 every iteration//5), no LR warmup ramp.
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(config.iteration // 5, 1), gamma=0.1)
        else:
            # LR warmup over the shock-introduction window: full LR during the ramps caused loss crises.
            lr_ramp_end = min(2 * warmup_iter, max(config.iteration // 2, 1))
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=lr_ramp_end),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.iteration - lr_ramp_end,
                                                           eta_min=config.eta_min),
            ], milestones=[lr_ramp_end])
        criterion = torch.nn.MSELoss(reduction="mean")

        names = self.RESIDUAL_NAMES[phase]
        loss_dict = {"iteration": [], "total": []}
        loss_dict.update({name: [] for name in names})
        loss_dict.update({f"w_{name}": [] for name in names})

        pbar = trange(config.iteration)
        running_loss = 0.0
        loss_components = None
        for i in pbar:
            scale_ind = self.adjust_slowly(0.1, 1.0, i, ramp_range=range(0, warmup_iter, 1))
            if phase == "full":
                scale_agg = self.adjust_slowly(0.1, 1.0, i, ramp_range=range(warmup_iter, 2 * warmup_iter, 1))
            else:
                scale_agg = 0.0

            for o in range(config.internal):
                optimizer.zero_grad()
                e = self.draw_shocks(batch=config.batch, mc=config.mc, antithetic=True, device=device)
                residuals = residual_fn(e=e, mc=config.mc, scale_ind=scale_ind, scale_agg=scale_agg,
                                        adjusted_residuals=config.adjusted_residuals, detach_next=config.detach_next)
                loss, loss_components = self.loss(residuals=residuals, criterion=criterion, weights=weights, names=names)
                loss.backward()
                for net in trained:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip, error_if_nonfinite=True)
                optimizer.step()

                if o == 0:
                    loss_dict["iteration"].append(i)
                    loss_dict["total"].append(loss.item())
                    for key, value in loss_components.items():
                        loss_dict[key].append(value.item())
                    for k in range(self.N_RESIDUALS):
                        loss_dict[f"w_{names[k]}"].append(weights[k])
                    running_loss += loss.item()

            current = [loss_components[name].item() for name in names]
            weights, weight_state = scheme.update(weights, current, i, warmup_iter, weight_state)

            if i % config.print_after == 0:
                pbar.set_postfix({"lr": scheduler.get_last_lr()[0], "loss": running_loss / config.print_after,
                                  "w": [round(w, 2) for w in weights]})
                running_loss = 0.0

            scheduler.step()

            if i % config.par_draw_after == 0:
                self.par_draw = self.draw_parameters((config.batch, 1, 1), device=device)
                self.ss = self.steady_state()
                step_fn(batch=config.batch, device=device, steps=5, scale_ind=scale_ind, scale_agg=scale_agg, advance_first=False)

            if phase == "full" and (torch.any(self.state_ind.b < self.par_draw.b_min - 50)
                                    or torch.any(self.state_ind.b > self.par_draw.b_max + 50)):
                self.initialize_state(batch=config.batch, device=device)

            # Training advances the panel in the residual convention so the simulated states match
            # what residuals() evaluates (IRF/PF keep the default advance-then-observe).
            step_fn(batch=config.batch, device=device, steps=config.steps,
                    scale_ind=scale_ind, scale_agg=scale_agg, advance_first=False)

        for net in trained:
            net.eval()
        return loss_dict

    def train_dss_model(self, config: TrainingConfig):
        """Train the deterministic-steady-state networks (``dss_hh`` + ``dss``)."""
        self.dss_training_conf = config
        self.dss_loss_dict = self._train("dss", config)

    def train_model(self, config: TrainingConfig):
        """Train the full-model networks (``hh`` + ``agg``); ``dss`` stays frozen."""
        self.training_conf = config
        self.loss_dict = self._train("full", config)
