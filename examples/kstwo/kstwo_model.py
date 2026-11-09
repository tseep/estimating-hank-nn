"""Krusell-Smith model with two assets (capital and a bond), solved with neural networks.

`KrusellSmithTwoAsset` maps (idiosyncratic state, asset distribution, aggregate state, parameters)
to household consumption and bond holdings, and to the bond price `Q`. The parameters map to the
deterministic-steady-state productivity level `A_dss` and bond price `Q_dss`. Four networks:
`dss_hh_network`/`dss_network` (deterministic steady state, DSS phase) and
`hh_network`/`agg_network` (full stochastic model). Capital follows from the budget constraint.
"""

import math
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
    """Hyperparameters for ``KrusellSmithTwoAsset.train_dss_model`` / ``train_model``."""

    # --- Loop structure ---
    iteration: int = 10_000
    internal: int = 20
    steps: int = 5
    batch: int = 100
    mc: int = 50
    par_draw_after: int = 1000
    redraw_steps: int = 5      # simulation steps right after a parameter redraw
    steps_start: int = 1000    # iterations with one period per iteration: the shock ramp and the learning-rate warmup

    # --- Optimizer ---
    lr: float = 1e-4
    eta_min: float = 1e-8      # cosine LR floor
    warmup_frac: float = 0.05  # cosine: the LR warmup runs over twice this share of the iterations
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # --- Loss ---
    weights: object = field(default_factory=ConstantWeights)
    detach_next: bool = True   # False backprops through the next-period expectation
    log_loss: bool = False     # minimize log(loss) instead of the loss

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
    """Grouped MLP of the original guide: per-group normalize + linear projection over the
    concatenated inputs of the group, summed, then a shared trunk. Input groups are ``state_ind``,
    ``state_dist``, ``state_agg`` and ``par``. One flat band per group: identity on the household
    states, (-0.1, 0.1) on the aggregate block."""

    STATE_RANGES = {"state_ind": (-1.0, 1.0), "state_dist": (-1.0, 1.0), "state_agg": (-0.1, 0.1)}

    def __init__(self, n_inputs, n_outputs, ranges, hidden=128, layers=5, activation=torch.nn.CELU()):
        super().__init__()
        self.normalizers = torch.nn.ModuleDict()
        self.projections = torch.nn.ModuleDict()

        if n_inputs.get("par", 0) > 0:
            self.normalizers["par"] = NormalizeLayer(ranges.low_tensor(), ranges.high_tensor())
        for name, dim in n_inputs.items():
            if name != "par" and dim > 0:
                low, high = self.STATE_RANGES[name]
                self.normalizers[name] = NormalizeLayer(torch.ones(dim) * low, torch.ones(dim) * high)
        for name, dim in n_inputs.items():
            if dim > 0:
                self.projections[name] = torch.nn.Linear(dim, hidden)

        middle_layers = []
        for _ in range(layers - 1):
            middle_layers.extend([torch.nn.Linear(hidden, hidden), activation])
        self.middle = torch.nn.Sequential(*middle_layers)
        self.output = torch.nn.Linear(hidden, n_outputs)

    def forward(self, **inputs):
        projs = []
        for name, layer in self.projections.items():
            if inputs.get(name) is not None:
                projs.append(layer(self.normalizers[name](inputs[name])))
        x = sum(projs)
        x = self.middle(x)
        return self.output(x)


class KrusellSmithTwoAsset(object):
    N_RESIDUALS = 5  # number of residual equations
    RESIDUAL_NAMES = {                           # positional, matching residuals / dss_residuals
        "full": ["res_fb_k", "res_fb_b", "res_k", "res_k_next", "res_b"],
        "dss": ["res_fb_k", "res_fb_b", "res_y", "res_k", "res_b"],
    }

    def __init__(self, parameters, ranges, shocks_ind, shocks_agg, *,
                 N_agents=100, N_dist=100, hidden=128, layers=5, compile_networks=True):
        self.range = Ranges(parameters, ranges)
        self.shock_ind = Shocks(shocks_ind)
        self.shocks_agg = Shocks(shocks_agg)
        self.par = Parameters(parameters)

        self.par_draw = None
        self.ss = None
        self.state_ind = None
        self.state_dist = None
        self.state_agg = None
        self.sim = None              # aggregates of the last simulated period, set in step

        # Dimensions
        self.N_agents = N_agents
        self.N_dist = N_dist
        self.N_par = len(self.par)
        self.N_states_ind = 3        # capital k, bonds b, log productivity s
        self.N_states_agg = 2        # productivity deviation A_scale, aggregate capital Kl

        self.s_min = 0.25            # lower bound of the productivity level exp(s)

        # Loss / training state
        self.dss_loss_dict = None
        self.loss_dict = None
        self.dss_training_conf = None
        self.training_conf = None
        self.batch = None

        self.hh_network_conf = {
            "n_inputs": {"state_ind": self.N_states_ind, "state_dist": self.N_dist * self.N_states_ind,
                         "state_agg": self.N_states_agg, "par": self.N_par},
            "n_outputs": 2,          # consumption, bonds
            "ranges": self.range, "hidden": hidden, "layers": layers,
        }
        self.agg_network_conf = {
            "n_inputs": {"state_dist": self.N_dist * self.N_states_ind, "state_agg": self.N_states_agg, "par": self.N_par},
            "n_outputs": 1,          # bond price
            "ranges": self.range, "hidden": hidden, "layers": layers,
        }
        self.dss_network_conf = {
            "n_inputs": {"par": self.N_par},
            "n_outputs": 2,          # DSS productivity level, DSS bond price
            "ranges": self.range, "hidden": hidden, "layers": layers,
        }

        self.dss_hh_network = BaseNetwork(**self.hh_network_conf)
        self.hh_network = BaseNetwork(**self.hh_network_conf)
        self.dss_network = BaseNetwork(**self.dss_network_conf)
        self.agg_network = BaseNetwork(**self.agg_network_conf)
        if compile_networks:
            self.dss_hh_network = torch.compile(self.dss_hh_network, dynamic=True)
            self.hh_network = torch.compile(self.hh_network, dynamic=True)
            self.dss_network = torch.compile(self.dss_network, dynamic=True)
            self.agg_network = torch.compile(self.agg_network, dynamic=True)

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

    # --- Serialization (state_dict checkpoints) ---

    def save(self, path, name="model"):
        """Save the sub-networks' state_dicts (compiled prefix stripped) + loss/config."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.to("cpu")
        checkpoint = {
            "networks": {k: _orig(net).state_dict() for k, net in self.networks.items()},
            "dss_loss_dict": self.dss_loss_dict,
            "loss_dict": self.loss_dict,
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
        """Representative-agent steady state with output normalized to one. It scales the network
        outputs; the steady state with idiosyncratic risk is learned in the DSS phase."""
        if par is None:
            par = self.par_draw

        r = 1 / par.beta - 1
        N = torch.ones_like(r)
        Y = torch.ones_like(r)
        K = par.alpha * Y / (r + par.delta)
        C = Y - par.delta * K
        w = (1 - par.alpha) * Y / N
        A = Y / (K ** par.alpha * N ** (1 - par.alpha))

        return State({"r": r, "N": N, "Y": Y, "K": K, "C": C, "w": w, "A": A})

    def initialize_state(self, batch, par=None, device="cpu"):
        """All households start at the steady-state capital stock, the bond supply and mean
        productivity."""
        if par is None:
            par = self.par_draw
        ss = self.steady_state(par=par)
        k_start = ss.K
        noise_b = 0.0

        k_ind = torch.ones((batch, self.N_agents, 1), device=device) * k_start + torch.randn((batch, self.N_agents, 1), device=device) * noise_b * ss.K
        b_ind = torch.ones((batch, self.N_agents, 1), device=device) * par.B + torch.randn((batch, self.N_agents, 1), device=device) * noise_b * par.B
        s_ind = torch.zeros((batch, self.N_agents, 1), device=device)
        self.state_ind = State({"k": k_ind, "b": b_ind, "s": s_ind})

        k_dist = torch.ones((batch, self.N_dist, 1), device=device) * k_start + torch.randn((batch, self.N_dist, 1), device=device) * noise_b * ss.K
        b_dist = torch.ones((batch, self.N_dist, 1), device=device) * par.B + torch.randn((batch, self.N_dist, 1), device=device) * noise_b * par.B
        s_dist = torch.zeros((batch, self.N_dist, 1), device=device)
        self.state_dist = State({"k": k_dist, "b": b_dist, "s": s_dist})

        A_scale = torch.ones((batch, 1, 1), device=device) * ss.A - ss.A
        Kl = torch.ones((batch, 1, 1), device=device) * k_start
        self.state_agg = State({"A_scale": A_scale, "Kl": Kl})

        self.ss = ss

    def draw_parameters(self, shape, device="cpu"):
        return self.range.sample(shape, device=device)

    def draw_shocks(self, batch, mc=None, antithetic=True, device="cpu"):
        if mc is None:
            shape_ind = (batch, self.N_agents, 1)
            shape_dist = (batch, self.N_dist, 1)
            shape_agg = (batch, 1, 1)
        else:
            shape_ind = (mc, batch, self.N_agents, 1)
            shape_dist = (mc, batch, self.N_dist, 1)
            shape_agg = (mc, batch, 1, 1)

        shocks_ind = self.shock_ind.sample(shape_ind, antithetic=antithetic, device=device)
        shocks_dist = self.shock_ind.sample(shape_dist, antithetic=antithetic, device=device)
        shocks_agg = self.shocks_agg.sample(shape_agg, antithetic=antithetic, device=device)

        return {"shocks_ind": shocks_ind, "shocks_dist": shocks_dist, "shocks_agg": shocks_agg}

    # --- Model blocks shared by the residuals and the simulator ---

    def _ar1_next(self, s, shock_s, par, scale_ind):
        """Next log productivity: AR(1), floored at ``log(s_min)``, mean level normalized to one."""
        s_next = par.rho_s * s + shock_s * par.sigma_s * scale_ind
        s_next = torch.clamp(s_next, min=math.log(self.s_min))
        return torch.log(torch.exp(s_next) / torch.mean(torch.exp(s_next), dim=-2, keepdim=True))

    @staticmethod
    def factor_prices(A, Kl, N, par):
        """Output and factor prices from the Cobb-Douglas technology."""
        Y = A * Kl ** par.alpha * N ** (1 - par.alpha)
        r = par.alpha * Y / Kl - par.delta
        w = (1 - par.alpha) * Y / N
        return Y, r, w

    @staticmethod
    def _fb_bounds(x, low, high, multiplier):
        """Fisher-Burmeister residual of an asset's first-order condition with the lower limit
        ``low`` and, in the upper part of the asset range, the cap ``high``. ``multiplier`` is one
        when the first-order condition holds with equality."""
        return torch.where(
            x < high * 0.8,
            fisher_burmeister(x - low, 1.0 / multiplier - 1.0),
            fisher_burmeister(high - x, 1.0 - 1.0 / multiplier),
        )

    # --- Network input preparation and policy functions ---

    def _prepare_par_input(self, par, mc=None):
        input_par = par.cat()
        if mc is not None:
            input_par = input_par.unsqueeze(0).expand([mc] + list(input_par.shape))
        return input_par

    def _permute_dist(self, state, mc=None):
        """Distribution sorted by capital, agents moved to the feature dimension."""
        state = state.sort("k", dim=-2)
        return state.permute(dims=(0, 1, 3, 2) if mc is not None else (0, 2, 1))

    def _state_inputs(self, state_dist, state_agg, mc=None, state_ind=None):
        """Network inputs of the state groups: one concatenated tensor per group."""
        groups = {"state_dist": self._permute_dist(state_dist, mc=mc), "state_agg": state_agg}
        if state_ind is not None:
            groups["state_ind"] = state_ind
        return {name: state.cat() for name, state in groups.items()}

    def _hh_choices(self, network, state_ind, state_dist, state_agg, par, mc=None):
        """Household consumption relative to steady-state consumption (above 0.2) and bond
        holdings relative to the bond supply (above -0.2)."""
        hh_output = network(**self._state_inputs(state_dist, state_agg, mc=mc, state_ind=state_ind),
                            par=self._prepare_par_input(par=par, mc=mc))
        c_scale = 0.2 + torch.exp(hh_output[..., 0:1]) / (1.0 / (1 - 0.2))
        b_scale = -0.2 + torch.exp(hh_output[..., 1:2]) / (1.0 / (1 + 0.2))
        return c_scale, b_scale

    def dss_hh_policy(self, state_ind, state_dist, state_agg, par, mc=None):
        return self._hh_choices(self.dss_hh_network, state_ind, state_dist, state_agg, par, mc=mc)

    def hh_policy(self, state_ind, state_dist, state_agg, par, mc=None):
        return self._hh_choices(self.hh_network, state_ind, state_dist, state_agg, par, mc=mc)

    def agg_policy(self, state_dist, state_agg, par, mc=None):
        """Bond price relative to its DSS level, in [0.8, 1.2]."""
        agg_output = self.agg_network(**self._state_inputs(state_dist, state_agg, mc=mc),
                                      par=self._prepare_par_input(par=par, mc=mc))
        return 0.8 + torch.sigmoid(agg_output[..., 0:1]) * (1.2 - 0.8)

    def dss_policy(self, par, mc=None):
        """DSS productivity level relative to the steady-state level and DSS bond price, both in
        [0.8, 1.2]."""
        dss_output = self.dss_network(par=self._prepare_par_input(par=par, mc=mc))
        A_dss_scale = 0.8 + torch.sigmoid(dss_output[..., 0:1]) * (1.2 - 0.8)
        Q_dss = 0.8 + torch.sigmoid(dss_output[..., 1:2]) * (1.2 - 0.8)
        return A_dss_scale, Q_dss

    # --- Residuals ---

    def dss_residuals(self, e, mc, detach_next=True, scale_ind=1, scale_agg=0):
        par, ss = self.par_draw, self.ss
        state_ind, state_dist, state_agg = self.state_ind, self.state_dist, self.state_agg

        c_ind_scale, b_ind_scale = self.dss_hh_policy(state_ind, state_dist, state_agg, par)
        c_ind = c_ind_scale * ss.C
        b_ind = b_ind_scale * par.B
        c_dist_scale, b_dist_scale = self.dss_hh_policy(state_dist, state_dist, state_agg, par)
        c_dist = c_dist_scale * ss.C
        b_dist = b_dist_scale * par.B
        A_dss_scale, Q_dss = self.dss_policy(par)
        A_dss = A_dss_scale * ss.A

        C = torch.mean(c_dist, dim=-2, keepdim=True)
        Kl = state_agg.Kl
        N = ss.N * torch.mean(torch.exp(state_dist.s), dim=-2, keepdim=True)
        Y, r, w = self.factor_prices(A_dss, Kl, N, par)
        T = (1 - Q_dss) * par.B      # lump-sum tax that finances the bond discount
        K = Y - C + (1 - par.delta) * Kl

        # Budget constraint
        k_ind = (1 + r) * state_ind.k + state_ind.b + w * torch.exp(state_ind.s) * ss.N - Q_dss * b_ind - c_ind - T
        k_dist = (1 + r) * state_dist.k + state_dist.b + w * torch.exp(state_dist.s) * ss.N - Q_dss * b_dist - c_dist - T

        # Next period
        shape_ind = e["shocks_ind"].s.shape
        shape_dist = e["shocks_dist"].s.shape
        shape_agg = e["shocks_agg"].A_scale.shape

        s_ind_next = self._ar1_next(state_ind.s, e["shocks_ind"].s, par, scale_ind)
        s_dist_next = self._ar1_next(state_dist.s, e["shocks_dist"].s, par, scale_ind)
        A_next = (1 - par.rho_A) * A_dss + par.rho_A * (state_agg.A_scale + ss.A) + e["shocks_agg"].A_scale * par.sigma_A * scale_agg

        state_ind_next = State({"k": k_ind.expand(shape_ind), "b": b_ind.expand(shape_ind), "s": s_ind_next})
        state_dist_next = State({"k": k_dist.expand(shape_dist), "b": b_dist.expand(shape_dist), "s": s_dist_next})
        state_agg_next = State({"A_scale": A_next - ss.A, "Kl": K.expand(shape_agg)})

        with torch.set_grad_enabled(not detach_next):
            c_ind_scale_next, _ = self.dss_hh_policy(state_ind_next, state_dist_next, state_agg_next, par, mc=mc)
        c_ind_next = c_ind_scale_next * ss.C

        N_next = ss.N * torch.mean(torch.exp(state_dist_next.s), dim=-2, keepdim=True)
        _, r_next, _ = self.factor_prices(A_dss, K, N_next, par)

        # Expected discounted returns (log utility); gamma * (b - B) is the marginal holding cost of bonds
        q = par.beta * (c_ind_next / c_ind) ** (-1)
        lambda_comp = torch.mean(q * (r_next + 1), dim=0)
        psi_comp = (torch.mean(q, dim=0) - par.gamma * (b_ind - par.B)) / Q_dss

        R_fb_k = self._fb_bounds(k_ind, par.k_bar, par.k_down, lambda_comp)
        R_fb_b = self._fb_bounds(b_ind, par.b_bar, par.b_down, psi_comp)
        R_y = Y - ss.Y
        R_k = K - torch.mean(k_dist, dim=-2, keepdim=True)
        R_b = par.B - torch.mean(b_dist, dim=-2, keepdim=True)

        return [R_fb_k, R_fb_b, R_y, R_k, R_b]

    def residuals(self, e, mc, detach_next=True, scale_ind=1, scale_agg=1):
        par, ss = self.par_draw, self.ss
        state_ind, state_dist, state_agg = self.state_ind, self.state_dist, self.state_agg

        c_ind_scale, b_ind_scale = self.hh_policy(state_ind, state_dist, state_agg, par)
        c_ind = c_ind_scale * ss.C
        b_ind = b_ind_scale * par.B
        c_dist_scale, b_dist_scale = self.hh_policy(state_dist, state_dist, state_agg, par)
        c_dist = c_dist_scale * ss.C
        b_dist = b_dist_scale * par.B
        A_dss_scale, Q_dss = self.dss_policy(par)
        A_dss = A_dss_scale * ss.A
        Q = self.agg_policy(state_dist, state_agg, par) * Q_dss
        A = state_agg.A_scale + ss.A

        C = torch.mean(c_dist, dim=-2, keepdim=True)
        Kl = state_agg.Kl
        N = ss.N * torch.mean(torch.exp(state_dist.s), dim=-2, keepdim=True)
        Y, r, w = self.factor_prices(A, Kl, N, par)
        T = (1 - Q) * par.B          # lump-sum tax that finances the bond discount

        # Budget constraint
        k_ind = (1 + r) * state_ind.k + state_ind.b + w * torch.exp(state_ind.s) * ss.N - Q * b_ind - c_ind - T
        k_dist = (1 + r) * state_dist.k + state_dist.b + w * torch.exp(state_dist.s) * ss.N - Q * b_dist - c_dist - T

        K = Y - C + (1 - par.delta) * Kl

        # Next period
        shape_ind = e["shocks_ind"].s.shape
        shape_dist = e["shocks_dist"].s.shape
        shape_agg = e["shocks_agg"].A_scale.shape

        s_ind_next = self._ar1_next(state_ind.s, e["shocks_ind"].s, par, scale_ind)
        s_dist_next = self._ar1_next(state_dist.s, e["shocks_dist"].s, par, scale_ind)
        A_next = (1 - par.rho_A) * A_dss + par.rho_A * (state_agg.A_scale + ss.A) + e["shocks_agg"].A_scale * par.sigma_A * scale_agg

        state_ind_next = State({"k": k_ind.expand(shape_ind), "b": b_ind.expand(shape_ind), "s": s_ind_next})
        state_dist_next = State({"k": k_dist.expand(shape_dist), "b": b_dist.expand(shape_dist), "s": s_dist_next})
        state_agg_next = State({"A_scale": A_next - ss.A, "Kl": K.expand(shape_agg)})

        with torch.set_grad_enabled(not detach_next):
            c_ind_scale_next, _ = self.hh_policy(state_ind_next, state_dist_next, state_agg_next, par, mc=mc)
            c_dist_scale_next, b_dist_scale_next = self.hh_policy(state_dist_next, state_dist_next, state_agg_next, par, mc=mc)
            Q_scale_next = self.agg_policy(state_dist_next, state_agg_next, par, mc=mc)
        c_ind_next = c_ind_scale_next * ss.C
        c_dist_next = c_dist_scale_next * ss.C
        b_dist_next = b_dist_scale_next * par.B
        Q_next = Q_scale_next * Q_dss

        C_next = torch.mean(c_dist_next, dim=-2, keepdim=True)
        N_next = ss.N * torch.mean(torch.exp(state_dist_next.s), dim=-2, keepdim=True)
        Y_next, r_next, w_next = self.factor_prices(A_next, K, N_next, par)
        T_next = (1 - Q_next) * par.B

        # Expected discounted returns (log utility); gamma * (b - B) is the marginal holding cost of bonds
        q = par.beta * (c_ind_next / c_ind) ** (-1)
        lambda_comp = torch.mean(q * (r_next + 1), dim=0)
        psi_comp = (torch.mean(q, dim=0) - par.gamma * (b_ind - par.B)) / Q

        k_dist_next = (1 + r_next) * state_dist_next.k + state_dist_next.b + w_next * torch.exp(state_dist_next.s) * ss.N - Q_next * b_dist_next - T_next - c_dist_next
        K_next = Y_next - C_next + (1 - par.delta) * K

        R_fb_k = self._fb_bounds(k_ind, par.k_bar, par.k_down, lambda_comp)
        R_fb_b = self._fb_bounds(b_ind, par.b_bar, par.b_down, psi_comp)
        R_k = K - torch.mean(k_dist, dim=-2, keepdim=True)
        R_k_next = K_next - torch.mean(k_dist_next, dim=-2, keepdim=True)
        R_b = par.B - torch.mean(b_dist, dim=-2, keepdim=True)

        return [R_fb_k, R_fb_b, R_k, R_k_next, R_b]

    # --- Simulation ---

    @torch.no_grad()
    def dss_step(self, e, scale_ind=1, scale_agg=0, scale_exploration=0.0):
        par, ss = self.par_draw, self.ss
        state_ind, state_dist, state_agg = self.state_ind, self.state_dist, self.state_agg

        c_ind_scale, b_ind_scale = self.dss_hh_policy(state_ind, state_dist, state_agg, par)
        c_ind = c_ind_scale * ss.C
        b_ind = b_ind_scale * par.B
        c_dist_scale, b_dist_scale = self.dss_hh_policy(state_dist, state_dist, state_agg, par)
        c_dist = c_dist_scale * ss.C
        b_dist = b_dist_scale * par.B
        A_dss_scale, Q_dss = self.dss_policy(par)
        A_dss = A_dss_scale * ss.A

        # Exploratory noise on the policies of the state_ind households
        c_ind = c_ind + scale_exploration * torch.std(c_dist, dim=-2, keepdim=True) * torch.randn_like(c_ind)
        b_ind = b_ind + scale_exploration * torch.std(b_dist, dim=-2, keepdim=True) * torch.randn_like(b_ind)

        Kl = torch.mean(state_dist.k, dim=-2, keepdim=True)
        N = ss.N * torch.mean(torch.exp(state_dist.s), dim=-2, keepdim=True)
        _, r, w = self.factor_prices(A_dss, Kl, N, par)
        T = (1 - Q_dss) * par.B

        k_ind = (1 + r) * state_ind.k + state_ind.b + w * torch.exp(state_ind.s) * ss.N - Q_dss * b_ind - T - c_ind
        k_dist = (1 + r) * state_dist.k + state_dist.b + w * torch.exp(state_dist.s) * ss.N - Q_dss * b_dist - T - c_dist
        K = torch.mean(k_dist, dim=-2, keepdim=True)

        s_ind = self._ar1_next(state_ind.s, e["shocks_ind"].s, par, scale_ind)
        s_dist = self._ar1_next(state_dist.s, e["shocks_dist"].s, par, scale_ind)
        A_next = (1 - par.rho_A) * A_dss + par.rho_A * (state_agg.A_scale + ss.A) + e["shocks_agg"].A_scale * par.sigma_A * scale_agg

        return {"state_ind": State({"k": k_ind, "b": b_ind, "s": s_ind}),
                "state_dist": State({"k": k_dist, "b": b_dist, "s": s_dist}),
                "state_agg": State({"A_scale": A_next - ss.A, "Kl": K})}

    @torch.no_grad()
    def step(self, e, scale_ind=1, scale_agg=1, scale_exploration=0.0):
        par, ss = self.par_draw, self.ss
        state_ind, state_dist, state_agg = self.state_ind, self.state_dist, self.state_agg

        c_ind_scale, b_ind_scale = self.hh_policy(state_ind, state_dist, state_agg, par)
        c_ind = c_ind_scale * ss.C
        b_ind = b_ind_scale * par.B
        c_dist_scale, b_dist_scale = self.hh_policy(state_dist, state_dist, state_agg, par)
        c_dist = c_dist_scale * ss.C
        b_dist = b_dist_scale * par.B
        A_dss_scale, Q_dss = self.dss_policy(par)
        A_dss = A_dss_scale * ss.A
        Q = self.agg_policy(state_dist, state_agg, par) * Q_dss

        # Exploratory noise on the consumption policy of the state_ind households
        c_ind = c_ind + scale_exploration * torch.std(c_dist, dim=-2, keepdim=True) * torch.randn_like(c_ind)

        A = state_agg.A_scale + ss.A
        C = torch.mean(c_dist, dim=-2, keepdim=True)
        Kl = state_agg.Kl
        N = ss.N * torch.mean(torch.exp(state_dist.s), dim=-2, keepdim=True)
        Y, r, w = self.factor_prices(A, Kl, N, par)
        T = (1 - Q) * par.B

        k_ind = (1 + r) * state_ind.k + state_ind.b + w * torch.exp(state_ind.s) * ss.N - Q * b_ind - T - c_ind
        k_dist = (1 + r) * state_dist.k + state_dist.b + w * torch.exp(state_dist.s) * ss.N - Q * b_dist - T - c_dist
        K = torch.mean(k_dist, dim=-2, keepdim=True)

        self.sim = {"Y": Y, "C": C, "r": r, "w": w, "A": A, "N": N, "K": K, "Kl": Kl, "Q": Q,
                    "A_scale": state_agg.A_scale}

        s_ind = self._ar1_next(state_ind.s, e["shocks_ind"].s, par, scale_ind)
        s_dist = self._ar1_next(state_dist.s, e["shocks_dist"].s, par, scale_ind)
        A_next = (1 - par.rho_A) * A_dss + par.rho_A * (state_agg.A_scale + ss.A) + e["shocks_agg"].A_scale * par.sigma_A * scale_agg

        return {"state_ind": State({"k": k_ind, "b": b_ind, "s": s_ind}),
                "state_dist": State({"k": k_dist, "b": b_dist, "s": s_dist}),
                "state_agg": State({"A_scale": A_next - ss.A, "Kl": K})}

    def set_state(self, state):
        self.state_ind = state["state_ind"]
        self.state_dist = state["state_dist"]
        self.state_agg = state["state_agg"]

    def dss_steps(self, batch, device, steps, scale_ind=1, scale_agg=0, scale_exploration=0.0):
        for _ in range(steps):
            e = self.draw_shocks(batch, mc=None, antithetic=False, device=device)
            self.set_state(self.dss_step(e, scale_ind=scale_ind, scale_agg=scale_agg, scale_exploration=scale_exploration))

    def steps(self, batch, device, steps, scale_ind=1, scale_agg=1, scale_exploration=0.0):
        for _ in range(steps):
            e = self.draw_shocks(batch, mc=None, antithetic=False, device=device)
            self.set_state(self.step(e, scale_ind=scale_ind, scale_agg=scale_agg, scale_exploration=scale_exploration))

    # --- Training ---

    def loss(self, residuals, criterion, names, weights=None):
        """Weighted sum of per-residual MSEs (the scalar to backprop) plus the unweighted per-residual components for logging."""
        if weights is None:
            weights = [1.0] * len(residuals)
        loss = [weights[i] * criterion(r, target=torch.zeros_like(r)) for i, r in enumerate(residuals)]
        with torch.no_grad():
            loss_components = {name: torch.mean(r ** 2) for name, r in zip(names, residuals)}
        return sum(loss), loss_components

    def initialize(self, batch, device):
        """Set up a batch: fix ``par_draw`` at the calibration, start the state panel at the steady state, move to device."""
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

    def _economies_out_of_range(self):
        """Economies whose panel cannot enter the residuals: a simulated asset holding is not finite
        or is outside its bounds by more than half the distance between them, or aggregate capital
        is more than a factor 5 away from its steady-state level. A boolean per economy of the batch."""
        par = self.par_draw
        Kl, K = self.state_agg.Kl, self.ss.K
        bad = ((Kl < K / 5) | (Kl > 5 * K)).flatten(1).any(dim=1)
        for asset, low, high in [("k", par.k_bar, par.k_down), ("b", par.b_bar, par.b_down)]:
            x = torch.cat([self.state_ind[asset], self.state_dist[asset]], dim=-2)
            margin = 0.5 * (high - low)
            bad = bad | ((x < low - margin) | (x > high + margin) | ~torch.isfinite(x)).flatten(1).any(dim=1)
        return bad

    def _reset_economies(self, bad, batch, device):
        """Give every economy in ``bad`` the panel of a healthy economy, in index order. With no
        healthy economy left, the whole panel restarts at the steady state."""
        healthy = torch.where(~bad)[0]
        if len(healthy) == 0:
            self.initialize_state(batch=batch, device=device)
            return
        dst = torch.where(bad)[0]
        src = healthy[torch.arange(len(dst), device=device) % len(healthy)]
        for state in [self.state_ind, self.state_dist, self.state_agg]:
            for value in state.__dict__.values():
                value[dst] = value[src]

    def _sample_states(self, phase, i, config, scale_ind, scale_agg):
        """Advance the state panel by simulation between two training iterations: one period per
        iteration for the first ``steps_start`` iterations, while the untrained policy settles, then
        ``steps`` periods, with the aggregate shocks amplified by 1.25 as in the guide."""
        steps = 1 if i < config.steps_start else config.steps
        if phase == "dss":
            self.dss_steps(batch=config.batch, device=config.device, steps=steps, scale_ind=scale_ind)
        else:
            self.steps(batch=config.batch, device=config.device, steps=steps, scale_ind=scale_ind, scale_agg=1.25 * scale_agg)

    def _train(self, phase, config):
        """Shared training loop. ``phase`` in {"dss", "full"}."""
        seed_all(config.seed)
        device = config.device

        if phase == "dss":
            trained = [self.dss_hh_network, self.dss_network]
            for net in [self.hh_network, self.agg_network]:
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

        weights = config.weights.initial(self.N_RESIDUALS)

        optimizer = torch.optim.AdamW([p for net in trained for p in net.parameters()],
                                      lr=config.lr, eps=config.adam_eps, weight_decay=config.weight_decay)
        warmup_iter = max(int(config.iteration * config.warmup_frac), 50)
        lr_ramp_end = min(2 * warmup_iter, max(config.iteration // 2, 1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=lr_ramp_end),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.iteration - lr_ramp_end,
                                                       eta_min=config.eta_min),
        ], milestones=[lr_ramp_end])
        criterion = torch.nn.MSELoss(reduction="mean")

        names = self.RESIDUAL_NAMES[phase]
        loss_dict = {"iteration": [], "total": [], "panel_resets": []}
        loss_dict.update({name: [] for name in names})

        pbar = trange(config.iteration)
        running_loss = 0.0
        for i in pbar:
            # Idiosyncratic risk is introduced in the DSS phase, aggregate risk in the full phase.
            if phase == "dss":
                scale_ind = self.adjust_slowly(0.1, 1.0, i, ramp_range=range(1, 1000, 1))
                scale_agg = 0.0
            else:
                scale_ind = 1.0
                scale_agg = self.adjust_slowly(0.1, 1.0, i, ramp_range=range(100, 1000, 1))

            for o in range(config.internal):
                optimizer.zero_grad()
                e = self.draw_shocks(batch=config.batch, mc=config.mc, antithetic=True, device=device)
                residuals = residual_fn(e=e, mc=config.mc, scale_ind=scale_ind, scale_agg=scale_agg,
                                        detach_next=config.detach_next)
                loss, loss_components = self.loss(residuals=residuals, criterion=criterion, weights=weights, names=names)
                if config.log_loss:
                    loss.log().backward()
                else:
                    loss.backward()
                for net in trained:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip, error_if_nonfinite=True)
                optimizer.step()

                if o == 0:
                    loss_dict["iteration"].append(i)
                    loss_dict["total"].append(loss.item())
                    for key, value in loss_components.items():
                        loss_dict[key].append(value.item())
                    running_loss += loss.item()

            if i % config.print_after == 0:
                pbar.set_postfix({"lr": scheduler.get_last_lr()[0], "loss": running_loss / config.print_after})
                running_loss = 0.0

            scheduler.step()

            if i % config.par_draw_after == 0:
                self.par_draw = self.draw_parameters((config.batch, 1, 1), device=device)
                self.ss = self.steady_state()
                step_fn(batch=config.batch, device=device, steps=config.redraw_steps,
                        scale_ind=scale_ind, scale_agg=scale_agg)

            self._sample_states(phase, i, config, scale_ind, scale_agg)

            bad = self._economies_out_of_range()
            if bad.any():
                loss_dict["panel_resets"].append(i)
                self._reset_economies(bad, batch=config.batch, device=device)

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
