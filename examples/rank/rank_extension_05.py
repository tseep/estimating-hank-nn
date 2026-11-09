# %% [markdown]
# # Extension 5: Pseudo-state variables
#
# We fix the Taylor-rule coefficients at phipi = 1.875 and phiy = 0.25. The network now varies over eight estimated parameters. The calibrated coefficients still enter the equations.
#
# Comments starting with `New in Extension` mark the additions.

# %% [markdown]
# ## 1. Set up
#
# Install the environment once with `pip install -r requirements.txt` and
# `pip install -e .` from the repository root. Select that environment as this
# notebook's kernel, then run the cells in order. Run the notebook from its own
# folder, which Jupyter and VS Code do by default.

# %%
from pathlib import Path

import torch

from rank_plots import plot_par_list, plot_3d_policy_par
from estimating_hank_nn.helpers import add_measurement_error
from estimating_hank_nn.particle_filter import ParticleFilter, compute_likelihood
from estimating_hank_nn.plots import (
    plot_loss, plot_pf_surrogate_loss, plot_posterior_single, plot_posterior_parameters,
)
from estimating_hank_nn.plot_style import set_rc_params, save_figure

# %% [markdown]
# ### Load saved results or retrain
#
# Leave the flags `True` to load saved results. Set `load_model = False` to
# solve the model again and rebuild its observations and likelihood results.
# Set `load_pf_dataset = False` to regenerate likelihood targets and retrain the
# likelihood network. Set only `load_pf_network = False` to retrain that network
# using saved targets.
#
# Any missing stage is computed automatically. Full training can take a while.
# The cells print whether they are loading or training.

# %%
load_model = True
load_pf_dataset = True
load_pf_network = True
DEVICE = "cpu"  # Use "cuda" on an NVIDIA GPU or "mps" on an Apple GPU.

torch.set_num_threads(4)
set_rc_params()

# Outputs are stored next to this notebook.
folder = Path("outputs") / "extension_05"
figures = folder / "figures"
figures.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Connect the equations to the code
#
# The network predicts the policies in `POLICY_NAMES` from the states in `STATE_NAMES`
# and the parameters that have a training range.

# %% [markdown]
# The equations are
#
# $$
# \begin{aligned}
# \hat X_t &= E_t\hat X_{t+1}-\sigma^{-1}(\hat I_t-E_t\hat\Pi_{t+1}-\hat R_t^*),\\
# \hat\Pi_t &= \kappa\hat X_t+\beta E_t\hat\Pi_{t+1},\\
# \hat I_t &= \rho_I\hat I_{t-1}+(1-\rho_I)(\phi_\Pi\hat\Pi_t+\phi_Y\hat X_t)+\widehat{mp}_t,\\
# \hat R_t^* &= \rho_A\hat R_{t-1}^*+\sigma(\rho_A-1)\omega\sigma_A\epsilon_t^A,\\
# \widehat{mp}_t &= \sigma_{MP}\epsilon_t^{MP},\qquad \epsilon_t^{MP}\sim N(0,1).
# \end{aligned}
# $$
#
# where $\kappa=(1-\phi)(1-\phi\beta)(\sigma+\eta)/\phi$ and
# $\omega=(1+\eta)/(\sigma+\eta)$. In the code, `zeta` denotes the natural rate,
# `X` the output gap, `Pi` inflation, and `I` the nominal rate.
#
# The complete model is defined below. Read `policy` for the network inputs and
# outputs, `next_state` for the state transition, and `residuals` for the equation errors.
# The remaining methods draw training data, update the weights, and save and load results.
#
# ### Loss
#
# For a batch of $N_{batch}$ simulated economies and equation residuals $F_j$, we minimize
#
# $$\mathcal L=\frac{1}{N_{batch}}\sum_{b=1}^{N_{batch}}\sum_{j=1}^{J}
# \alpha_j\left(F_j^{(b)}\right)^2.$$
#
# Each weight $\alpha_j$ is 1,000, as in the replication. A trained network's mean squared
# residuals are around $10^{-8}$, so most gradients fall below the optimizer's $\epsilon=10^{-8}$
# and the updates stall. The factor of 1,000 keeps them well above it. `loss` averages each
# squared residual over the batch. The residuals $F_j$ are the equations above, in the order of
# `RESIDUAL_NAMES`.
# The training loop differentiates this sum and updates the network.

# %% [markdown]
# ### Complete model
#
# This cell defines the model but does not train it. The next section controls
# whether to load saved weights or solve the model.

# %%
from dataclasses import asdict, dataclass, field

from tqdm import trange

from estimating_hank_nn.helpers import ergodic_sigma, seed_all
from estimating_hank_nn.networks import NormalizeLayer
from estimating_hank_nn.loss_weights import ConstantWeights, ReLoBRaLo
from estimating_hank_nn.structures import Element, Parameters, Ranges, Shocks, State


@dataclass
class TrainingConfig:
    """Settings for the policy-network training loop."""

    # Sizes of the training loop.
    iteration: int = 10000
    steps: int = 10
    batch: int = 100
    mc: int = 2
    inner_steps: int = 5
    par_draw_after: int = 1

    # Learning rate and its lower limit under cosine decay.
    lr: float = 1e-3
    min_lr: float = 1e-10

    # ConstantWeights fixes the equation weights, and ReLoBRaLo adapts them.
    weights: object = field(default_factory=ConstantWeights)
    detach_next: bool = True

    device: str = "cpu"
    print_after: int = 100
    seed: int | None = None


# Reference values. They also generate the data later on.
PARAMETERS = {
    "beta": 0.97,
    "sigma": 2.0,
    "eta": 1.125,
    "phi": 0.7,
    "phipi": 1.875,
    "phiy": 0.25,
    "rho_i": 0.5,  # New in Extension 3: positive reference value.
    "rho_a": 0.875,
    "sigma_a": 0.06,
    "sigma_mp": 0.0075,  # New in Extension 4: monetary-shock volatility.
}

# Training ranges. Every parameter with a range is a network input.
# New in Extension 5: phipi and phiy have no range, so they stay at their reference values.
RANGES = {
    "beta": torch.distributions.Uniform(0.95, 0.99),
    "sigma": torch.distributions.Uniform(1.0, 3.0),
    "eta": torch.distributions.Uniform(0.25, 2.0),
    "phi": torch.distributions.Uniform(0.5, 0.9),
    "rho_i": torch.distributions.Uniform(0.0, 0.75),
    "rho_a": torch.distributions.Uniform(0.8, 0.95),
    "sigma_a": torch.distributions.Uniform(0.02, 0.1),
    "sigma_mp": torch.distributions.Uniform(0.005, 0.01),
}

# Distributions of the shock innovations.
SHOCKS = {"zeta": torch.distributions.Normal(0.0, 1.0), "mp": torch.distributions.Normal(0.0, 1.0)}


class NKModelExt5:
    """Complete model, with output, inflation, and the nominal rate as policies."""

    # New in Extension 1: the network also returns the nominal interest rate.
    POLICY_NAMES = ("X", "Pi", "I")
    # New in Extension 3: the lagged interest rate is a state.
    # New in Extension 4: the monetary shock is an exogenous state.
    STATE_NAMES = ("zeta", "IL", "mp")
    # Equation residuals, in the order of the equations above.
    RESIDUAL_NAMES = ("res_bond_euler", "res_nkpc", "res_taylor")
    RESIDUAL_LABELS = ("Euler equation", "Phillips curve", "Taylor rule")

    def __init__(self, parameters, ranges, shocks, seed=1234):
        seed_all(seed)
        self.par = Parameters(parameters)
        self.ranges = Ranges(parameters, ranges)
        self.shocks = Shocks(shocks)
        self.network = self.make_network()
        self.par_draw = self.ss = self.state = None
        self.loss_dict = self.training_conf = None

    def state_bounds(self):
        return torch.tensor([-0.025, -0.025, -0.025]), torch.tensor([0.025, 0.025, 0.025])

    def make_network(self, hidden=64, layers=5, activation=torch.nn.CELU):
        """Normalize states and parameters, then apply a fully connected network."""
        layer_list = []

        # Normalize layer
        state_low, state_high = self.state_bounds()
        # New in Extension 5: only the parameters with a range are inputs.
        keys = self.ranges.keys_estimable()
        low = torch.cat((state_low, self.ranges.low_tensor(keys)))
        high = torch.cat((state_high, self.ranges.high_tensor(keys)))
        layer_list.append(NormalizeLayer(low, high))

        # First layer
        layer_list.append(torch.nn.Linear(len(low), hidden))
        layer_list.append(activation())

        # Middle layers
        for _ in range(1, layers):
            layer_list.append(torch.nn.Linear(hidden, hidden))
            layer_list.append(activation())

        # Last layer
        layer_list.append(torch.nn.Linear(hidden, len(self.POLICY_NAMES)))

        return torch.nn.Sequential(*layer_list)

    def policy(self, state=None, par=None):
        state = self.state if state is None else state
        par = self.par_draw if par is None else par

        # Vectors of states and parameters.
        states = state.cat(keys=self.STATE_NAMES)
        # New in Extension 5: pass only the parameters with a range.
        parameters = par.cat(keys=self.ranges.keys_estimable())

        # Future states have an extra leading dimension for the shock draws.
        # Repeat the parameters along it to evaluate expectations.
        if states.dim() > parameters.dim():
            parameters = parameters.expand(states.shape[0], *parameters.shape)

        # Evaluate the network on the combined inputs.
        inputs = torch.cat((states, parameters), dim=-1)
        output = self.network(inputs)

        X = output[..., 0:1] / 100
        Pi = output[..., 1:2] / 100
        I = output[..., 2:3] / 100

        return X, Pi, I

    def steady_state(self, par=None):
        par = self.par_draw if par is None else par
        kappa = (1 - par.phi) * (1 - par.phi * par.beta) * (par.sigma + par.eta) / par.phi
        omega = (1 + par.eta) / (par.sigma + par.eta)
        return Parameters({"kappa": kappa, "omega": omega})

    def initialize_state(self, par=None, batch=100, device="cpu"):
        """Draw the natural rate from its stationary distribution."""
        par = self.par_draw if par is None else par
        loading = par.sigma_a * par.sigma * (par.rho_a - 1) * self.steady_state(par).omega
        std = ergodic_sigma(par.rho_a, loading.abs())
        # New in Extension 3: start with a range of lagged interest rates.
        lagged_rate = -0.01 + 0.02 * torch.rand(batch, 1, device=device)
        zeta = torch.randn(batch, 1, device=device) * std
        # New in Extension 4: draw today's monetary shock.
        mp = par.sigma_mp * torch.randn(batch, 1, device=device)
        return State({"zeta": zeta, "IL": lagged_rate, "mp": mp})

    def state_at_ss(self, batch, device="cpu"):
        zeros = torch.zeros(batch, 1, device=device)
        return State({"zeta": zeros, "IL": zeros.clone(), "mp": zeros.clone()})

    def draw_parameters(self, shape, device="cpu"):
        return self.ranges.sample(shape, device=device)

    def draw_shocks(self, shape, antithetic=False, device="cpu"):
        return self.shocks.sample(shape, antithetic, device=device)

    def prepare(self, par_draw, batch, device="cpu"):
        self.par_draw = par_draw.to(device)
        self.ss = self.steady_state()
        self.state = self.initialize_state(batch=batch, device=device)

    @torch.no_grad()
    def next_state(self, e):
        """Return next period's state for the shock draws in e, without changing the current state."""
        par = self.par_draw
        loading = par.sigma_a * par.sigma * (par.rho_a - 1) * self.ss.omega
        zeta_next = par.rho_a * self.state.zeta + loading * e.zeta
        # New in Extension 3: today's interest rate becomes next period's lagged rate.
        # Repeat it across the shock draws to match the shape of zeta_next.
        _, _, I = self.policy()
        # New in Extension 4: the next monetary shock is independent of today's state.
        mp_next = par.sigma_mp * e.mp
        return State({"zeta": zeta_next, "IL": I.expand_as(zeta_next), "mp": mp_next})

    @torch.no_grad()
    def step(self, e, advance_first=True):
        """Move the economies one period and return the realized variables.

        With advance_first=True (particle filter and simulation), the state moves first and the
        policies are evaluated at the new state. With advance_first=False (training), they are
        evaluated at the current state, consistent with residuals.
        """
        state_next = self.next_state(e)
        state = state_next if advance_first else self.state
        X, Pi, I = self.policy(state)
        self.state = state_next
        return {"R": state.zeta, "X": X, "Pi": Pi, "I": I, "mp": state.mp}

    def steps(self, batch, device, steps, shocks=None):
        """Simulate the training economies. shocks (name -> (steps, batch, 1)) replaces fresh draws."""
        for t in range(steps):
            if shocks is None:
                e = self.draw_shocks((batch, 1), device=device)
            else:
                e = Element({key: value[t] for key, value in shocks.items()})
            self.step(e, advance_first=False)

    def residuals(self, e, detach_next=True):
        par = self.par_draw

        # Evaluate today's policies.
        X, Pi, I = self.policy()

        # Next period's state for each shock draw.
        state_next = self.next_state(e)

        # Evaluate next period's policies at that state.
        with torch.set_grad_enabled(not detach_next):
            X_next, Pi_next, _ = self.policy(state_next)

        # Average over the draws to approximate the conditional expectations.
        EX_next = X_next.mean(dim=0)
        EPi_next = Pi_next.mean(dim=0)

        # F_1: Euler equation.
        euler = X - (EX_next - (I - EPi_next - self.state.zeta) / par.sigma)
        # F_2: Phillips curve.
        phillips = Pi - (self.ss.kappa * X + par.beta * EPi_next)
        # F_3: Taylor rule. New in Extension 4: the monetary shock enters.
        taylor_rate = par.rho_i * self.state.IL + (1 - par.rho_i) * (par.phipi * Pi + par.phiy * X)
        taylor = I - (taylor_rate + self.state.mp)
        return [euler, phillips, taylor]

    def loss(self, residuals, weights):
        """Return the weighted mean squared error and each unweighted equation loss."""
        equation_losses = [residual.square().mean() for residual in residuals]
        loss = sum(weight * value for weight, value in zip(weights, equation_losses))
        components = {name: value.item() for name, value in zip(self.RESIDUAL_NAMES, equation_losses)}
        return loss, components

    def train_model(self, config):
        """Train the policy network with the settings in config."""
        seed_all(config.seed)
        self.training_conf = config

        # Set the network to train mode.
        self.network.train()
        self.to(config.device)

        # Draw parameters and initial states.
        par_draw = self.draw_parameters((config.batch, 1), device=config.device)
        self.prepare(par_draw, batch=config.batch, device=config.device)

        # Starting weights for the equations.
        weights = config.weights.initial(len(self.RESIDUAL_NAMES))
        weight_state = None

        # Optimizer and scheduler.
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.iteration, eta_min=config.min_lr)

        # Loss history: the total, each equation, and each equation weight.
        self.loss_dict = {"iteration": [], "total": []}
        for name in self.RESIDUAL_NAMES:
            self.loss_dict[name] = []
            self.loss_dict[f"w_{name}"] = []

        # Training loop.
        pbar = trange(config.iteration)
        running_loss = 0.0
        for i in pbar:
            # Update the network several times on the same states, with new shocks each time.
            for o in range(config.inner_steps):
                optimizer.zero_grad()
                shape = (config.mc, config.batch, 1)
                e = self.draw_shocks(shape, antithetic=True, device=config.device)
                residuals = self.residuals(e, config.detach_next)
                loss, components = self.loss(residuals, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

                # Record the first update, before the network adapts to these states.
                if o == 0:
                    self.loss_dict["iteration"].append(i)
                    self.loss_dict["total"].append(loss.item())
                    for name, weight in zip(self.RESIDUAL_NAMES, weights):
                        self.loss_dict[name].append(components[name])
                        self.loss_dict[f"w_{name}"].append(weight)
                    running_loss += loss.item()

            # Update adaptive weights. Fixed weights stay unchanged.
            losses = list(components.values())
            weights, weight_state = config.weights.update(
                weights, losses, i, warmup_iter=None, state=weight_state)

            # Show the running loss in the progress bar.
            if i % config.print_after == 0:
                pbar.set_postfix({
                    "loss": running_loss / config.print_after,
                    "lr": scheduler.get_last_lr()[0],
                })
                running_loss = 0.0

            # Update the learning rate.
            scheduler.step()

            # Draw new parameters.
            if i % config.par_draw_after == 0:
                self.par_draw = self.draw_parameters((config.batch, 1), device=config.device)
                self.ss = self.steady_state()

            # Sample new states by simulation.
            self.steps(config.batch, config.device, config.steps)

        # Set the network to evaluation mode.
        self.network.eval()
        self.to("cpu")

    # --- Interface used by the particle filter ---

    def observe(self, obs_raw):
        return obs_raw

    @torch.no_grad()
    def forecast(self, shocks=None):
        """Advance the state and return the observed variables. shocks (name -> (1, batch, 1))
        replaces fresh draws."""
        if shocks is None:
            e = self.draw_shocks(self.state.zeta.shape, device=self.state.zeta.device)
        else:
            e = Element({key: value[0] for key, value in shocks.items()})
        return self.observe(self.step(e, advance_first=True))

    def resample(self, idx):
        self.state = self.state.reindex(idx)

    @torch.no_grad()
    def simulate(self, batch=1, par=None, burn=1000, steps=1000, device="cpu", seed=12345, shocks=None):
        """Simulate the model and return each observed variable as a time series. shocks
        (name -> (steps, batch, 1)) replays fixed innovations in the recorded periods."""
        seed_all(seed)
        self.to(device)

        # Start every economy at the given parameters and burn in.
        par = self.par if par is None else par
        self.prepare(par.expand((batch, 1)), batch, device)
        self.steps(batch, device, burn)

        # Record the observed variables, replaying the given shocks if any.
        path = []
        for t in range(steps):
            if shocks is None:
                shocks_t = None
            else:
                shocks_t = {key: value[t:t + 1] for key, value in shocks.items()}
            path.append(self.forecast(shocks_t))

        # Stack each observed variable over time.
        series = {}
        for key in path[0]:
            values = [obs[key].squeeze(-1) for obs in path]
            series[key] = torch.stack(values, dim=-1).cpu()
        return series

    def to(self, device):
        for obj in (self.par, self.par_draw, self.ss, self.state, self.network):
            if obj is not None:
                obj.to(device)
        return self

    def save(self, path):
        torch.save({"state_dict": self.network.state_dict(), "loss_dict": self.loss_dict,
                    "training_conf": asdict(self.training_conf)}, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.loss_dict = checkpoint["loss_dict"]
        self.training_conf = checkpoint["training_conf"]
        self.network.eval()

# %% [markdown]
# ## 3. Solve the model
#
# The policy network has five hidden layers of 64 units with CELU activation.
# Training draws parameters and shocks, evaluates the equation errors, and updates
# the weights. `TrainingConfig` collects the settings used by the training loop.
# `detach_next=True` holds next-period policies fixed in the gradient.
# Set `weights=ReLoBRaLo()` for adaptive equation weights.

# %%
model = NKModelExt5(PARAMETERS, RANGES, SHOCKS)
model_path = folder / "model.pt"
train_model = not load_model or not model_path.exists()

training_config = TrainingConfig(
    iteration=10000,
    inner_steps=5,
    steps=10,
    batch=500,
    mc=2,
    par_draw_after=1,
    lr=1e-3,
    min_lr=1e-10,
    device=DEVICE,
    seed=1234,
    weights=ConstantWeights([1000.0, 1000.0, 1000.0]),
    detach_next=True,
)

if train_model:
    print("Training the policy network...")
    model.train_model(training_config)
    # A new policy needs new observations, so delete the saved ones.
    (folder / "observations.pt").unlink(missing_ok=True)
    model.save(model_path)
else:
    print("Loading the saved policy network.")
    model.load(model_path)

# %% [markdown]
# ### Inspect the training loss
#
# As in the replication, the plots report mean squared residuals: their average
# across equations on top and each equation below. The curves use a moving
# median over 100 iterations.

# %%
fig, _ = plot_loss(
    model.loss_dict, residual_names=model.RESIDUAL_LABELS, smooth=("median", 100),
)
save_figure(fig, str(figures / "policy_loss"), format_list=["pdf", "png"])

# %% [markdown]
# ### Inspect the learned policies
#
# These parameter panels use the plotting layout from the analytical replication.
# Each panel varies one parameter while holding the others at their reference values.
# We set the natural rate to the response to a productivity innovation of −1.
# These extensions show the neural policies without an analytical comparison.
# Other state variables are held at zero.

# %%
parameter_labels = {
    "beta": ("Discount factor", r"$\beta$"),
    "sigma": ("Relative risk aversion", r"$\sigma$"),
    "eta": ("Inverse Frisch elasticity", r"$\eta$"),
    "phi": ("Price duration", r"$\phi$"),
    "phipi": ("MP inflation response", r"$\phi_{\Pi}$"),
    "phiy": ("MP output response", r"$\phi_{Y}$"),
    "rho_i": ("Interest-rate persistence", r"$\rho_I$"),
    "rho_a": ("Persistence shock", r"$\rho_A$"),
    "sigma_a": ("Standard deviation shock", r"$\sigma_A$"),
    "sigma_mp": ("Monetary shock volatility", r"$\sigma_{MP}$"),
}

for policy in model.POLICY_NAMES:
    fig, _ = plot_par_list(
        model, model.policy,
        policy=policy, par_list=model.ranges.keys_estimable(),
        par_labels=parameter_labels, n_cols=2, panel_titles=False,
    )
    fig.suptitle({"X": "Output gap", "Pi": "Inflation", "I": "Nominal interest rate"}[policy])
    save_figure(fig, str(figures / f"policy_{policy}"), format_list=["pdf", "png"])

# %% [markdown]
# The surface shows how inflation varies jointly with price stickiness and the
# natural rate. The dashed line marks the reference value of price stickiness.

# %%
fig, _ = plot_3d_policy_par(
    model, model.policy, plot="Pi", par_name="phi", par_labels=parameter_labels, n=30,
)
save_figure(fig, str(figures / "inflation_surface"), format_list=["pdf", "png"])

# %% [markdown]
# ## 4. Generate observations
#
# Simulate the solved model at the reference parameters. Add independent measurement
# noise with variance equal to 10% of each simulated series' variance. `R` below is
# the measurement-error covariance matrix passed to the particle filter.
#
# We reuse these observations unless they are missing or the policy was retrained.

# %%
data_path = folder / "observations.pt"
pf_dataset_path = folder / "datasets" / "likelihood.pt"
make_data = train_model or not data_path.exists()

if make_data:
    data = model.simulate(steps=1000, seed=12345)
    data, R = add_measurement_error(data, error_share=0.1, seed=12345)
    # Discard likelihood results that used the previous observations.
    pf_dataset_path.unlink(missing_ok=True)
    (folder / "surrogate.pt").unlink(missing_ok=True)
    (folder / "likelihood_sigma.pt").unlink(missing_ok=True)
    (folder / "likelihood_all.pt").unlink(missing_ok=True)
    torch.save({"data": data, "R": R}, data_path)
else:
    observations = torch.load(data_path, weights_only=True)
    data, R = observations["data"], observations["R"]

pf = ParticleFilter(model, data, R)

# %% [markdown]
# ## 5. Learn the likelihood
#
# The particle filter evaluates the likelihood for 5,000 parameter draws. A second
# network learns the mapping from parameters to log likelihood. It has three hidden
# layers of 64 CELU units, and 20% of the likelihood dataset is reserved for validation.
#
# Likelihood targets are saved separately so the network can be retrained
# without running the particle filter again. If the saved network is reused,
# its training targets do not need to be loaded.
#
# ### Generate or load likelihood targets

# %%
surrogate_path = folder / "surrogate.pt"
train_pf_network = not load_pf_network or not load_pf_dataset or make_data or not surrogate_path.exists()
make_pf_dataset = train_pf_network and (not load_pf_dataset or make_data or not pf_dataset_path.exists())

if make_pf_dataset:
    print("Generating likelihood targets...")
    pf.generate_likelihood_dataset(
        N=5000, P=100, sim=100, burn=100, sobol=True, device=DEVICE, seed=12345,
    )
    pf_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pf.dataset, pf_dataset_path)
elif train_pf_network:
    print("Loading the saved likelihood targets.")
    pf.dataset = torch.load(pf_dataset_path, weights_only=True)

# %% [markdown]
# ### Train or load the likelihood network

# %%
if train_pf_network:
    (folder / "likelihood_sigma.pt").unlink(missing_ok=True)
    (folder / "likelihood_all.pt").unlink(missing_ok=True)
    print("Training the likelihood network...")
    pf.make_surrogate_network(hidden=64, layers=3, activation="CELU", seed=1234)
    pf.train(
        batch=100, epochs=5000, lr=1e-3, eta_min=1e-8,
        validation_share=0.2, print_after=100, device=DEVICE, seed=1234,
    )
    pf.dataset = None  # Targets are stored separately in pf_dataset_path.
    pf.save(folder, name="surrogate")
else:
    print("Loading the saved likelihood network.")
    pf.load(surrogate_path)

# %% [markdown]
# The training and validation losses help us see whether the network approximates
# likelihood values for parameter draws it did not train on.
# Choose the number of epochs with the dataset and batch sizes in mind. If validation
# loss keeps rising while training loss falls, the likelihood network may be overfitting.

# %%
fig, _ = plot_pf_surrogate_loss(pf.loss_dict)
save_figure(fig, str(figures / "likelihood_loss"), format_list=["pdf", "png"])

# %% [markdown]
# ## 6. Compare the likelihood curves
#
# Hold the other parameters at their reference values and vary one parameter at a time.
# The line is the likelihood network. The dots are particle-filter evaluations at random
# values of that parameter, with the same 100 particles and 100 observations as the
# likelihood targets. The dots scatter because each evaluation is a noisy estimate, and the
# network should run through their center. As in the paper, we first show risk aversion,
# then every estimated parameter.
#
# The evaluations are saved and reused. Retraining the likelihood also recomputes them.

# %%
likelihood_path = folder / "likelihood_sigma.pt"
likelihood_all_path = folder / "likelihood_all.pt"
# Use the same particle-filter settings as for the likelihood targets.
pf_settings = dict(P=100, sim=100, burn=100, seed=1234)

if train_pf_network or not likelihood_path.exists():
    likelihood = compute_likelihood(pf, "sigma", kind="surrogate", n=50)
    likelihood.update(compute_likelihood(pf, "sigma", kind="random", n=500, **pf_settings))
    torch.save(likelihood, likelihood_path)
else:
    likelihood = torch.load(likelihood_path, weights_only=True)

fig, _ = plot_posterior_single(
    likelihood, "sigma", true_value=model.par.sigma.item(), par_labels=parameter_labels,
)
save_figure(fig, str(figures / "likelihood"), format_list=["pdf", "png"])

# %% [markdown]
# The same comparison for every estimated parameter.

# %%
if train_pf_network or not likelihood_all_path.exists():
    likelihood_all = {}
    for name in model.ranges.keys_estimable():
        result = compute_likelihood(pf, name, kind="surrogate", n=50)
        result.update(compute_likelihood(pf, name, kind="random", n=200, **pf_settings))
        likelihood_all[name] = result
    torch.save(likelihood_all, likelihood_all_path)
else:
    likelihood_all = torch.load(likelihood_all_path, weights_only=True)

fig, _ = plot_posterior_parameters(
    likelihood_all, true_values={name: model.par[name].item() for name in likelihood_all},
    par_labels=parameter_labels, y_buffer=50, n_cols=2,
)
save_figure(fig, str(figures / "likelihood_all"), format_list=["pdf", "png"])
