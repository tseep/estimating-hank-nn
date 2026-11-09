"""Particle filter and surrogate likelihood estimator.

The model contract: a model passed to ``ParticleFilter`` must expose
``prepare(par_draw, batch, device)``, ``forecast(shocks=None)``, ``resample(idx)``,
and ``to(device)``. ``forecast`` advances one period and returns the observables of
the freshly-propagated state (realize + measure) — one policy eval, on the diverged
particles rather than the resampled ones (no impoverishment). The filter burns in by
calling ``forecast`` (discarding obs, so any ``state_aux`` is current), then filters
with ``forecast`` each period. ``resample`` reindexes the model's state along the
batch (particle) dim: ``state`` (the essential Markov state) and optional
``state_aux`` (quantities carried only for observable construction, e.g. a lagged Y
for growth rates) for representative-agent models, or several containers (individual /
distribution / aggregate) for heterogeneous-agent models.
"""

import math
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from tqdm import trange

from .structures import Parameters
from .networks import NormalizeLayer, ScaleLayer
from .helpers import seed_all

# Surrogate activations by name; checkpoints store the name(s), so save/load round-trips.
ACTIVATIONS = {"CELU": torch.nn.CELU, "ReLU": torch.nn.ReLU, "Tanh": torch.nn.Tanh,
               "SiLU": torch.nn.SiLU, "Mish": torch.nn.Mish}


# %% ParticleFilter
class ParticleFilter:
    """Bootstrap particle filter with Cholesky log-prob and Kitagawa resampling.

    Args:
        model: Model object implementing the generic PF interface.
        data: Dict of observable tensors, each with shape (..., T).
        R: Measurement error covariance matrix (S x S).

    The likelihood path (error, log-prob, accumulators) runs on CPU in float32 by design;
    ``device`` only sets where the model propagates.
    """

    def __init__(self, model, data, R, seed=None):
        seed_all(seed)
        self.model = model
        self.data = data
        self.observable_keys = tuple(data)   # row order of R/L; forecast output is read in this order
        self.S = len(data)
        self.R = R
        self.L = torch.linalg.cholesky(R)
        self.diagnostics = None
        self.dataset = None
        self.network = None
        self.network_arch = None
        self.loss_dict = None

    def to(self, device):
        self.model.to(device)
        if self.network is not None:
            self.network.to(device)
        self.data = {k: v.to(device) for k, v in self.data.items()}
        self.R = self.R.to(device)
        self.L = self.L.to(device)
        if isinstance(self.dataset, dict):
            self.dataset = {k: v.to(device) for k, v in self.dataset.items()}

    def save(self, path, name="pf"):
        """Save ParticleFilter checkpoint: state_dicts and plain data, no pickled modules."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.to("cpu")
        checkpoint = {
            "network_state": self.network.state_dict() if self.network is not None else None,
            "network_arch": self.network_arch,
            "dataset": self.dataset,
            "loss_dict": self.loss_dict,
            "diagnostics": self.diagnostics,
        }
        # Write-then-rename so a crash mid-save cannot corrupt an existing checkpoint.
        torch.save(checkpoint, f"{path}/{name}.pt.tmp")
        Path(f"{path}/{name}.pt.tmp").replace(f"{path}/{name}.pt")

    def load(self, path):
        """Load a checkpoint, rebuilding the surrogate from its stored architecture."""
        checkpoint = torch.load(path, weights_only=True)
        self.dataset = checkpoint.get("dataset")
        self.loss_dict = checkpoint.get("loss_dict")
        self.diagnostics = checkpoint.get("diagnostics")
        self.network = None
        if checkpoint.get("network_state") is not None:
            self.make_surrogate_network(**checkpoint["network_arch"])
            if checkpoint["network_arch"]["scale_output"] and self.dataset is None:
                # Output bounds are saved buffers; loading does not need the training targets.
                state = checkpoint["network_state"]
                prefix = str(len(self.network))
                if f"{prefix}.lower_bound" in state:
                    self.network.append(ScaleLayer(
                        state[f"{prefix}.lower_bound"], state[f"{prefix}.upper_bound"],
                    ))
            self.network.load_state_dict(checkpoint["network_state"])
            self.network.eval()

    def add_dataset(self, paths, reset=False):
        """Load dataset(s) from saved checkpoints and merge into self.dataset."""
        if isinstance(paths, (str, Path)):
            paths = [paths]

        par_chunks = []
        ll_chunks = []

        if (not reset) and self.dataset is not None:
            par_chunks.append(self.dataset["par"])
            ll_chunks.append(self.dataset["log_likelihood"])

        for p in paths:
            ds = torch.load(p, weights_only=False)["dataset"]
            par_chunks.append(ds["par"])
            ll_chunks.append(ds["log_likelihood"])

        self.dataset = {
            "par": torch.cat(par_chunks, dim=0),
            "log_likelihood": torch.cat(ll_chunks, dim=0),
        }

    # --- Core filter ---

    @staticmethod
    def kitagawa(w_norm, P, aux=None):
        """Systematic (Kitagawa) resampling with optional deterministic offset."""
        device = w_norm.device
        dtype = w_norm.dtype

        cum_dist = torch.cumsum(w_norm, dim=-1)
        cum_dist[-1] = torch.tensor(1.0, device=device, dtype=dtype)

        if aux is None:
            u0 = torch.rand((), device=device, dtype=dtype) / P
        else:
            u0 = (torch.as_tensor(aux, device=device, dtype=dtype) % 1.0) / P

        u = u0 + torch.arange(P, device=device, dtype=dtype) / P
        eps = torch.finfo(dtype).eps
        u = torch.clamp(u, max=1.0 - eps)

        idx = torch.searchsorted(cum_dist, u, right=True)
        return idx.to(torch.long)

    @staticmethod
    def log_prob(error_tensor, L):
        """Cholesky-based multivariate normal log probability."""
        z = torch.linalg.solve_triangular(L, error_tensor, upper=False)
        quad = torch.sum(z * z, dim=0)
        logdetR = 2.0 * torch.sum(torch.log(torch.diag(L)))
        S = error_tensor.size(0)
        return -0.5 * (S * math.log(2.0 * math.pi) + logdetR + quad)

    @torch.no_grad()
    def filter(self, P, burn=100, sim=100, par=None, device="cpu", burn_shocks=None, sim_shocks=None, crn_seed=None):
        """Run the bootstrap particle filter.

        Args:
            P: Number of particles.
            burn: Number of burn-in steps.
            sim: Number of filtering steps.
            par: Parameters object (uses model.par if None).
            device: Device to run on.
            burn_shocks: Optional pre-generated burn-in shocks of shape (burn, P, 1).
            sim_shocks: Optional pre-generated filter shocks of shape (sim, P, 1).
            crn_seed: If set (and no explicit shocks are given), reseed before this run so it draws
                the SAME shock (and init-state) stream as every other call — common random numbers
                across parameter draws. Safe because callers draw their parameters ex ante.

        Returns:
            (log_likelihood, filtered_output) tuple.
        """
        if par is None:
            par = deepcopy(self.model.par)
        else:
            par = deepcopy(par)

        S = self.S
        par.to(device)
        self.model.to(device)

        if burn_shocks is not None:
            burn_shocks = {k: v.to(device) for k, v in burn_shocks.items()}
        if sim_shocks is not None:
            sim_shocks = {k: v.to(device) for k, v in sim_shocks.items()}

        # CRN: replay the same shock stream for every call when shocks are drawn internally.
        if crn_seed is not None and burn_shocks is None and sim_shocks is None:
            seed_all(crn_seed)

        self.model.prepare(par.expand((P, 1)), batch=P, device=device)

        if burn_shocks is not None:
            self.model.state = self.model.state_at_ss(P, device=device)

        # Burn-in through forecast (realize each step) so state_aux is current before filtering.
        for t in range(burn):
            if burn_shocks is not None:
                self.model.forecast(shocks={k: v[t:t+1] for k, v in burn_shocks.items()})
            else:
                self.model.forecast()

        # Filter
        filtered_model_out = {key: [] for key in self.data.keys()}
        log_likelihood = torch.empty(sim)
        ess = torch.empty(sim)   # effective sample size per step (degeneracy diagnostic)

        try:
            for t in range(sim):
                # Forecast = step(advance_first=True) + observe: advance, realize the new (propagated,
                # diverged) state, then transform. Weighting the post-advance state (never the resampled
                # one) avoids impoverishment, at one policy eval per step.
                if sim_shocks is not None:
                    model_out = self.model.forecast(shocks={k: v[t:t+1] for k, v in sim_shocks.items()})
                else:
                    model_out = self.model.forecast()

                # Error (on CPU for numerical stability)
                error = torch.empty((S, P), device="cpu")
                for i, key in enumerate(self.observable_keys):
                    error[i, :] = model_out[key].squeeze(-1).cpu() - self.data[key][..., t].cpu()

                # Log probabilities
                log_prob = self.log_prob(error, self.L.cpu())

                # Incremental log-likelihood
                loglik_t = torch.logsumexp(log_prob, dim=0) - math.log(P)
                log_likelihood[t] = loglik_t

                # Normalized weights + effective sample size (1 = full collapse, P = uniform weights)
                w_norm = torch.softmax(log_prob, dim=0)
                ess[t] = 1.0 / torch.sum(w_norm ** 2)

                # Resample (filtered state at time t)
                idx = self.kitagawa(w_norm, P, aux=0.4532)
                self.model.resample(idx)

                # Filtered output
                for key, value in model_out.items():
                    filtered_model_out[key].append(torch.mean(value[idx, ...], dim=0))

            for key, value in filtered_model_out.items():
                filtered_model_out[key] = torch.stack(value, dim=-1)

            # Degeneracy diagnostics from the most recent filter run (resampling stays every-step).
            self.diagnostics = {"ess": ess, "loglik_t": log_likelihood.clone(), "P": P}
            return torch.sum(log_likelihood), filtered_model_out

        # Numerical failures at extreme draws degrade to a NaN target; other errors crash loud.
        except RuntimeError as e:
            print(f"WARNING: Exception in ParticleFilter.filter: {e}")
            return torch.tensor(float("nan")), None

    # --- Dataset generation ---

    def evaluate_likelihood_grid(self, par_name, n=25, sim=100, par=None, P=1000, surrogate=False, random_draws=False, device="cpu", seed=None, burn=100, burn_shocks=None, sim_shocks=None):
        """Run filter over a 1D grid of one parameter."""
        seed_all(seed)
        # Deepcopy either way: the sweep mutates par in place, and a passed object stays the caller's.
        par = deepcopy(self.model.par) if par is None else deepcopy(par)

        lower_bound = self.model.ranges.get(par_name).support.lower_bound
        upper_bound = self.model.ranges.get(par_name).support.upper_bound

        if random_draws:
            grid = lower_bound + (upper_bound - lower_bound) * torch.rand(n, 1)
        else:
            grid = torch.linspace(lower_bound, upper_bound, n).view(-1, 1)

        log_likelihood = torch.empty(n)
        for i in trange(n):
            par.set(par_name, grid[i])

            if surrogate is False:
                try:
                    # random-draw dots exist to show raw filter noise — no CRN for them
                    log_likelihood[i], _ = self.filter(
                        P, par=par, sim=sim, device=device, burn=burn,
                        burn_shocks=burn_shocks, sim_shocks=sim_shocks,
                        crn_seed=None if random_draws else seed,
                    )
                except Exception as e:
                    print(f"WARNING: filter failed at {par_name}={grid[i].item():.6g}: {e}")
                    log_likelihood[i] = float("nan")
            elif surrogate is True:
                with torch.no_grad():
                    net_device = next(self.network.parameters()).device
                    log_likelihood[i] = self.network(par.to(net_device).cat()).item()

        return grid, log_likelihood

    def generate_likelihood_dataset(self, par_names=None, N=1000, par=None, P=1000, sim=100, device="cpu", sobol=False, fast_forward=0, seed=None, burn=100, burn_shocks=None, sim_shocks=None):
        """Generate a dataset of (parameter, log-likelihood) pairs."""
        seed_all(seed)
        if par is None:
            par = deepcopy(self.model.par)

        if par_names is None:
            par_names = self.model.ranges.keys_estimable()
        if isinstance(par_names, str):
            par_names = [par_names]

        if sobol:
            # Unscrambled Sobol: a deterministic sequence, identical regardless of any seed.
            sampler = torch.quasirandom.SobolEngine(len(par_names))
            if isinstance(fast_forward, (int, np.integer)) and fast_forward > 0:
                sampler.fast_forward(int(fast_forward))
            sobol_draws = sampler.draw(N)

        # Draw every parameter vector up front (ex ante), so the per-call CRN reseed inside `filter`
        # cannot interfere with parameter sampling (whether Sobol or RNG-sampled).
        par_draws = []
        for n in range(N):
            par_draw = {}
            idx = 0
            for key in par.keys():
                if key in par_names:
                    if sobol:
                        low = self.model.ranges.get(key).support.lower_bound
                        high = self.model.ranges.get(key).support.upper_bound
                        par_draw[key] = low + (high - low) * sobol_draws[n, idx].view(1)
                        idx += 1
                    else:
                        par_draw[key] = self.model.ranges.get(key).sample((1,))
                else:
                    par_draw[key] = par.get(key)
                par_draw[key] = par_draw[key].to(device)
            par_draws.append(Parameters(par_draw))

        par_list = []
        log_likelihood_list = []
        for n in trange(N):
            par_draw = par_draws[n]
            log_likelihood, _ = self.filter(
                P, par=par_draw, sim=sim, device=device,
                burn=burn, burn_shocks=burn_shocks, sim_shocks=sim_shocks, crn_seed=seed,
            )
            par_list.append(par_draw.cat().to("cpu"))
            log_likelihood_list.append(log_likelihood)

        par_tensor = torch.stack(par_list, dim=0)
        log_likelihood_tensor = torch.stack(log_likelihood_list, dim=0)

        self.dataset = {"par": par_tensor, "log_likelihood": log_likelihood_tensor}

    # --- Surrogate network ---

    def make_surrogate_network(self, N_inputs=None, hidden=64, layers=3, activation="CELU", normalize=True, scale_output=True, seed=None):
        """Create the surrogate likelihood network. ``activation`` is an ``ACTIVATIONS`` name,
        or one name per layer (e.g. ``["Tanh", "CELU", "CELU", "CELU"]``)."""
        seed_all(seed)
        if N_inputs is None:
            N_inputs = len(self.model.par)
        activations = [activation] * layers if isinstance(activation, str) else list(activation)
        assert len(activations) == layers

        N_outputs = 1
        layer_list = []

        if normalize:
            lower_bound = self.model.ranges.low_tensor()
            upper_bound = self.model.ranges.high_tensor()
            layer_list.append(NormalizeLayer(lower_bound, upper_bound))

        layer_list.append(torch.nn.Linear(N_inputs, hidden))
        layer_list.append(ACTIVATIONS[activations[0]]())

        for i in range(1, layers):
            layer_list.append(torch.nn.Linear(hidden, hidden))
            layer_list.append(ACTIVATIONS[activations[i]]())

        layer_list.append(torch.nn.Linear(hidden, N_outputs))

        if scale_output and self.dataset is not None:
            y = self.dataset["log_likelihood"].reshape(-1)
            y = y[torch.isfinite(y)]   # a NaN/inf target must not poison the scale bounds
            lb = y.min()
            ub = y.max()
            if lb == ub:
                lb = lb - 1.0
                ub = ub + 1.0
            layer_list.append(ScaleLayer(lb.view(1, 1), ub.view(1, 1)))

        self.network_arch = {"N_inputs": N_inputs, "hidden": hidden, "layers": layers,
                             "activation": activation,
                             "normalize": normalize, "scale_output": scale_output}
        self.network = torch.nn.Sequential(*layer_list)
        return self.network

    def train(self, batch=64, epochs=10000, device="cpu", lr=1e-3, eta_min=1e-6,
              validation_share=0.2, print_after=1000, seed=None,
              optimizer_name="AdamW", scheduler_name="cosine", step_size=1000, gamma=0.1):
        """Train with AdamW/Adam and a cosine/step learning-rate schedule."""
        seed_all(seed)
        self.network.to(device)

        optimizers = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}
        if optimizer_name not in optimizers:
            raise ValueError("optimizer_name must be AdamW or Adam")
        optimizer = optimizers[optimizer_name](self.network.parameters(), lr=lr)
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=eta_min)
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError("scheduler_name must be cosine or step")

        # Keep only finite targets (without mutating self.dataset)
        finite = torch.isfinite(self.dataset["log_likelihood"])
        if not finite.all():
            print("WARNING: Dataset contains non-finite log-likelihood values.")
            print(f"{(~finite).sum()} non-finite values out of {finite.numel()} samples.")
            par = self.dataset["par"][finite, ...]
            ll = self.dataset["log_likelihood"][finite, ...]
        else:
            par = self.dataset["par"]
            ll = self.dataset["log_likelihood"]

        # Shuffle local views; ``self.dataset`` retains its original ordering so
        # repeated train() calls (with the same seed) produce the same split.
        idx = torch.randperm(par.size(0))
        par = par[idx, ...]
        ll = ll[idx, ...]

        # Train/validation split
        N = par.size(0)
        N_train = int(N * (1 - validation_share))
        dataset_train = torch.utils.data.TensorDataset(par[:N_train, ...], ll[:N_train, ...].unsqueeze(-1))
        dataset_validation = torch.utils.data.TensorDataset(par[N_train:, ...], ll[N_train:, ...].unsqueeze(-1))

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch, shuffle=True)
        dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch, shuffle=True)

        self.network.train()
        self.network.to(device)

        pbar = trange(epochs, desc="PF NN Training")
        self.loss_dict = {"iteration": [], "training_loss": [], "validation_loss": []}

        for i in pbar:
            for x, y in dataloader_train:
                x = x.to(device)
                y = y.to(device)
                y_hat = self.network(x)
                loss = torch.nn.functional.mse_loss(y_hat, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            if i % print_after == 0:
                with torch.no_grad():
                    training_loss = 0.0
                    for x, y in dataloader_train:
                        x = x.to(device)
                        y = y.to(device)
                        y_hat = self.network(x)
                        training_loss += torch.nn.functional.mse_loss(y_hat, y, reduction="sum").item()
                    training_loss = training_loss / len(dataset_train)

                with torch.no_grad():
                    validation_loss = 0.0
                    for x, y in dataloader_validation:
                        x = x.to(device)
                        y = y.to(device)
                        y_hat = self.network(x)
                        validation_loss += torch.nn.functional.mse_loss(y_hat, y, reduction="sum").item()
                    validation_loss = validation_loss / len(dataset_validation)

                self.loss_dict["iteration"].append(i)
                self.loss_dict["training_loss"].append(training_loss)
                self.loss_dict["validation_loss"].append(validation_loss)

                pbar.set_postfix({"lr:": scheduler.get_last_lr()[0], "training": training_loss, "validation": validation_loss}, refresh=False)

        self.network.eval()


# %% Compute likelihood grid
def compute_likelihood(pf, parameter, kind="surrogate", n=None, P=100, sim=100, device="cpu", seed=None, burn=100, burn_shocks=None, sim_shocks=None, par=None):
    """Evaluate the likelihood of one parameter on a grid.

    Args:
        kind: One of "surrogate", "full", "random".
            - "surrogate": linspace grid, surrogate network (fast).
            - "full":      linspace grid, particle filter.
            - "random":    uniform random draws, particle filter.
        n: Grid size. Defaults to 200 for "random", 50 otherwise.
        P: Number of particles (ignored for "surrogate").
        sim: Simulation length (ignored for "surrogate").
        burn_shocks, sim_shocks: Optional pre-loaded PF shocks (ignored for
            "surrogate").
        par: Parameters the non-swept entries are held at (default: the model calibration).

    Returns dict with kind-specific keys:
        surrogate -> {"grid", "ll_grid_surrogate"}
        full      -> {"grid", "ll_grid_full"}
        random    -> {"grid_random", "ll_grid_random"}
    """
    if n is None:
        n = 200 if kind == "random" else 50
    grid, ll = pf.evaluate_likelihood_grid(
        par_name=parameter, n=n, P=P, sim=sim, device=device,
        surrogate=(kind == "surrogate"),
        random_draws=(kind == "random"),
        seed=seed,
        burn=burn,
        burn_shocks=burn_shocks,
        sim_shocks=sim_shocks,
        par=par,
    )
    grid_key = "grid_random" if kind == "random" else "grid"
    return {grid_key: grid, f"ll_grid_{kind}": ll}


def add_log_prior(result, prior, parameter):
    """Add posterior_* keys to a likelihood result dict (in place)."""
    if parameter not in prior:
        return result
    if "ll_grid_surrogate" in result:
        result["posterior_surrogate"] = result["ll_grid_surrogate"] + prior[parameter].log_prob(result["grid"].view(-1))
    if "ll_grid_full" in result:
        result["posterior_full"] = result["ll_grid_full"] + prior[parameter].log_prob(result["grid"].view(-1))
    if "ll_grid_random" in result:
        result["posterior_random"] = result["ll_grid_random"] + prior[parameter].log_prob(result["grid_random"].view(-1))
    return result
