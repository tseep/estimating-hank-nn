# %% [markdown]
# # Krusell–Smith model with two assets
#
# The one-asset model with a second asset, a government bond in fixed supply. Households
# choose consumption and bonds, and capital follows from the budget constraint. We first
# solve the deterministic steady state (DSS), then the model with aggregate risk, and
# compare both with the sequence-space Jacobian (SSJ) reference. The website walks through
# the model and the code.

# %% [markdown]
# ## 1. Set up

# %%
import math
from pathlib import Path

import pandas as pd
import torch

from estimating_hank_nn.helpers import seed_all
from estimating_hank_nn.loss_weights import ConstantWeights
from estimating_hank_nn.plot_style import set_rc_params, save_figure, DUO_SIZE, MULTI_SIZE
from estimating_hank_nn.plots import plot_loss

from kstwo_model import KrusellSmithTwoAsset, TrainingConfig
import kstwo_analysis as analysis
import kstwo_plots as plots

# %% [markdown]
# The saved networks and analysis results load by default. Set a flag to `False` to
# recompute that stage. Retraining takes hours, even on a GPU.

# %%
load_dss = True       # DSS networks
load_full = True      # full-model networks, which also hold the DSS networks
load_analysis = True  # analysis results
DEVICE = "cpu"        # "cuda" for an NVIDIA GPU, "mps" for an Apple GPU

# Stop with an error if the selected GPU is unavailable, rather than switching devices.
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable. Select DEVICE = "mps" or "cpu" explicitly.')
if DEVICE == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError('MPS is unavailable. Select DEVICE = "cuda" or "cpu" explicitly.')

seed_all(1234)
set_rc_params()

# Output folders.
outputs = Path("outputs")
folder = outputs / "baseline"
networks = folder / "networks"
figures = folder / "figures"
results = folder / "results"
for path in (networks, figures, results):
    path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Parameters
#
# The SSJ reference provides the calibration and the curves we compare with.

# %%
reference_dir = outputs / "reference"
steady_state = pd.read_csv(reference_dir / "ssj_steady_state.csv").iloc[0]
reference = {
    "ss": {key: float(value) for key, value in steady_state.items()},
    "hh_c": pd.read_csv(reference_dir / "ssj_hh_policy_c.csv"),
    "hh_k": pd.read_csv(reference_dir / "ssj_hh_policy_k.csv"),
    "hh_b": pd.read_csv(reference_dir / "ssj_hh_policy_b.csv"),
    "e_grid": pd.read_csv(reference_dir / "ssj_e_grid.csv"),
    "asset_dist": pd.read_csv(reference_dir / "ssj_asset_dist.csv"),
    "irf_linear_dev": pd.read_csv(reference_dir / "ssj_irf_linear_dev.csv"),
}
ref_ss = reference["ss"]

# %%
parameters = {
    # Household and technology, from the SSJ calibration.
    "beta": ref_ss["beta"],
    "delta": ref_ss["delta"],
    "alpha": ref_ss["alpha"],
    "rho_s": ref_ss["rho_e"],
    "sigma_s": ref_ss["sd_e"] * math.sqrt(1 - ref_ss["rho_e"] ** 2),
    # Aggregate productivity shock.
    "rho_A": 0.9,
    "sigma_A": 0.01,
    # Bounds on capital.
    "k_bar": 0.0,
    "k_down": 40.0,
    # Bond: holding cost, bounds, and supply.
    "gamma": ref_ss["gamma"],
    "b_bar": 0.0,
    "b_down": 5.0,
    "B": ref_ss["B"],
}
ranges = {}  # No parameter varies during training.
shocks_ind = {"s": torch.distributions.Normal(0.0, 1.0)}
shocks_agg = {"A_scale": torch.distributions.Normal(0.0, 1.0)}

# %% [markdown]
# ## 3. Model and training settings

# %%
model = KrusellSmithTwoAsset(
    parameters=parameters,
    ranges=ranges,
    shocks_ind=shocks_ind,
    shocks_agg=shocks_agg,
    N_agents=100,  # households that evaluate the first-order conditions
    N_dist=100,    # households that represent the distribution
    hidden=128,
    layers=5,
    compile_networks=(DEVICE == "cuda"),
)

# The DSS stage tracks capital consistency but gives it no weight.
dss_config = TrainingConfig(
    iteration=10000,
    internal=20,
    batch=100,
    mc=50,
    lr=1e-4,
    adam_eps=1e-8,
    weights=ConstantWeights(weights=[1.0, 1.0, 1.0, 0.0, 1.0]),
    device=DEVICE,
    seed=1234,
)

full_config = TrainingConfig(
    iteration=20000,
    internal=20,
    batch=100,
    mc=50,
    lr=1e-4,
    adam_eps=1e-10,
    redraw_steps=0,
    weights=ConstantWeights(weights=[1.0] * model.N_RESIDUALS),
    device=DEVICE,
    seed=1234,
)

# %% [markdown]
# ## 4. Solve the deterministic steady state

# %%
dss_path = networks / "dss.pt"
full_path = networks / "full.pt"
train_full = not load_full or not full_path.exists()
train_dss = train_full and (not load_dss or not dss_path.exists())

if train_dss:
    print("Training the deterministic steady state...")
    model.train_dss_model(dss_config)
    model.save(networks, name="dss")
    # A new DSS needs a new full model, so delete the saved one.
    full_path.unlink(missing_ok=True)
elif train_full:
    print("Loading the deterministic steady state.")
    model.load(dss_path)

# %% [markdown]
# ## 5. Solve the model with aggregate risk
#
# The household network starts from the DSS weights, and a new network learns the bond price.

# %%
if train_full:
    print("Training the model with aggregate risk...")
    model.warm_start_full()
    model.train_model(full_config)
    model.save(networks, name="full")
else:
    print("Loading the model with aggregate risk.")
    model.load(full_path)

# %% [markdown]
# ### Training loss
#
# Shading marks the iterations in which the shocks are phased in.

# %%
loss_layout = dict(
    smooth=("median", 100),
    n_cols=3,
    panel_h=MULTI_SIZE[1] + 0.25,
    avg_h=DUO_SIZE[1],
)

# DSS stage: idiosyncratic shocks are phased in.
phase = {"start": 0, "end": 1000, "color": "C3", "alpha": 0.15,
         "label": "Phase: idiosyncratic shocks"}
labels = [plots.EULER_TITLES[name] for name in model.RESIDUAL_NAMES["dss"]]
fig, _ = plot_loss(
    loss_dict=model.dss_loss_dict,
    residual_names=labels,
    phases=[phase],
    **loss_layout,
)
save_figure(fig, str(figures / "dss_loss"), format_list=["pdf", "png"])

# Full model: the aggregate shock is phased in.
phase = {"start": 0, "end": 1000, "color": "C4", "alpha": 0.15,
         "label": "Phase: aggregate shocks"}
labels = [plots.EULER_TITLES[name] for name in model.RESIDUAL_NAMES["full"]]
fig, _ = plot_loss(
    loss_dict=model.loss_dict,
    residual_names=labels,
    phases=[phase],
    **loss_layout,
)
save_figure(fig, str(figures / "full_loss"), format_list=["pdf", "png"])

# %% [markdown]
# ## 6. Inspect the solution
#
# The analysis runs on a CPU in about 10 minutes. Its results are saved and reused, and
# they are recomputed after retraining.

# %%
reuse = load_analysis and not train_full

# %% [markdown]
# ### Household policies
#
# Policies of the DSS networks at the SSJ productivity grid points, for a household that
# holds the bond supply, against the SSJ policies.

# %%
path = results / "dss_hh_policy.pt"
if reuse and path.exists():
    curves = torch.load(path, weights_only=False)
else:
    curves = analysis.policy_over_state(
        model=model,
        s_values=reference["e_grid"]["e_grid_log"],
        n=100,
        batch=100,
        burn=1000,
        device=DEVICE,
    )
    torch.save(curves, path)

print(f"Steady-state productivity level: NN {curves['A_dss']:.4f}, SSJ {ref_ss['Z']:.4f}")
print(f"Steady-state bond price:         NN {curves['Q_dss']:.5f}, SSJ {ref_ss['Q']:.5f}")
fig, _ = plots.plot_hh_policy(
    curves=curves,
    reference=reference,
    labels=("NN method (DSS)", "SSJ method"),
)
save_figure(fig, str(figures / "dss_hh_policy"), format_list=["pdf", "png"])

# %% [markdown]
# Consumption over capital and bonds at the 10th and 50th percentiles of productivity.

# %%
surfaces = {}
for q in (0.1, 0.5):
    path = results / f"dss_hh_policy_surface_q{int(q * 100)}.pt"
    if reuse and path.exists():
        surfaces[q] = torch.load(path, weights_only=False)
    else:
        surfaces[q] = analysis.policy_surface(
            model=model,
            s_quantile=q,
            n_k=40,
            n_b=40,
            batch=100,
            burn=1000,
            device=DEVICE,
        )
        torch.save(surfaces[q], path)

# Both surfaces share one consumption axis.
c_low = min(surface["c"].min() for surface in surfaces.values())
c_high = max(surface["c"].max() for surface in surfaces.values())
for q, surface in surfaces.items():
    fig, _ = plots.plot_policy_surface(surface=surface, zlim=(c_low, c_high))
    save_figure(fig, str(figures / f"dss_hh_policy_surface_q{int(q * 100)}"), format_list=["pdf", "png"])

# %% [markdown]
# ### Asset distributions
#
# 500 economies simulated without aggregate shocks, against the SSJ steady state.

# %%
for which, label in (("dss", "NN method (DSS)"), ("full", "NN method")):
    path = results / f"{which}_asset_hist.pt"
    if reuse and path.exists():
        histogram = torch.load(path, weights_only=False)
    else:
        histogram = analysis.asset_histogram(
            model=model,
            which=which,
            batch=500,
            burn=500,
            amax=parameters["k_down"],
            device=DEVICE,
        )
        torch.save(histogram, path)
    share = histogram["share_above"]
    print(f"{which}: share above the bounds, capital {share['k']:.2%}, bonds {share['b']:.2%}")
    fig, _ = plots.plot_asset_distribution(
        hist=histogram,
        reference=reference,
        labels=(label, "SSJ method"),
    )
    save_figure(fig, str(figures / f"{which}_asset_distribution"), format_list=["pdf", "png"])

# %% [markdown]
# ### Equation errors after training
#
# Errors at 50 simulated periods of 10 economies, with 1,000 draws for each expectation.

# %%
for which in ("dss", "full"):
    path = results / f"{which}_euler_errors.pt"
    if reuse and path.exists():
        errors = torch.load(path, weights_only=False)
    else:
        errors = analysis.residual_accuracy(
            model=model,
            which=which,
            batch=10,
            mc=1000,
            burn=500,
            sim=50,
            device=DEVICE,
        )
        # Drop the next-period residuals, which hold every expectation draw.
        errors = {key: value for key, value in errors.items() if not key.endswith("_next")}
        torch.save(errors, path)

    means = analysis.mean_log10_errors(errors)
    print(f"{which} errors, mean log10: " + ", ".join(f"{k} {v:.2f}" for k, v in means.items()))
    fig, _ = plots.plot_euler_errors(errors=errors, names=["res_fb_k", "res_fb_b"])
    save_figure(fig, str(figures / f"{which}_euler_errors"), format_list=["pdf", "png"])

# %% [markdown]
# ## 7. A productivity impulse response
#
# A one-standard-deviation negative productivity innovation, averaged over 200 economies
# that share their household shocks with unshocked twins.

# %%
path = results / "irf.pt"
if reuse and path.exists():
    response = torch.load(path, weights_only=False)
else:
    response = analysis.irf(
        model=model,
        shock_name="A_scale",
        shock_value=-1.0,
        batch=200,
        burn=1000,
        sim=42,
        device=DEVICE,
    )
    torch.save(response, path)

fig, _ = plots.plot_irf_comparison(nn_irf=response, reference=reference, horizon=40)
save_figure(fig, str(figures / "irf"), format_list=["pdf", "png"])
