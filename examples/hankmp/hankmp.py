# %% [markdown]
# # One-asset HANK model with monetary-policy shocks
#
# Households differ in productivity and bond holdings, firms set prices subject to
# adjustment costs, and monetary policy follows a Taylor rule with persistent shocks. We
# first solve the deterministic steady state (DSS), then the model with aggregate risk, and
# compare both with the sequence-space Jacobian (SSJ) reference. The website walks through
# the model and the code.

# %% [markdown]
# ## 1. Set up

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from estimating_hank_nn.helpers import seed_all
from estimating_hank_nn.plot_style import set_rc_params, save_figure, MULTI_SIZE
from estimating_hank_nn.plots import plot_loss, plot_avg_loss

from hankmp_model import OneAssetHANK, TrainingConfig
import hankmp_analysis as analysis
import hankmp_plots as plots

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
# The SSJ reference provides the curves we compare with. The household parameters are
# those of its steady-state solve.

# %%
reference_dir = outputs / "reference"
steady_state = pd.read_csv(reference_dir / "ssj_steady_state.csv").iloc[0]
reference = {
    "ss": {key: float(value) for key, value in steady_state.items()},
    "hh_n": pd.read_csv(reference_dir / "ssj_hh_policy_n.csv"),
    "hh_c": pd.read_csv(reference_dir / "ssj_hh_policy_c.csv"),
    "e_grid": pd.read_csv(reference_dir / "ssj_e_grid.csv"),
    "asset_dist": pd.read_csv(reference_dir / "ssj_asset_dist.csv"),
    "irf_linear_dev": pd.read_csv(reference_dir / "ssj_irf_linear_dev_0.0025.csv"),
    "irf_nonlinear_dev": pd.read_csv(reference_dir / "ssj_irf_nonlinear_dev_0.0025.csv"),
}

# %%
parameters = {
    # Households, from the SSJ steady-state solve.
    "beta": 0.9822435537836073,
    "gamma": 2.0,
    "eta": 2.0,
    "chi": 0.7864334221642553,
    "rho_s": 0.966,
    "sigma_s": 0.12927103310486848,
    "B": 5.6,
    "phipi": 1.5,
    # Borrowing limit and numerical upper asset bound.
    "b_min": -0.0,
    "b_max": 125.0,
    # Reference nominal rate for initialization. The DSS network learns R_DSS.
    "Rn_ss": 1.005,
    # Dormant TFP process.
    "rho_z": 0.0,
    "sigma_z": 0.0,
    # Monetary-policy shock.
    "rho_mps": 0.6,
    "sigma_mps": 0.0025,
    # Firms.
    "varphi": 60.0,
    "theta": 6.0,
}
ranges = {"rho_mps": torch.distributions.Uniform(0.5, 0.7)}  # Varies during training.
shocks_ind = {"s": torch.distributions.Normal(0.0, 1.0)}
shocks_agg = {
    "z": torch.distributions.Normal(0.0, 1.0),
    "mps": torch.distributions.Normal(0.0, 1.0),
}

# %% [markdown]
# ## 3. Model and training settings

# %%
model = OneAssetHANK(
    parameters=parameters,
    ranges=ranges,
    shocks_ind=shocks_ind,
    shocks_agg=shocks_agg,
    N_agents=100,  # households that evaluate the Euler equation
    N_dist=100,    # households that represent the distribution
    hidden=128,
    layers=5,
    compile_networks=(DEVICE == "cuda"),
)

# Both stages use the same settings. Aggregate residuals are divided by their scale.
config = TrainingConfig(
    iteration=20000,
    internal=10,
    steps=5,
    batch=50,
    mc=100,
    lr=1e-4,
    eta_min=1e-8,
    adjusted_residuals=True,
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
    model.train_dss_model(config)
    model.save(networks, name="dss")
    # A new DSS needs a new full model, so delete the saved one.
    full_path.unlink(missing_ok=True)
elif train_full:
    print("Loading the deterministic steady state.")
    model.load(dss_path)

# %% [markdown]
# ## 5. Solve the model with aggregate risk
#
# The household network starts from the DSS weights, and an aggregate network learns
# inflation and the wage.

# %%
if train_full:
    print("Training the model with aggregate risk...")
    model.warm_start_full()
    model.train_model(config)
    model.save(networks, name="full")
else:
    print("Loading the model with aggregate risk.")
    model.load(full_path)

# %% [markdown]
# ### Training loss
#
# Shading marks the iterations in which the shocks are phased in.

# %%
warmup = 1000  # 5% of 20,000 iterations
phase_ind = {"start": 0, "end": warmup, "color": "C3", "alpha": 0.15,
             "label": "Phase 1: idiosyncratic shocks"}
phase_agg = {"start": warmup, "end": 2 * warmup, "color": "C4", "alpha": 0.15,
             "label": "Phase 2: aggregate shocks"}

labels = [plots.RESIDUAL_LABELS[name] for name in model.RESIDUAL_NAMES["dss"]]
fig, _ = plot_loss(
    loss_dict=model.dss_loss_dict,
    residual_names=labels,
    phases=[phase_ind],
    smooth=("median", 100),
)
save_figure(fig, str(figures / "dss_loss"), format_list=["pdf", "png"])

labels = [plots.RESIDUAL_LABELS[name] for name in model.RESIDUAL_NAMES["full"]]
fig, _ = plot_loss(
    loss_dict=model.loss_dict,
    residual_names=labels,
    phases=[phase_ind, phase_agg],
    smooth=("median", 100),
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
# ### Steady-state values
#
# The steady-state interest rate and output do not depend on $\rho_{mp}$, so they should be
# flat across its training range and lie on the SSJ steady state.

# %%
sweep = analysis.policy_over_par(model=model, n=100, device=DEVICE)
fig, _ = plots.plot_dss_values(
    sweep=sweep,
    true_values={"Rn_dss": reference["ss"]["Rr"], "Y_dss": reference["ss"]["Y"]},
    par_labels={"rho_mps": r"$\rho_{mp}$"},
)
save_figure(fig, str(figures / "dss_values"), format_list=["pdf", "png"])

# %% [markdown]
# ### Household policies
#
# Labor and consumption at the SSJ productivity grid points, from the DSS networks and
# from the full model, against the SSJ policies.

# %%
for which, label in (("dss", "NN method (DSS)"), ("full", "NN method")):
    path = results / f"{which}_hh_policy.pt"
    if reuse and path.exists():
        curves = torch.load(path, weights_only=False)
    else:
        curves = analysis.policy_over_state(
            model=model,
            s_values=reference["e_grid"]["e_grid_log"],
            which=which,
            n=100,
            batch=100,
            burn=1000,
            device=DEVICE,
        )
        torch.save(curves, path)

    fig, _ = plots.plot_hh_policy(
        curves=curves,
        reference=reference,
        labels=(label, "SSJ method"),
    )
    save_figure(fig, str(figures / f"{which}_hh_policy"), format_list=["pdf", "png"])

# %% [markdown]
# The same policies at the 10th, 50th, and 90th percentiles of productivity, with the
# assets held by these households shaded behind.

# %%
path = results / "dss_hh_policy_percentiles.pt"
if reuse and path.exists():
    curves, densities = torch.load(path, weights_only=False)
else:
    by_income = analysis.policy_by_income(
        model=model,
        which="dss",
        percentiles=(0.1, 0.5, 0.9),
        n=100,
        batch=200,
        burn=1000,
        device=DEVICE,
    )
    # One column per percentile, as in the replication's saved results.
    curves = pd.DataFrame({"b_grid": by_income["b_grid"]})
    densities = pd.DataFrame({"center": by_income["centers"], "width": np.diff(by_income["edges"])})
    for pct, group in zip(("p10", "p50", "p90"), by_income["groups"]):
        curves[f"n_{pct}"] = group["n"]
        curves[f"c_{pct}"] = group["c"]
        densities[f"d_{pct}"] = group["density"]
    torch.save((curves, densities), path)

fig, _ = plots.plot_hh_policy_ladder(runs={"NN method (DSS)": (curves, densities)})
save_figure(fig, str(figures / "dss_hh_policy_histogram"), format_list=["pdf", "png"])

# %% [markdown]
# ### Asset distribution
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
            n_bins=50,
            device=DEVICE,
        )
        torch.save(histogram, path)

    below_zero = histogram["tail_mass"]["below_zero"]
    print(f"{which}: asset mass below zero before clamping {below_zero:.4f}")
    fig, _ = plots.plot_asset_distribution(
        hist=histogram,
        reference=reference,
        labels=(label, "SSJ method"),
    )
    save_figure(fig, str(figures / f"{which}_asset_distribution"), format_list=["pdf", "png"])

# %% [markdown]
# ### Equation errors after training
#
# Errors at 100 simulated states of 10 economies, with 1,000 draws for each expectation.
# The aggregate residuals are scaled as in training.

# %%
path = results / "full_euler_errors.pt"
if reuse and path.exists():
    errors = torch.load(path, weights_only=False)
else:
    errors = analysis.residual_accuracy(
        model=model,
        which="full",
        batch=10,
        mc=1000,
        burn=200,
        draws=100,
        adjusted_residuals=True,
        device=DEVICE,
    )
    # Drop the next-period residuals, which hold every expectation draw.
    errors = {name: value for name, value in errors.items() if not name.endswith("_next")}
    torch.save(errors, path)

fig, _ = plots.plot_euler_errors(hist=plots.euler_error_histograms(errors))
save_figure(fig, str(figures / "full_euler_errors"), format_list=["pdf", "png"])

# %% [markdown]
# ## 7. A monetary-policy impulse response
#
# A one-standard-deviation monetary easing at $\rho_{mp}=0.6$, averaged over 100 economies
# that share their household shocks with unshocked twins. The SSJ lines are the linear and
# nonlinear responses to the same shock path.

# %%
path = results / "irf.pt"
if reuse and path.exists():
    nn_irf = torch.load(path, weights_only=False)
else:
    responses = analysis.irf(
        model=model,
        shock_name="mps",
        shock_value=-1.0,
        rho_mps=0.6,
        batch=100,
        burn=1000,
        sim=40,
        method="crn",
        device=DEVICE,
    )
    # The figure plots the mean over the economies, so save only that.
    nn_irf = {name: values.mean(dim=1, keepdim=True) for name, values in responses.items()}
    torch.save(nn_irf, path)

fig, _ = plots.plot_irf_comparison(
    nn_irf=nn_irf,
    reference=reference,
    mapping=plots.IRF_MAPPING_PAPER,
    horizon=20,
    n_cols=3,
    panel_size=(MULTI_SIZE[0], MULTI_SIZE[1] + 0.5),
)
save_figure(fig, str(figures / "irf"), format_list=["pdf", "png"])

# %% [markdown]
# ## 8. Robustness
#
# The replication package solves the model again with more households, more Monte Carlo
# draws, smaller shocks, and a ten times larger shock. Those networks are not shipped, but
# their saved results in `outputs/robustness` are enough to draw the comparisons.

# %%
robustness = outputs / "robustness"
comparisons = robustness / "figures"
comparisons.mkdir(parents=True, exist_ok=True)


def ladder(runs):
    """Asset histograms and percentile policies of several runs, keyed by legend label."""
    hists = {}
    policies = {}
    for label, tag in runs.items():
        hists[label] = pd.read_csv(robustness / tag / "asset_hist_full.csv")
        policies[label] = (
            pd.read_csv(robustness / tag / "hh_policy_income_full.csv"),
            pd.read_csv(robustness / tag / "hh_policy_income_density_full.csv"),
        )
    return hists, policies


# %% [markdown]
# ### Number of households
#
# Tracking 200, 500, or 1,000 households instead of 100 barely changes the asset
# distribution or the policies, while the run time on the GPU grows from 1.5 to 6 hours.

# %%
hists, policies = ladder({
    "100 agents": "baseline",
    "200 agents": "D200",
    "500 agents": "D500",
    "1,000 agents": "D1000",
})
fig, _ = plots.plot_asset_distribution_ladder(hists=hists)
save_figure(fig, str(comparisons / "asset_distribution_agents"), format_list=["pdf", "png"])
fig, _ = plots.plot_hh_policy_ladder(runs={k: policies[k] for k in ("100 agents", "1,000 agents")})
save_figure(fig, str(comparisons / "hh_policy_agents"), format_list=["pdf", "png"])

# %% [markdown]
# ### Number of Monte Carlo draws
#
# With 250 or 500 draws per expectation instead of 100, the results are nearly identical.

# %%
hists, policies = ladder({
    "100 MC draws": "baseline",
    "250 MC draws": "MC250",
    "500 MC draws": "MC500",
})
fig, _ = plots.plot_asset_distribution_ladder(hists=hists)
save_figure(fig, str(comparisons / "asset_distribution_mc"), format_list=["pdf", "png"])
fig, _ = plots.plot_hh_policy_ladder(runs={k: policies[k] for k in ("100 MC draws", "500 MC draws")})
save_figure(fig, str(comparisons / "hh_policy_mc"), format_list=["pdf", "png"])

# %% [markdown]
# ### Smaller shocks
#
# Networks trained with progressively smaller $\sigma_{mp}$, with each response rescaled
# to the baseline shock size.

# %%
SIGMA = 0.0025
PANELS = {"Rr": "Interest rate", "Y": "Output", "C": "Consumption"}


def nn_response(tag, sigma, label, color, scale=1.0, ls="-", lw=None):
    """A saved NN response of one run, with its stochastic steady state if available."""
    curve = {
        "label": label,
        "color": color,
        "ls": ls,
        "scale": scale,
        "dev": pd.read_csv(robustness / tag / f"nn_irf_dev_{sigma:g}.csv", index_col="h"),
    }
    sss = robustness / tag / f"nn_sss_{sigma:g}.csv"
    curve["ss"] = pd.read_csv(sss).iloc[0].to_dict() if sss.exists() else None
    if lw is not None:
        curve["lw"] = lw
    return curve


def ssj_response(sigma, label):
    """The SSJ linear response to a shock of size sigma, with the SSJ steady state."""
    dev = pd.read_csv(reference_dir / f"ssj_irf_linear_dev_{sigma:g}.csv")
    dev = dev.rename(columns={"w": "W"})
    ss = {("W" if key == "w" else key): value for key, value in reference["ss"].items()}
    return {"label": label, "color": "black", "ls": "--", "scale": 1.0, "dev": dev, "ss": ss}


# Run folder, shock size, line color, and line style of each variant.
variants = [
    ("baseline", "0.0025", "C0", "-"),
    ("sigma_0.00125", "0.00125", "C4", "--"),
    ("sigma_0.000625", "0.000625", "C3", "-."),
    ("sigma_0.0001", "0.0001", "C1", "-"),
    ("sigma_0.00001", "0.00001", "C5", "-"),
]
nn = {}
for tag, sigma, color, ls in variants:
    nn[tag] = nn_response(
        tag=tag,
        sigma=float(sigma),
        label=rf"NN ($\sigma_{{mp}} = {sigma}$)",
        color=color,
        scale=SIGMA / float(sigma),
        ls=ls,
        lw=2.5 if tag == "baseline" else None,
    )
ssj = ssj_response(sigma=SIGMA, label="SSJ linear")

# Paint the dashed variants above the thicker baseline and the SSJ line on top.
order = [nn["sigma_0.00001"], nn["sigma_0.0001"], nn["baseline"],
         nn["sigma_0.000625"], nn["sigma_0.00125"], ssj]
fig, _ = plots.plot_irf_curves(
    curves=[ssj] + list(nn.values()),
    panels=PANELS,
    ylabel="Deviation (scaled)",
    panel_size=(8 / 3, 2.8),
    draw_order=order,
)
save_figure(fig, str(comparisons / "irf_variants_deviation"), format_list=["pdf", "png"])

# %% [markdown]
# ### Large aggregate uncertainty
#
# A ten times larger shock ($\sigma_{mp}=0.025$) makes aggregate risk matter for household
# behavior, and the responses become visibly nonlinear. This run uses its own training recipe.

# %%
big = robustness / "big"

# Average training loss.
loss = pd.read_csv(big / "loss_full.csv")
phases = [
    {"start": 0, "end": 1000, "color": "C3", "alpha": 0.15, "label": "Phase 1: idiosyncratic shocks"},
    {"start": 1000, "end": 2000, "color": "C4", "alpha": 0.15, "label": "Phase 2: aggregate shocks"},
]
fig, _ = plot_avg_loss(
    loss_dict={"iteration": loss["iteration"], "res_avg": loss["avg_loss"]},
    phases=phases,
    smooth=("median", 100),
)
save_figure(fig, str(comparisons / "big_full_avg_loss"), format_list=["pdf", "png"])

# Asset distribution, from the saved bin centers, widths, densities, and cumulative shares.
frame = pd.read_csv(big / "asset_hist_full.csv")
right = (frame["center"] + frame["width"] / 2).to_numpy()
histogram = {
    "centers": frame["center"].to_numpy(),
    "edges": np.append(right[0] - frame["width"][0], right),
    "density": frame["density"].to_numpy(),
    "cdf": np.append(0.0, frame["cdf_right"]),
}
fig, _ = plots.plot_asset_distribution(hist=histogram, reference=reference)
save_figure(fig, str(comparisons / "big_full_asset_distribution"), format_list=["pdf", "png"])

# Household policies, one column per productivity state.
frame = pd.read_csv(big / "hh_policy_full.csv")
n_states = sum(column.startswith("n_e") for column in frame.columns)
curves_big = {
    "b_grid": frame["b_grid"].to_numpy(),
    "n": np.stack([frame[f"n_e{i}"] for i in range(n_states)]),
    "c": np.stack([frame[f"c_e{i}"] for i in range(n_states)]),
}
fig, _ = plots.plot_hh_policy(curves=curves_big, reference=reference)
save_figure(fig, str(comparisons / "big_full_hh_policy"), format_list=["pdf", "png"])

# Impulse responses in levels, against the SSJ linear response to the same shock.
fig, _ = plots.plot_irf_curves(
    curves=[ssj_response(sigma=0.025, label="SSJ linear, big shock"),
            nn_response(tag="big", sigma=0.025, label="NN method, big shock", color="C1")],
    panels=PANELS,
    levels=True,
    legend_cols=2,
    panel_size=(8 / 3, 2.8),
)
save_figure(fig, str(comparisons / "irf_big_shock_levels"), format_list=["pdf", "png"])
