import json
from pathlib import Path

import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam


def run_training(seed=0):
    notebook = json.loads((Path(__file__).parents[1] / "Bayesian_regression_model.ipynb").read_text())
    ns = dict(torch=torch, pyro=pyro, dist=dist, Predictive=Predictive, SVI=SVI, Trace_ELBO=Trace_ELBO, Adam=Adam)
    for i in (9, 11, 13, 15, 17, 19, 21, 23, 25, 27):
        exec("".join(notebook["cells"][i]["source"]).replace("pyro.set_rng_seed(0)", f"pyro.set_rng_seed({seed})"), ns)
    return ns


def test_notebook_resets_param_store_for_repeatable_recovery():
    pyro.clear_param_store()
    pyro.param("stale_parameter", torch.tensor(8.0))
    first = run_training()
    assert "stale_parameter" not in pyro.get_param_store()
    second = run_training()
    for name in ("slope_mean", "intercept_mean", "sigma_mean"):
        assert first[name].item() == second[name].item()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_parameter_recovery(seed):
    pyro.clear_param_store()
    ns = run_training(seed)
    assert ns["slope_mean"].item() == pytest.approx(2.0, abs=0.15)
    assert ns["intercept_mean"].item() == pytest.approx(1.0, abs=0.5)
    assert 0.6 < ns["sigma_mean"].item() < 1.5


def test_held_out_posterior_predictive_coverage_and_ols_sanity():
    ns = run_training()
    design = torch.stack([torch.ones_like(ns["X"]), ns["X"]], dim=1)
    ols = torch.linalg.lstsq(design, ns["Y"]).solution
    assert abs(ns["slope_mean"].item() - ols[1].item()) < 0.15
    assert abs(ns["intercept_mean"].item() - ols[0].item()) < 0.5
    pyro.set_rng_seed(113)
    x = torch.linspace(-2, 12, 50)
    y = 1 + 2 * x + torch.randn(50)
    samples = Predictive(ns["model"], guide=ns["guide"], num_samples=1000)(x, None)["obs"]
    low, high = torch.quantile(samples, torch.tensor([0.025, 0.975]), dim=0)
    coverage = ((y >= low) & (y <= high)).float().mean().item()
    assert 0.85 <= coverage <= 1.0
