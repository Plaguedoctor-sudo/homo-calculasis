import numpy as np
import pytest
from inference.design import select_optimal_clamp
from inference.hierarchical import fit_hierarchical_eb
from inference.npe_train import train_amortized_npe
from inference.simulator import jacobian_rank, simulate_batch
from params.space import ParameterSpace
from validation.ppc import posterior_predictive_check
from validation.prospective import evaluate_prospective_trial, predict_arm_probabilities
from validation.recovery import evaluate_recovery

@pytest.fixture(scope="module")
def space():
    return ParameterSpace()

@pytest.fixture(scope="module")
def trained_pipeline(space):
    train_thetas = space.sample_prior(1200, seed=10)
    train_xs = simulate_batch(train_thetas, space, noise_sd=0.015, seed=11)
    npe = train_amortized_npe(train_thetas, train_xs, space)
    return npe

def test_full_jacobian_identifiability_with_clamps(space):
    rank_all, cond_all = jacobian_rank(space, active_clamps=None)
    rank_unclamped, _ = jacobian_rank(space, active_clamps=("trace_panel",))
    assert rank_all == space.dim == 16
    assert cond_all < 50.0
    assert rank_unclamped < rank_all, "Clamps must increase identifiable subspace rank"

@pytest.mark.gono
def test_tier1_synthetic_parameter_recovery(space, trained_pipeline):
    test_thetas = space.sample_prior(200, seed=22)
    test_xs = simulate_batch(test_thetas, space, noise_sd=0.015, seed=23)
    report = evaluate_recovery(trained_pipeline, test_thetas, test_xs)
    print(f"\nT1 mean R^2 across 16 params = {report.mean_r2:.3f}")
    report.assert_gate(min_mean_r2=0.75)
    assert all(c > 0.5 for c in report.contraction_per_param.values())

def test_tier2_posterior_predictive_check(space, trained_pipeline):
    patient_theta = space.sample_prior(1, seed=42)[0]
    x_obs = simulate_batch(patient_theta[None, :], space, noise_sd=0.01, seed=43)[0]
    ppc = posterior_predictive_check(trained_pipeline, x_obs, n_draws=150, seed=44)
    assert ppc["nrmse"] < 0.25
    assert ppc["coverage_95"] >= 0.85

def test_hierarchical_subtype_eigenstructure(space, trained_pipeline):
    # Generate 2 distinct MDD biotypes: Biotype 1 (inflammatory/IDO) vs Biotype 2 (anhedonic/low-Fe)
    b1 = space.sample_prior(100, seed=50)
    b1[:, space.names.index("ido_gain")] = 2.6
    b2 = space.sample_prior(100, seed=51)
    b2[:, space.names.index("reward_rho")] = 0.35
    b2[:, space.names.index("fe_intake")] = 0.25
    cohort_thetas = np.vstack([b1, b2])

    cohort_xs = simulate_batch(cohort_thetas, space, noise_sd=0.015, seed=52)
    pat_means, post_cov = trained_pipeline.predict_posterior(cohort_xs)
    hier = fit_hierarchical_eb(pat_means, post_cov)
    assert hier.effective_rank > 1.5, "Kill criterion 2 triggered: Sigma collapsed to rank 1"

def test_active_experiment_design_reduces_entropy(space):
    best_clamp, gains = select_optimal_clamp(space, already_applied=("trace_panel",), n_sim=250, seed=60)
    assert best_clamp in gains
    assert gains[best_clamp] > 0.5, "Optimal clamp failed to reduce posterior entropy"

@pytest.mark.gono
def test_tier3_and_tier4_prospective_drug_vs_placebo(space, trained_pipeline):
    cohort_true = space.sample_prior(300, seed=70)
    cohort_xs = simulate_batch(cohort_true, space, noise_sd=0.015, seed=71)
    cohort_inf, _ = trained_pipeline.predict_posterior(cohort_xs)

    rep = evaluate_prospective_trial(cohort_true, cohort_inf, space, seed=72)
    print(
        f"\nT3/T4 Prospective: AUC={rep.auc_sertraline:.3f}, "
        f"Slope={rep.calibration_slope:.3f}, Divergence={rep.drug_placebo_divergence:.3f}"
    )
    rep.assert_clinical_gates()

    # Verify rTMS outperforms SSRI in severe iron/cofactor synthesis deficiency
    cof_depleted = space.mean.copy()
    cof_depleted[space.names.index("fe_intake")] = 0.12
    cof_depleted[space.names.index("se_intake")] = 0.12
    cof_depleted[space.names.index("eaat2_capacity")] = 1.8
    p_ssri = predict_arm_probabilities(cof_depleted[None, :], space, arm="sertraline")[0]
    p_rtms = predict_arm_probabilities(cof_depleted[None, :], space, arm="rtms")[0]
    assert p_rtms > p_ssri
