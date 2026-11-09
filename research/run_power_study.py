import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.special import betaincc, stdtr
from scipy.stats import beta, binomtest

from ectc import CubicDesign, fitted_topology


ROOT = Path("research/p03")
METHODS = ("cone", "witness_0", "witness_0_5", "PTCS")


def exact_interval(successes, count, error=0.05):
    lower = 0.0 if successes == 0 else beta.ppf(error / 2, successes, count - successes + 1)
    upper = 1.0 if successes == count else beta.ppf(1 - error / 2, successes + 1, count - successes)
    return float(lower), float(upper)


def paired_interval(wins, losses, count, family=1):
    win_interval = exact_interval(wins, count, 0.05 / (2 * family))
    loss_interval = exact_interval(losses, count, 0.05 / (2 * family))
    return max(-1.0, win_interval[0] - loss_interval[1]), min(1.0, win_interval[1] - loss_interval[0])


def predictor_for(size, design):
    if design == "uniform":
        return np.linspace(-1, 1, size)
    if design == "right_weighted":
        return 2 * np.sqrt(np.linspace(0, 1, size)) - 1
    raise ValueError("Unknown prespecified design")


def evaluate_batch(model, samples, variances, alpha):
    contrasts = np.array([[1, -1, 1], [1, 1, 1], [1, 0, 0], [1, 0.5, 0.25]])
    standard_errors = np.sqrt(np.einsum("ij,jk,ik->i", contrasts, model.covariance, contrasts))
    statistics = (samples @ contrasts.T) / (np.sqrt(variances)[:, None] * standard_errors)
    endpoint_p = np.max(stdtr(model.degrees, -statistics[:, :2]), axis=1)
    distances = np.array([model.cone.project(sample)[1] for sample in samples])
    fraction = distances / (distances + model.degrees * variances)
    dimensions = np.array([1.0, 2.0, 3.0])
    cone_p = betaincc(dimensions[None, :] / 2, model.degrees / 2, fraction[:, None]) @ model.cone.weights[2::-1]
    cone_p[distances == 0] = 1.0
    pvalues = np.column_stack((np.maximum(endpoint_p, cone_p),
                              np.maximum(endpoint_p, stdtr(model.degrees, statistics[:, 2])),
                              np.maximum(endpoint_p, stdtr(model.degrees, statistics[:, 3]))))
    ptcs = np.all(statistics[:, :2] > 0, axis=1) & (
        np.minimum(np.min(statistics[:, :2]**2, axis=1), distances / variances) > model.ptcs_radius_squared)
    decisions = np.column_stack((pvalues <= alpha, ptcs))
    if not np.all(np.isfinite(pvalues)) or np.any(pvalues < 0) or np.any(pvalues > 1):
        raise ArithmeticError("Invalid batch p-value")
    if np.any(ptcs & ~decisions[:, 0]):
        raise ArithmeticError("PTCS result violates the expected cone-test containment")
    return decisions, pvalues


def check_preflight(generator, models):
    checked = 0
    for key, model in models.items():
        second_witness = CubicDesign(model.predictor, [-1, 1], witness=0.5)
        samples = generator.normal(size=(64, 3)) @ np.linalg.cholesky(model.covariance).T
        samples += generator.normal(size=(64, 3))
        variances = generator.chisquare(model.degrees, 64) / model.degrees
        decisions, pvalues = evaluate_batch(model, samples, variances, model.alpha)
        for index, (sample, variance) in enumerate(zip(samples, variances)):
            result = model.evaluate(sample, variance)
            alternate = second_witness.evaluate(sample, variance)
            expected = [result['pvalues']['N'], result['witness_pvalues']['N'], alternate['witness_pvalues']['N']]
            if not np.allclose(pvalues[index], expected, rtol=0, atol=1e-11):
                raise AssertionError(f'Batch/core mismatch: {key}, draw {index}')
            if bool(decisions[index, 3]) != result['ptcs_N']:
                raise AssertionError('PTCS batch/core mismatch')
            checked += 1
    for count in (10, 100, 1000):
        for successes in (0, 1, count // 2, count - 1, count):
            expected = binomtest(successes, count).proportion_ci(method='exact')
            assert np.allclose(exact_interval(successes, count), [expected.low, expected.high], atol=1e-10)
            paired = paired_interval(successes, count - successes, count, family=144)
            estimate = (2 * successes - count) / count
            assert paired[0] <= estimate <= paired[1]
    return {'batch_core_agreement_draws': checked, 'interval_checks': 'PASS', 'designs_checked': len(models)}
