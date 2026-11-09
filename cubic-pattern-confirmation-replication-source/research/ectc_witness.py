import numpy as np
from scipy.special import stdtr


def fit_statistics(model, responses):
    responses = np.asarray(responses, dtype=float)
    if responses.ndim != 2 or responses.shape[1] != len(model.predictor) or not np.all(np.isfinite(responses)):
        raise ValueError("A finite response matrix matching the fixed design is required")
    coefficients = responses @ model.inverse_map.T
    residuals = responses - coefficients @ model.matrix.T
    variances = np.einsum("ij,ij->i", residuals, residuals) / model.degrees
    if np.any(variances <= np.finfo(float).eps * np.mean(responses**2, axis=1)):
        raise ArithmeticError("Numerically zero residual variance")
    return coefficients[:, 1:] * np.array([1, 2, 3]), variances


def slope_statistics(model, derivatives, variances, locations):
    locations = np.asarray(locations, dtype=float)
    contrasts = np.vander(locations, 3, increasing=True)
    standard_errors = np.sqrt(np.einsum("ij,jk,ik->i", contrasts, model.covariance, contrasts))
    return derivatives @ contrasts.T / (np.sqrt(variances)[:, None] * standard_errors)


def grid_pvalues(model, derivatives, variances, grid):
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("A nonempty finite grid is required")
    if np.any(np.abs(grid) >= 1) or len(np.unique(grid)) != len(grid):
        raise ValueError("Witness grid points must be distinct and strictly interior")
    statistics = slope_statistics(model, derivatives, variances, np.r_[-1, 1, grid])
    endpoint_p = np.max(stdtr(model.degrees, -statistics[:, :2]), axis=1)
    interior_p = np.minimum(1.0, len(grid) * np.min(stdtr(model.degrees, statistics[:, 2:]), axis=1))
    return np.maximum(endpoint_p, interior_p)


def select_training_witness(training_derivatives):
    training_derivatives = np.asarray(training_derivatives, dtype=float)
    if training_derivatives.ndim != 2 or training_derivatives.shape[1] != 3 or not np.all(np.isfinite(training_derivatives)):
        raise ValueError("A finite training derivative matrix is required")
    positive_curvature = training_derivatives[:, 2] > 0
    locations = np.divide(-training_derivatives[:, 1], 2 * training_derivatives[:, 2],
                          out=np.zeros(len(training_derivatives)), where=positive_curvature)
    active = positive_curvature & (np.abs(locations) < 1)
    locations[~active] = 0.0
    return locations, active


def confirm_selected_witness(confirmation_model, confirmation_derivatives, confirmation_variances, locations, active):
    locations = np.asarray(locations, dtype=float)
    active = np.asarray(active, dtype=bool)
    count = len(confirmation_derivatives)
    if locations.shape != (count,) or active.shape != (count,) or np.any(~np.isfinite(locations)):
        raise ValueError("Selected witness arrays must match confirmation draws")
    if np.any(np.abs(locations[active]) >= 1):
        raise ValueError("Active witnesses must be strictly interior")
    contrasts = np.column_stack((np.ones(count), locations, locations**2))
    standard_errors = np.sqrt(confirmation_variances * np.einsum("ij,jk,ik->i", contrasts, confirmation_model.covariance, contrasts))
    statistics = np.einsum("ij,ij->i", confirmation_derivatives, contrasts) / standard_errors
    endpoints = slope_statistics(confirmation_model, confirmation_derivatives, confirmation_variances, [-1, 1])
    endpoint_p = np.max(stdtr(confirmation_model.degrees, -endpoints), axis=1)
    values = np.maximum(endpoint_p, stdtr(confirmation_model.degrees, statistics))
    values[~active] = 1.0
    return values