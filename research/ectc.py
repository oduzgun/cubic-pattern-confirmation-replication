import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad
from scipy.special import betaincc, stdtr
from scipy.stats import f


TOPOLOGIES = ("increasing", "decreasing", "maximum", "minimum", "N", "inverted_N")


def quadratic_minimum(coefficients):
    constant, linear, quadratic = np.asarray(coefficients, dtype=float)
    values = [constant - linear + quadratic, constant + linear + quadratic]
    if quadratic > 0:
        vertex = -linear / (2 * quadratic)
        if -1 < vertex < 1:
            values.append(constant + vertex * (linear + quadratic * vertex))
    return float(min(values))


def stationary_points(numerator, denominator):
    equation = 2 * numerator.deriv() * denominator - numerator * denominator.deriv()
    coefficients = equation.coef[:5]
    scale = np.max(np.abs(coefficients))
    if scale == 0:
        return np.array([-1.0, 1.0])
    roots = Polynomial(coefficients / scale).trim(tol=1e-13).roots()
    interior = [root.real for root in roots if abs(root.imag) < 1e-7 and -1 < root.real < 1]
    return np.array([-1.0, *interior, 1.0])


def variance_polynomial(covariance):
    coefficients = np.zeros(5)
    for row in range(3):
        for column in range(3):
            coefficients[row + column] += covariance[row, column]
    return Polynomial(coefficients)


class QuadraticCone:
    """Nonnegative quadratics on [-1,1], with covariance excluding sigma squared."""

    def __init__(self, covariance):
        self.covariance = np.asarray(covariance, dtype=float)
        if self.covariance.shape != (3, 3) or not np.all(np.isfinite(self.covariance)):
            raise ValueError("A finite 3 by 3 covariance is required")
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-12):
            raise ValueError("Covariance must be symmetric")
        np.linalg.cholesky(self.covariance)
        if np.linalg.cond(self.covariance) > 1e10:
            raise ValueError("Covariance is too ill-conditioned for this prototype")
        self.metric = np.linalg.inv(self.covariance)
        self.cap = np.array([1.0, 0.0, -1.0])
        self.ray_coefficients = np.array([[0., 0., 1.], [0., -2., 0.], [1., 0., 0.]])
        self.ray_norm = variance_polynomial(self.ray_coefficients.T @ self.metric @ self.ray_coefficients)
        self.faces = []
        for endpoint in (-1.0, 1.0):
            basis = np.column_stack(([1.0, -2 * endpoint, 1.0], self.cap))
            mapping = np.linalg.solve(basis.T @ self.metric @ basis, basis.T @ self.metric)
            self.faces.append((basis, mapping))
        self.weights, self.quadrature_error = self._weights()

    def _weights(self):
        primal = np.linalg.cholesky(self.metric).T
        dual = np.linalg.cholesky(self.covariance).T

        def speed(position, transformation, kind):
            if kind == "primal":
                ray = np.array([position**2, -2 * position, 1.0])
                tangent = np.array([2 * position, -2.0, 0.0])
            else:
                ray = np.array([1.0, position, position**2])
                tangent = np.array([0.0, 1.0, 2 * position])
            mapped, derivative = transformation @ ray, transformation @ tangent
            norm = np.linalg.norm(mapped)
            return np.linalg.norm(derivative - mapped * (mapped @ derivative) / norm**2) / norm

        def angle(left, right):
            cosine = left @ right / (np.linalg.norm(left) * np.linalg.norm(right))
            return np.arccos(np.clip(cosine, -1.0, 1.0))

        primal_length, primal_error = quad(speed, -1, 1, args=(primal, "primal"), epsabs=1e-10, epsrel=1e-10)
        dual_length, dual_error = quad(speed, -1, 1, args=(dual, "dual"), epsabs=1e-10, epsrel=1e-10)
        primal_length += angle(primal @ [1, 2, 1], primal @ self.cap)
        primal_length += angle(primal @ [1, -2, 1], primal @ self.cap)
        dual_length += angle(dual @ [1, -1, 1], dual @ [1, 1, 1])
        weight_two = primal_length / (4 * np.pi)
        weight_one = dual_length / (4 * np.pi)
        weights = np.array([0.5 - weight_two, weight_one, weight_two, 0.5 - weight_one])
        if np.min(weights) < -1e-10 or abs(weights.sum() - 1) > 1e-10:
            raise ArithmeticError("Invalid cone-mixture weights")
        return weights, (primal_error + dual_error) / (4 * np.pi)

    def project(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (3,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Three finite quadratic coefficients are required")
        magnitude = float(np.sqrt(coefficients @ self.metric @ coefficients))
        if magnitude == 0 or quadratic_minimum(coefficients) >= 0:
            return coefficients.copy(), 0.0
        normalized = coefficients / magnitude
        candidates = [np.zeros(3)]
        for basis, mapping in self.faces:
            amplitudes = mapping @ normalized
            if np.all(amplitudes >= 0):
                candidates.append(basis @ amplitudes)
        cap_amplitude = max(0.0, self.cap @ self.metric @ normalized / (self.cap @ self.metric @ self.cap))
        candidates.append(cap_amplitude * self.cap)
        numerator = Polynomial(self.ray_coefficients.T @ self.metric @ normalized)
        for location in stationary_points(numerator, self.ray_norm):
            ray = np.array([location**2, -2 * location, 1.0])
            amplitude = max(0.0, numerator(location) / self.ray_norm(location))
            candidates.append(amplitude * ray)
        candidates = np.asarray(candidates)
        residuals = normalized - candidates
        distances = np.einsum("ij,jk,ik->i", residuals, self.metric, residuals)
        selected = int(np.argmin(distances))
        projection = candidates[selected]
        residual_dual = self.metric @ (normalized - projection)
        moment = -residual_dual
        moment_matrix = np.array([[moment[0], moment[1]], [moment[1], moment[2]]])
        tolerance = 2e-8 * max(1.0, np.linalg.norm(moment))
        if (quadratic_minimum(projection) < -2e-8 * max(1.0, np.linalg.norm(projection)) or
                np.linalg.eigvalsh(moment_matrix).min() < -tolerance or
                moment[0] - moment[2] < -tolerance or
                abs(projection @ residual_dual) > 2e-8 * max(1.0, np.linalg.norm(projection) * np.linalg.norm(residual_dual))):
            raise ArithmeticError("Cone projection failed primal/dual/complementarity checks")
        return magnitude * projection, max(0.0, float(distances[selected] * magnitude**2))

    def pvalue(self, distance_squared, sigma2, degrees):
        if not np.isfinite(distance_squared) or distance_squared < 0:
            raise ValueError("Distance must be finite and nonnegative")
        if not np.isfinite(sigma2) or sigma2 <= 0 or not np.isfinite(degrees) or degrees <= 0:
            raise ValueError("Positive residual variance and degrees of freedom are required")
        if distance_squared == 0:
            return 1.0
        statistic = distance_squared / (distance_squared + degrees * sigma2)
        dimensions = np.array([1.0, 2.0, 3.0])
        return float(self.weights[2::-1] @ betaincc(dimensions / 2, degrees / 2, statistic))


class CubicDesign:
    """Fixed-design Gaussian cubic inference; interval and witness are prespecified."""

    def __init__(self, predictor, interval, witness=None, alpha=0.05):
        self.predictor = np.asarray(predictor, dtype=float)
        self.interval = np.asarray(interval, dtype=float)
        if self.predictor.ndim != 1 or len(self.predictor) <= 4 or not np.all(np.isfinite(self.predictor)):
            raise ValueError("More than four finite predictor observations are required")
        if self.interval.shape != (2,) or not np.all(np.isfinite(self.interval)) or self.interval[0] >= self.interval[1]:
            raise ValueError("A finite ordered interval is required")
        if not np.isfinite(alpha) or not 0 < alpha < 0.5:
            raise ValueError("alpha must lie between zero and one half")
        self.alpha = float(alpha)
        midpoint = self.interval.mean()
        halfwidth = np.diff(self.interval)[0] / 2
        witness = midpoint if witness is None else float(witness)
        if not np.isfinite(witness) or not self.interval[0] < witness < self.interval[1]:
            raise ValueError("The prespecified witness must be interior")
        normalized = (self.predictor - midpoint) / halfwidth
        self.matrix = np.vander(normalized, 4, increasing=True)
        left, singular, right = np.linalg.svd(self.matrix, full_matrices=False)
        if singular.min() <= 1e-10 * singular.max():
            raise ValueError("Rank-deficient or ill-conditioned cubic design")
        self.inverse_map = (right.T / singular) @ left.T
        full_covariance = (right.T / singular**2) @ right
        multiplier = np.diag([1.0, 2.0, 3.0])
        self.covariance = multiplier @ full_covariance[1:, 1:] @ multiplier
        self.cone = QuadraticCone(self.covariance)
        self.variance = variance_polynomial(self.covariance)
        self.locations = np.array([-1.0, 1.0, (witness - midpoint) / halfwidth])
        self.contrasts = np.vander(self.locations, 3, increasing=True)
        self.contrast_se = np.sqrt(np.einsum("ij,jk,ik->i", self.contrasts, self.covariance, self.contrasts))
        self.degrees = len(self.predictor) - 4
        self.ptcs_radius_squared = 3 * f.ppf(1 - self.alpha, 3, self.degrees)

    def fit(self, response):
        response = np.asarray(response, dtype=float)
        if response.shape != self.predictor.shape or not np.all(np.isfinite(response)):
            raise ValueError("Response must be finite and match the design")
        coefficients = self.inverse_map @ response
        residuals = response - self.matrix @ coefficients
        sigma2 = float(residuals @ residuals / self.degrees)
        if sigma2 <= np.finfo(float).eps * np.mean(response**2):
            raise ValueError("Numerically zero residual variance")
        return self.evaluate(coefficients[1:] * [1, 2, 3], sigma2)

    def evaluate(self, derivative_coefficients, sigma2):
        derivative_coefficients = np.asarray(derivative_coefficients, dtype=float)
        if derivative_coefficients.shape != (3,) or not np.all(np.isfinite(derivative_coefficients)):
            raise ValueError("Three finite derivative coefficients are required")
        if not np.isfinite(sigma2) or sigma2 <= 0:
            raise ValueError("Positive finite residual variance is required")
        derivative = Polynomial(derivative_coefficients)
        locations = stationary_points(derivative, self.variance)
        statistics = derivative(locations) / np.sqrt(sigma2 * self.variance(locations))
        contrast_statistics = self.contrasts @ derivative_coefficients / (np.sqrt(sigma2) * self.contrast_se)
        positive, negative = stdtr(self.degrees, -contrast_statistics), stdtr(self.degrees, contrast_statistics)
        _, distance_positive = self.cone.project(derivative_coefficients)
        _, distance_negative = self.cone.project(-derivative_coefficients)
        cone_positive = self.cone.pvalue(distance_positive, sigma2, self.degrees)
        cone_negative = self.cone.pvalue(distance_negative, sigma2, self.degrees)
        values = [stdtr(self.degrees, -statistics.min()), stdtr(self.degrees, statistics.max()),
                  max(positive[0], negative[1]), max(negative[0], positive[1]),
                  max(positive[0], positive[1], cone_positive),
                  max(negative[0], negative[1], cone_negative)]
        witness_values = [max(positive[0], positive[1], negative[2]),
                          max(negative[0], negative[1], positive[2])]
        ptcs_N = bool(np.all(contrast_statistics[:2] > 0) and
                      min(*contrast_statistics[:2]**2, distance_positive / sigma2) > self.ptcs_radius_squared)
        return {"pvalues": dict(zip(TOPOLOGIES, map(float, values))),
                "witness_pvalues": dict(zip(("N", "inverted_N"), map(float, witness_values))),
                "cone_pvalues": {"nonnegative": cone_positive, "nonpositive": cone_negative},
                "ptcs_N": ptcs_N,
                "bonferroni_labels": [label for label, value in zip(TOPOLOGIES, values) if value <= self.alpha / 6],
                "sigma2": float(sigma2), "df": self.degrees}


def fitted_topology(coefficients):
    coefficients = np.asarray(coefficients, dtype=float)
    lower = quadratic_minimum(coefficients)
    upper = -quadratic_minimum(-coefficients)
    left = coefficients @ [1, -1, 1]
    right = coefficients @ [1, 1, 1]
    if lower > 0:
        return "increasing"
    if upper < 0:
        return "decreasing"
    if left > 0 and right < 0:
        return "maximum"
    if left < 0 and right > 0:
        return "minimum"
    if left > 0 and right > 0 and lower < 0:
        return "N"
    if left < 0 and right < 0 and upper > 0:
        return "inverted_N"
    return "boundary"