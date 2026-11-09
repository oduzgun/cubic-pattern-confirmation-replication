"""Replay the locked P03 scientific calculation without the removed pilot audit.

The simulation loop is adapted from run_power_study.py. Scientific helpers,
protocol, stream ordering, methods, CSV schemas, and error handling are retained.
This is a new verification entrypoint, not the original historical launcher.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import scipy

from ectc import CubicDesign, fitted_topology
from run_power_study import METHODS, check_preflight, evaluate_batch, exact_interval, paired_interval, predictor_for


PINNED_SCIENTIFIC_INPUTS = {
    "research/ectc.py": "d4a3d40d5100fd9797d77d41086d350f21bfeb6648afd873e4da5ac3465808e0",
    "research/run_power_study.py": "28694bdc43204d9ca1499eba74002344117768ce3a93676af1c5df3f4af45e33",
    "research/p03/protocol.json": "58262319c7651499d85dc79ae16ecd699d756b52c138b9dedf7e7bfa64669a37",
}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(output, filename, rows):
    with (output / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay_p03(output):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    hashes = {name: fingerprint(name) for name in PINNED_SCIENTIFIC_INPUTS}
    if hashes != PINNED_SCIENTIFIC_INPUTS:
        raise RuntimeError("Pinned scientific source or protocol mismatch; no draws executed")
    protocol = json.loads(Path("research/p03/protocol.json").read_text())
    assert protocol["core_sha256"] == hashes["research/ectc.py"]
    assert tuple(protocol["methods"]) == METHODS
    hashes["research/replay_studies.py"] = fingerprint(__file__)
    difference = {
        "entrypoint": "new scientific replay of P03",
        "original_entrypoint": "research/run_power_study.py",
        "omitted": "historical_provenance() pilot-audit dependency check only",
        "reason": "research/validate_baseline.py is absent from the retained lean archive",
        "preserved": "Pinned scientific sources, protocol, seed, SeedSequence stream order, scientific preflight, draws, decision helpers, paired counts, intervals, and all CSV fields",
        "claim": "Scientific computation replay; not unchanged historical-entrypoint reproduction",
    }
    (output / "execution.json").write_text(json.dumps({
        "status": "RUNNING", "started_at": started_at, "source_sha256": hashes,
        "protocol": protocol["id"], "replay_difference": difference}, indent=2) + "\n")
    keys = sorted({(case["n"], case["design"]) for case in protocol["cases"]})
    models = {key: CubicDesign(predictor_for(*key), protocol["interval"]) for key in keys}
    streams = np.random.SeedSequence(protocol["seed"]).spawn(len(protocol["cases"]) + 1)
    checks = check_preflight(np.random.default_rng(streams[0]), models)
    print(f"Preflight PASS: {checks}", flush=True)
    rates, differences, alarms = [], [], []
    for case, stream in zip(protocol["cases"], streams[1:]):
        model = models[(case["n"], case["design"])]
        truth = fitted_topology(case["derivative"])
        if (case["kind"] == "alternative") != (truth == "N"):
            raise AssertionError(f"Invalid prespecified truth: {case['id']}, {truth}")
        generator = np.random.default_rng(stream)
        count = case["replications"]
        samples = np.array(case["derivative"]) + generator.normal(size=(count, 3)) @ np.linalg.cholesky(model.covariance).T
        variances = generator.chisquare(model.degrees, count) / model.degrees
        decisions, pvalues = evaluate_batch(model, samples, variances, protocol["alpha"])
        metadata = {name: case.get(name) for name in ("id", "kind", "n", "design", "center", "halfwidth", "depth", "null_type")}
        for index, method in enumerate(METHODS):
            successes = int(decisions[:, index].sum())
            lower, upper = exact_interval(successes, count)
            null_lower, null_upper = (None, None)
            if case["kind"] == "null":
                null_lower, null_upper = exact_interval(successes, count, 0.05 / protocol["reporting"]["null_family"])
                if null_lower > protocol["alpha"]:
                    alarms.append({"case": case["id"], "method": method, "lower": null_lower})
            rates.append(dict(metadata, method=method, successes=successes, replications=count,
                              rate=successes/count, ci_lower=lower, ci_upper=upper,
                              null_family_lower=null_lower, null_family_upper=null_upper))
        if case["kind"] == "alternative":
            for index, method in enumerate(METHODS[1:], start=1):
                wins = int(np.sum(decisions[:, 0] & ~decisions[:, index]))
                losses = int(np.sum(~decisions[:, 0] & decisions[:, index]))
                lower, upper = paired_interval(wins, losses, count)
                family_lower, family_upper = paired_interval(wins, losses, count, protocol["reporting"]["simultaneous_power_family"])
                differences.append(dict(metadata, comparator=method, wins=wins, losses=losses,
                                        replications=count, difference=(wins-losses)/count,
                                        ci_lower=lower, ci_upper=upper,
                                        family_lower=family_lower, family_upper=family_upper))
        print(f"{case['id']} {case['kind']} n={case['n']} {case['design']}: " +
              ", ".join(f"{method}={decisions[:, index].mean():.3f}" for index, method in enumerate(METHODS)), flush=True)
    assert len(differences) == protocol["reporting"]["simultaneous_power_family"] == 144
    assert len(rates) == len(protocol["cases"]) * len(METHODS) == 288
    save_csv(output, "rates.csv", rates)
    save_csv(output, "paired_differences.csv", differences)
    comparisons = {}
    for method in METHODS[1:]:
        selected = [row for row in differences if row["comparator"] == method]
        comparisons[method] = {"cells": len(selected),
            "cone_higher_family_bound": sum(row["family_lower"] > 0 for row in selected),
            "cone_lower_family_bound": sum(row["family_upper"] < 0 for row in selected),
            "unresolved_family_bound": sum(row["family_lower"] <= 0 <= row["family_upper"] for row in selected),
            "range_point_difference": [min(row["difference"] for row in selected), max(row["difference"] for row in selected)]}
    after = {name: fingerprint(name) for name in hashes}
    if after != hashes:
        raise RuntimeError("Inputs changed during scientific replay")
    report = {
        "status": "COMPLETED" if not alarms else "COMPLETED; SCIENTIFIC HOLD: null calibration alarm",
        "protocol": protocol["id"], "source_sha256": hashes, "source_sha256_after": after,
        "started_at": started_at, "replay_difference": difference, "checks": checks,
        "seed": protocol["seed"], "rng": protocol["rng"], "cases_completed": len(protocol["cases"]),
        "replications": sum(case["replications"] for case in protocol["cases"]),
        "numerical_failures": 0, "null_alarms": alarms, "comparison_summary": comparisons,
        "maximum_observed_null_rate": max(row["rate"] for row in rates if row["kind"] == "null"),
        "geometries": {f"{key[0]}_{key[1]}": {"weights": model.cone.weights.tolist(),
                       "quadrature_error_estimate": model.cone.quadrature_error} for key, model in models.items()},
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {name: fingerprint(output / name) for name in ("rates.csv", "paired_differences.csv")},
        "limits": ["Seeded scientific replay, not a new study or independent numerical implementation",
                   "Historical pilot provenance preflight omitted explicitly; original launcher remains unchanged",
                   "No rigorous floating-point error enclosure or software-level size guarantee"]}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "execution.json").write_text(json.dumps({
        "status": report["status"], "source_sha256": hashes,
        "validation_sha256": fingerprint(output / "validation.json"),
        "replay_difference": difference}, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "replications", "comparison_summary", "null_alarms", "elapsed_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["p03"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    if output.is_absolute() or ".." in output.parts or not output.parts or output.parts[0] != "research":
        raise SystemExit("Use a fresh relative output directory under research/")
    output.mkdir(parents=True, exist_ok=False)
    try:
        replay_p03(output)
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"status": "FAILED", "error": repr(error)}, indent=2) + "\n")
        raise
