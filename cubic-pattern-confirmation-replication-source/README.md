# Cubic Pattern Confirmation — Replication Code

**Research status:** Associated methodological manuscript, *Finite-Sample Confirmation of an Increase–Decrease–Increase Pattern in Cubic Regression*, remains unpublished. This repository does not imply journal acceptance or an assigned DOI.

This is a self-contained **software and numerical replication** of the P03/P04 finite-sample simulation studies for correctly specified fixed-design Gaussian cubic regression. It tests the strict `+,-,+` pattern on a prespecified interval, comparing the complete-pattern cone IUT, fixed witnesses, multiplicity-adjusted witness grids, an independently selected split-sample witness and coefficient-ellipsoid containment.

## Reproduce

Requires Python 3.12, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python research/replay_studies.py --study p03 --output-dir research/p03_rerun
python research/run_witness_study.py --output-dir research/p04_rerun
python scripts/verify_release.py --compare-rerun
```

P03 and P04 each have **96,000** seeded simulation replications; their five frozen CSV tables total **1,080** data rows. The complete re-execution passed on 9 November 2026 against the frozen results, as did the separately curated run. GitHub Actions runs the same checks on each push and pull request. The resulting `research/p03_rerun/` and `research/p04_rerun/` directories are ignored by Git.

## Files

- `research/ectc.py`, `research/ectc_witness.py`: statistical algorithms
- `research/run_power_study.py`, `research/run_witness_study.py`: computational helpers and P04 study
- `research/replay_studies.py`: frozen P03 scientific replay entry point
- `research/p03/`, `research/p04/`: design protocols and archived numerical results
- `scripts/verify_release.py`: source integrity and regenerated-output comparison
- `MANIFEST.json`: SHA-256 file manifest and scientific source fingerprints

## Scope and limitations

These simulations are **numerical evidence**, not a universal power ordering, a new proof of polynomial positivity, or a rigorous floating-point enclosure. The 19-point witness grid includes all simulated alternative centers; the simulated null family excludes the reflected left-endpoint boundary under the right-weighted design. Four separately recorded numerical validation requests in the broader research archive were refused; a refusal is not evidence of non-rejection.

The distributed P03/P04 source is a **scientifically validated curated variant** of an internal research archive: administrative histories were omitted and computational sources/protocols re-pinned. The scientific CSV results were unchanged and were reproduced after another 192,000 Monte Carlo draws. The unavailable internal archive is not required to execute this repository.

## Citation and license

See `CITATION.cff` for author/manuscript attribution and `LICENSE` for the MIT license governing original code. The manuscript text is not distributed. An archival DOI will only be added when actually assigned.
