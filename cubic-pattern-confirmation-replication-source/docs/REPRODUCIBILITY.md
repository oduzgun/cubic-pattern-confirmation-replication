# Reproduction and statistical interpretation

The finite design uses a fixed Gaussian cubic model and prespecified strict `+,-,+` confirmation targets. `research/p03/protocol.json` and `research/p04/protocol.json` are the freeze-defining specifications. Each has 72 cells and 96,000 seeded replications. The source and protocol hashes are pinned in `MANIFEST.json` and checked by `scripts/verify_release.py`.

Execute the two scripts in the README and compare against the five frozen CSV files using the verifier. The comparison permits relative/absolute differences of `1e-11` in numeric cells; it is not a bitwise proof or a replacement for independent methodological review. A failed or unavailable numerical p-value must never be interpreted as a non-rejection.

The archived implementation is a carefully curated scientific replay variant; administrative audit routines and decision records are not distributed. The full simulation was successfully rerun from this variant in GitHub Actions on 2026-11-09 before this public-source packaging step. No causal/empirical data or third-party private records are involved.
