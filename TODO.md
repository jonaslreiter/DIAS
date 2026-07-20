# TODO

Follow-up items intentionally deferred from the initial public release.

## Reproducibility surface

- **Done — aggregation across (domain, seed).** `scripts/collect_results.py`
  reads the per-run JSONs under `logs/validation/<experiment>/<domain>/seedN.json`
  (or the W&B group) and prints the paper table (P, R, N_S, N_C, N_SC over seeds).
- **Done — sweep launcher + MLP/noise experiments.** `scripts/launch.py` submits
  a training array plus a dependent validation array per domain, with
  `--experiment-type {dias,mlp,noise}`. The MLP baseline trains via
  `state_dependent_effects`; noise uses the `expected_count` model.
- **L1 baseline sweep.** `launch.py` covers checkpoint-based experiments; L1
  validates an externally-produced symbolic domain and is still run per-instance
  via `scripts/validation/run_l1.py`. A thin L1 sweep orchestrator is not wired up.

## Code organization

- Re-examine the model file (`src/dias/models/dias.py`) for ablation branches that
  are no longer exercised by the shipped configs and can be removed.

## Docs

- Smoke test (10-step single-domain training) wired into CI.
