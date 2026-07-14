# TODO

Follow-up items intentionally deferred from the initial public release.

## Reproducibility surface

- **Aggregation across (domain, seed).** `scripts/run_all_domains.py`
  produces per-pair JSONs under `logs/validation/<group>/<domain>/seedN.json`
  and per-run W&B entries grouped by `--group`. The paper-table metrics
  (P, R, N_S, N_C, N_SC) aggregated over the 10 seeds per domain still
  need a dedicated collector. Port `scripts/aggregate_mlp_baseline_results.py`
  conceptually to read these JSONs (or the W&B group) and emit the table.
- **MLP / KR2024 baseline sweeps.** `scripts/run_all_domains.py` covers
  the DIAS model. Add analogous orchestrators for the baselines so the
  full paper grid can be reproduced from this repo alone.

## Code organization

- Re-examine the model file (`dias.py`, ~1056 lines) for
  ablation branches that are no longer exercised by the shipped configs
  and can be removed (separate from the utils pass).
- Re-examine the model file (`dias.py`) for ablation branches that
  are no longer exercised by the shipped configs and can be removed.

## Docs

- Add an end-to-end "reproduce paper table X" recipe for each baseline once
  the sweep system is back.
- Smoke test (10-step single-domain training) wired into CI.
