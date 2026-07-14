# TODO

Follow-up items intentionally deferred from the initial public release.

## Reproducibility surface

- **Aggregation across (domain, seed).** ✅ `scripts/aggregate_results.py`
  reads the JSON tree under `logs/validation/<group>/` and emits a
  precision / recall / accuracy table (mean ± std over seeds × problems)
  in table, CSV, or JSON format.
- **MLP / KR2024 baseline sweeps.** ✅ Added sweep orchestrators:
  - `scripts/run_all_domains_mlp.py` — reads run-ids from a prior DIAS
    sweep and runs MLP-baseline validation for every (domain, seed).
  - `scripts/run_all_domains_kr2024.py` — validates pre-generated KR-2024
    learned PDDL files for every (domain, seed).

## Code organization

- Re-examine the model file (`dias.py`) for ablation branches that are no
  longer exercised by the shipped configs and can be removed.

## Docs

- Add an end-to-end "reproduce paper table X" recipe for each baseline once
  the sweep system is back.
- ✅ Smoke test (10-step single-domain training) wired into CI via
  `configs/smoke_test.yaml` and `.github/workflows/smoke_test.yml`.
