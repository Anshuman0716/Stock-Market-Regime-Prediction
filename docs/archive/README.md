# Archive: historical project notes

> **Do not quote numbers from these files.** They were written before the
> improvement roadmap (October 2026) and describe an earlier version of the
> project. Their performance figures come from an evaluation that counted
> about ten years of pre-model cash as strategy performance, used an older
> feature set and a different signal. Every current result lives in the
> main [README](../../README.md) and in `data/regime_store.db`.

They are kept because they record how the project got here: what was
audited, which problems were found, and why later design decisions were
made.

| File | What it was |
|---|---|
| `audit_baseline.md` | Audit of the original notebook-only code: duplicated logic, look-ahead risks |
| `feature_notes.md` | The original 18 indicators and their expected regime behaviour (later cut to 8) |
| `gate_phase1.md`, `gate_phase2.md` | Review checklists for the first two build phases |
| `phase2_summary.md` | Why the feature set, covariance type and training window were changed |
| `statistical_findings.md` | First significance tests and feature ablation (numbers superseded) |
| `final_audit.md` | Pre-roadmap model-validation review. Its findings (duplicated smoothing, raw labels in the database, backfill re-fitting) were fixed in Steps 2–3 |
| `number_provenance.md` | Where the pre-roadmap README numbers came from. Those numbers, including the "56% → 22% drawdown" claim, came from the evaluation bug fixed in Step 1 |
| `figures/bic_aic_sweep.png` | Early BIC / AIC sweep over the number of states |

What changed since, in short:

- **Evaluation window.** Every strategy is now scored only on out-of-sample
  days, starting at the model's first prediction (Step 1).
- **One signal.** Smoothed regime probabilities, regime and position size are
  defined once in `src/models/signal.py` (Step 2).
- **Scale-free features.** ATR and MACD are divided by price; OBV ratio was
  replaced by a bounded volume-flow measure (Step 2).
- **One pipeline, one store.** `scripts/run_pipeline.py` trains once and
  writes everything the dashboard and notebooks read (Step 3).
- **Realistic P&L and a tougher test (Steps 6–7).** Tradable funds, interest
  on cash, a volatility-targeting baseline, and states labeled by what they
  earned rather than how calm they were.
