# Standing Rules for the Market Regime Detection Project

These rules apply to every phase of the build. Read them before writing any
code or making any decision.

---

## 1. Single Source of Truth

`src/` is the single source of truth for all logic — feature engineering,
HMM training/prediction, regime labeling, backtest mechanics, risk
calculations, and statistical tests.

Notebooks and `app.py` **import** from `src/`; they never redefine or
recompute the same logic locally. If you catch yourself pasting a function
into a second file, stop and import it instead.

---

## 2. No Future Information — Anywhere

If a computation for day *t* touches data from day *t+1* or later, it is a
bug. This applies to every stage of the pipeline, without exception:

- **Scaling:** `StandardScaler` (or any scaler) must be fit *only* on the
  training window at each walk-forward step — never on the full dataset.
- **HMM fitting:** The model at time *t* must never have seen data from
  after *t*. Use walk-forward (expanding or rolling window) fitting.
- **Smoothing:** Use a trailing-only rolling mode window. No centered
  windows. No future days are allowed into today's label.
- **Feature windows:** All rolling calculations (`rolling()`, `ewm()`,
  etc.) must use only past and current observations.
- **No `bfill()`, `fillna(method='bfill')`, `shift(-n)`, or backward
  `interpolate()`** on any feature, label, or signal.

---

## 3. Comments Explain *Why*, Not Just *What*

Anshuman is building this project to learn ML/quant concepts, not just to
ship it. In `src/`, comments should explain **why** a design choice was
made — e.g. why walk-forward instead of a single fit, why trailing
smoothing instead of centered, why full covariance instead of diagonal —
not just restate what the line does.

---

## 4. Explicit Out-of-Scope List

Do **not** add any of the following, even if they seem like a nice-to-have:

| Category | Excluded Tools |
|---|---|
| AI/LLM tooling | RAG, vector DB, LangChain, any LLM or agent framework |
| Orchestration | Airflow, dbt, Spark, Kafka |
| Containerization / deployment | Docker, FastAPI, any cloud deployment target |
| Experiment tracking | MLflow |

If a task on the roadmap appears to require one of these, it is a sign the
task has drifted out of scope — **flag it** rather than building it.

---

## 5. Factual Claims Only

Never write a number, Sharpe ratio, or performance claim into `README.md`
(or any documentation) that has not been produced by code that has actually
run. Sections marked `[TODO]` stay as `[TODO]` until the corresponding
phase is complete and measured.

---

## 6. Data Flow Direction

The pipeline flows in one direction:

```
loader → engineering → walk-forward HMM → labeling → backtest/risk → SQLite store
```

Notebooks and `app.py` *read* from `src/` and the store; they never
recompute independently.

---

## 7. Stay in Scope

This project targets **Data Scientist, quant/fintech flavor**. It does not
stretch to ML Engineer, AI Engineer, or Data Engineer scope. See README
Section 2 for the full keep/cut/fix/add table.
