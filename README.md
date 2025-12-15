# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding

### Basel II, risk measurement, and interpretability
Basel II formalizes a risk-sensitive approach to capital: banks are expected to **measure, monitor, and control credit risk** using quantifiable parameters (e.g., probability of default and loss estimates) and to demonstrate that these measures are governed by **sound model risk management**. Practically, this pushes us toward a model that is not only accurate but also **interpretable, auditable, and well-documented**. Interpretability and documentation support:

- **Transparency**: being able to explain which factors drive a score and in what direction.
- **Validation & governance**: enabling independent review, stress testing, stability monitoring, and change control.
- **Regulatory defensibility**: justifying design choices, data lineage, and assumptions during audits.

### Why a proxy target is necessary (and its business risks)
In many alternative-data settings we do not observe a clean, contractual **"default"** outcome (e.g., 90+ DPD, charge-off). To train supervised models, we create a **proxy label** that approximates credit distress using available signals (e.g., severe delinquency indicators, write-down events, collections, or behavioral patterns).

Using a proxy is necessary to:

- **Enable learning** from historical observations when true default is unavailable.
- **Operationalize risk** into a measurable target aligned with a business objective (reject/approve, pricing, limits).

However, proxy-based prediction has material risks:

- **Label risk / misalignment**: the proxy may not match true default, causing systematic bias.
- **Adverse selection**: miscalibration can approve riskier borrowers or decline good borrowers.
- **Fairness & compliance risk**: proxy construction may embed historical or measurement bias.
- **Profit and capital impact**: errors propagate into pricing, provisioning, and capital planning.

This means the proxy definition must be explicitly documented, sensitivity-tested, and monitored for drift.

### Simple scorecard (Logistic Regression + WoE) vs complex models (Gradient Boosting)
In regulated credit decisioning, the trade-offs are typically:

- **Interpretability vs predictive power**
  - Logistic Regression with Weight of Evidence (WoE) yields clear, monotonic relationships and supports a traditional scorecard that is easy to explain.
  - Gradient Boosting often improves discrimination (e.g., AUC/KS) but is harder to explain and govern.

- **Governance & documentation burden**
  - Simpler models reduce validation complexity (stability, reason codes, overrides).
  - Complex models require stronger explainability tooling, more extensive validation, and tighter monitoring.

- **Stability & monitoring**
  - Scorecards often behave more predictably under population shifts and are easier to recalibrate.
 

- **Operational constraints**
  - Scorecards are easier to implement consistently across channels and to provide actionable decline reasons.
  - Complex models can still be used, but usually with explicit explainability, challenger frameworks, and strong model risk controls.
