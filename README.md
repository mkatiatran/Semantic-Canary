# SemanticCanary

**SemanticCanary** is an ML reliability framework for detecting *meaning-preserving-looking but model-breaking* changes in production tabular data.

Instead of monitoring only marginal feature drift, SemanticCanary combines:

1. **Distribution canaries** — PSI-style population stability and KS statistics.
2. **Relational canaries** — correlations and conditional relationships learned from a trusted reference window.
3. **Counterfactual canaries** — controlled feature perturbations used to test whether model behavior remains consistent with learned directional relationships.
4. **Schema/domain invariants** — rules such as `claim_filed_date >= incident_date`.
5. **Root-cause ranking** — ranks features and relationships that changed the most.
6. **Failure replay** — generates controlled production failures so the monitoring system can be benchmarked.

The project is designed as an ML engineering portfolio project: reproducible experiments, a model-serving API, automated tests, Docker, and an experiment report.

## Research question

> **Can semantic/relational canaries detect model-breaking changes that marginal-distribution monitoring misses?**

The project deliberately distinguishes *distribution drift* from *semantic drift*. A feature can retain nearly the same marginal distribution while its relationship to other features changes.

## Dataset

The default experiment uses the synthetic `FreeInsuranceClaims100M` dataset on Hugging Face. It contains 100 million synthetic P&C insurance claims with fields including claim type, state, policyholder tenure, previous claims, incident/filing dates, claim amount, deductible, claim status, resolution time, and a fraud ground-truth flag. The repository samples a manageable subset rather than downloading the entire dataset.

The dataset is synthetic and does not contain real policyholders.

Alternative: the Kaggle Allstate Claims Severity dataset can be used for a claims-severity experiment, but its competition terms apply and the raw data should not be redistributed with this repository.

## Quick start

```bash
git clone <your-repo-url>
cd semantic-canary

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m semantic_canary.cli train --rows 100000
python -m semantic_canary.cli benchmark --rows 100000
python -m semantic_canary.cli report
```

To run the API:

```bash
uvicorn semantic_canary.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Repository structure

```text
semantic-canary/
├── semantic_canary/
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── monitoring.py
│   ├── canaries.py
│   ├── failures.py
│   └── report.py
├── tests/
├── configs/
│   └── default.yaml
├── notebooks/
│   └── experiment.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
├── .gitignore
├── LICENSE
└── README.md
```

## Important interpretation

A semantic canary is **not claiming that every learned relationship is causal or universally true**. Relationships are treated as operational expectations learned from a trusted reference period. A canary should be promoted to a hard production constraint only when a domain owner validates it.

## Portfolio framing

This is best presented as:

> **SemanticCanary: Detecting Relational ML Failures Beyond Marginal Drift**

The strongest result is not a single model accuracy number. It is a benchmark showing which failure modes conventional drift monitoring misses, how quickly semantic canaries detect them, and the false-positive/compute tradeoff.
