# SemanticCanary: Detecting Relational ML Failures Beyond Marginal Drift

## Abstract

Production ML systems can fail without an obvious change in the marginal distribution of any individual feature. An upstream system may preserve a feature's range and frequency while changing its relationship to other variables, units, or business semantics. Conventional drift monitoring can therefore produce a false sense of safety.

SemanticCanary is an ML reliability prototype that supplements marginal drift monitoring with relational and counterfactual canaries. The system learns operational expectations from a trusted reference window and tests whether those expectations remain stable in a current window.

The central hypothesis is:

> Relational canaries detect a class of model-breaking changes that univariate distribution monitoring cannot reliably identify.

The project evaluates this hypothesis using controlled corruption scenarios on a synthetic insurance-claims dataset.

## 1. Motivation

A common production monitoring pattern is:

```text
training data
     ↓
feature distributions
     ↓
drift metrics
     ↓
alert
```

This is useful, but incomplete. Consider a feature `claim_amount`. Suppose its values remain distributed almost identically to the training period, but the upstream pipeline randomly associates those amounts with different policy histories.

The marginal distribution has not meaningfully changed:

P(X) ≈ P(X')

but the joint structure can change:

P(X, Z) ≠ P(X', Z')

A model can rely heavily on that relationship even though a univariate drift detector remains quiet.

SemanticCanary therefore monitors both marginal behavior and relationships.

## 2. System design

### 2.1 Distribution canaries

For numeric features we compute Population Stability Index and Kolmogorov-Smirnov statistics.

PSI is useful for identifying changes in a feature's marginal histogram. It is intentionally treated as a baseline rather than the complete monitoring solution.

### 2.2 Relational canaries

For a reference and current window, the system compares Spearman correlations between numeric features.

For a pair (X_i, X_j), the system measures the change in Fisher-transformed correlation:

z = (atanh(r_current) - atanh(r_reference)) / SE

A large absolute z-score indicates that the relationship changed relative to sampling variability.

This is a monitoring heuristic, not a causal test.

### 2.3 Counterfactual canaries

The model is queried on an original row and a controlled perturbation:

x → x + Δ

For features whose model response has a sufficiently stable direction during reference calibration, SemanticCanary records the expected direction.

The current window is then tested for consistency.

This creates a lightweight behavioral test of the model itself, rather than only a test of the input distribution.

### 2.4 Schema/domain invariants

Some checks do not require a model:

- filing date must not precede incident date
- claim amounts must be nonnegative
- deductibles must be nonnegative

These are examples of hard data-quality invariants. They should be extended with domain-approved rules in a production deployment.

## 3. Failure benchmark

The benchmark introduces controlled failures:

### A. Marginal-preserving relationship break

A feature is shuffled within narrow quantile bins. Its marginal distribution remains approximately unchanged while relationships with other features are disrupted.

This is the most important experiment because it directly tests the project's motivation.

### B. Unit change

Claim amounts are multiplied by a fixed factor. This simulates an upstream change such as dollars becoming monthly units or another unit conversion.

### C. Column swap

Two numeric columns are swapped, simulating a schema-mapping failure.

### D. Date corruption

Filing dates are shifted, simulating a timestamp transformation problem.

## 4. Evaluation

The experiment should report:

| Metric | Meaning |
|---|---|
| Detection rate | Fraction of injected failures detected |
| False positive rate | Fraction of clean windows incorrectly flagged |
| Detection latency | How quickly a failure is detected |
| Root-cause rank | Position of the actual corrupted feature |
| Compute overhead | Monitoring cost relative to inference |

The most important comparison is:

```text
Univariate drift only
            vs.
Univariate + relational canaries
            vs.
Univariate + relational + counterfactual canaries
```

The project should not claim superiority until these experiments are actually run.

## 5. Expected result

The marginal-preserving relationship-break scenario is specifically designed so that a strong univariate monitor can miss the failure while relational monitoring remains sensitive.

This gives the project a falsifiable result rather than an assumed conclusion.

## 6. Limitations

1. Correlation changes are not automatically semantic changes.
2. Learned model directions are not causal relationships.
3. Thresholds such as PSI=0.20 and |z|=4 are engineering defaults, not universal statistical laws.
4. Synthetic data cannot reproduce all production failure modes.
5. A production implementation would need feature lineage, model versioning, alert routing, historical windows, and domain-owner validation.

## 7. Future work

- conditional dependence tests beyond pairwise correlation
- learned feature embeddings for high-dimensional relational drift
- graph-based feature dependency monitoring
- automatic canary generation from model explanations
- calibration drift and subgroup performance monitoring
- online change-point detection
- integration with MLflow, Evidently, Feast, or OpenTelemetry
- shadow traffic replay
- automatic promotion of validated canaries into CI/CD regression tests

## 8. Conclusion

SemanticCanary reframes ML monitoring as a behavioral testing problem.

Instead of asking only:

> "Did the feature distribution change?"

the system also asks:

> "Do the relationships the model depends on still behave the way they did when the model was validated?"

That distinction is the core contribution of the project.
