from pathlib import Path
import json
import pandas as pd

def write_report(results, path="artifacts/benchmark.md"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SemanticCanary Benchmark",
        "",
        "This report is generated from the controlled failure benchmark.",
        "",
        "| Scenario | Distribution drift | Relation drift | Counterfactual failures |",
        "|---|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result.get('distribution_alerts', 0)} | "
            f"{result.get('relation_alerts', 0)} | "
            f"{result.get('counterfactual_alerts', 0)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The key experiment is the marginal-preserving relationship break. "
        "Its purpose is to test whether relational monitoring finds a failure "
        "even when a conventional marginal drift test has little signal.",
        "",
        "Do not treat learned relationships as causal claims. "
        "Production promotion of a canary should require domain validation.",
    ]
    Path(path).write_text("\n".join(lines))
    return path
