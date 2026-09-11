from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class MonitoringConfig:
    psi_threshold: float = 0.20
    ks_pvalue_threshold: float = 0.01
    relation_z_threshold: float = 4.0
    counterfactual_rate_threshold: float = 0.90
    bins: int = 10

@dataclass
class Config:
    dataset: str = "ziadatalabs/FreeInsuranceClaims100M"
    target: str = "is_fraud_flagged_ground_truth"
    sample_rows: int = 100000
    seed: int = 42
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

def load_config(path="configs/default.yaml"):
    p = Path(path)
    if not p.exists():
        return Config()
    raw = yaml.safe_load(p.read_text())
    m = MonitoringConfig(**raw.get("monitoring", {}))
    d = raw.get("data", {})
    return Config(
        dataset=d.get("hf_dataset", Config.dataset),
        sample_rows=d.get("sample_rows", 100000),
        seed=d.get("seed", 42),
        target=raw.get("target", Config.target),
        monitoring=m,
    )
