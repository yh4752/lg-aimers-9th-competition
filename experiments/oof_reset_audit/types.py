from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd


class TrustClass(str, Enum):
    RULE_SAFE = "rule_safe"
    QUARANTINED_DIAGNOSTIC = "quarantined_diagnostic"


class ArtifactRole(str, Enum):
    UNKNOWN = "unknown"
    STAGE_C_TABM = "stage_c_tabm"
    ROW_FEATURE = "row_feature"
    CATBOOST_BLEND = "catboost_blend"
    CATBOOST_DEPLOYMENT = "catboost_deployment"
    HIERARCHICAL = "hierarchical"
    QUARANTINED_XGBOOST = "quarantined_xgboost"


@dataclass(frozen=True)
class PredictionSet:
    artifact_path: Path
    artifact_sha256: str
    source_member: str
    prediction_sha256: str
    model_id: str
    fold: str
    trust: TrustClass
    frame: pd.DataFrame


@dataclass(frozen=True)
class ArtifactRecord:
    path: Path
    sha256: str
    role: ArtifactRole
    artifact_kind: str
    status: str
    detail: str | None
