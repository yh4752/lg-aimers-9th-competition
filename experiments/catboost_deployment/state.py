from __future__ import annotations

from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Mapping

from experiments.catboost_preprocessing.features import CatBoostFeatureState
from experiments.independent_dl.preprocessing import (
    PreprocessingSpec,
    PreprocessingState,
    normalize_spec,
)


class DeploymentStateError(ValueError):
    """Raised when a frozen CatBoost preprocessing state is ambiguous."""


_TOP_KEYS = {
    "schema_version",
    "artifact_kind",
    "preprocessing",
    "categorical_columns",
    "category_values",
    "state_sha256",
}
_PREPROCESSING_KEYS = {
    "spec",
    "source_columns",
    "output_columns",
    "categorical_columns",
    "numeric_columns",
    "numeric_median",
    "numeric_mean",
    "numeric_std",
    "yeo_johnson_lambda",
    "entity_frequency",
    "target_prior",
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _string_tuple(value: object, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise DeploymentStateError(f"{label} must be a string list")
    result = tuple(value)
    if (nonempty and not result) or len(result) != len(set(result)):
        raise DeploymentStateError(f"{label} is invalid")
    return result


def _finite_mapping(value: object, label: str) -> Mapping[str, float]:
    if type(value) is not dict:
        raise DeploymentStateError(f"{label} must be an object")
    output: dict[str, float] = {}
    for key, item in value.items():
        if type(key) is not str or type(item) not in (int, float) or type(item) is bool:
            raise DeploymentStateError(f"{label} contains an invalid value")
        number = float(item)
        if not math.isfinite(number):
            raise DeploymentStateError(f"{label} contains a non-finite value")
        output[key] = number
    return MappingProxyType(output)


def _frequency_mapping(value: object) -> Mapping[str, Mapping[str, int]]:
    if type(value) is not dict:
        raise DeploymentStateError("entity frequency must be an object")
    output: dict[str, Mapping[str, int]] = {}
    for column, mapping in value.items():
        if type(column) is not str or type(mapping) is not dict:
            raise DeploymentStateError("entity frequency contains an invalid mapping")
        counts: dict[str, int] = {}
        for key, count in mapping.items():
            if type(key) is not str or type(count) is not int or count < 0:
                raise DeploymentStateError("entity frequency contains an invalid count")
            counts[key] = count
        output[column] = MappingProxyType(counts)
    return MappingProxyType(output)


def _state_to_exact_payload(state: CatBoostFeatureState) -> dict[str, object]:
    preprocessing = state.preprocessing
    return {
        "schema_version": 1,
        "artifact_kind": "catboost_feature_state",
        "preprocessing": {
            "spec": {
                "profile": preprocessing.spec.profile,
                "components": list(preprocessing.spec.components),
            },
            "source_columns": list(preprocessing.source_columns),
            "output_columns": list(preprocessing.output_columns),
            "categorical_columns": list(preprocessing.categorical_columns),
            "numeric_columns": list(preprocessing.numeric_columns),
            "numeric_median": dict(preprocessing.numeric_median),
            "numeric_mean": dict(preprocessing.numeric_mean),
            "numeric_std": dict(preprocessing.numeric_std),
            "yeo_johnson_lambda": dict(preprocessing.yeo_johnson_lambda),
            "entity_frequency": {
                column: dict(values)
                for column, values in preprocessing.entity_frequency.items()
            },
            "target_prior": preprocessing.target_prior,
        },
        "categorical_columns": list(state.categorical_columns),
        "category_values": {
            column: list(values) for column, values in state.category_values.items()
        },
    }


def serialize_feature_state(state: CatBoostFeatureState) -> bytes:
    payload = _state_to_exact_payload(state)
    payload["state_sha256"] = sha256(canonical_json(payload)).hexdigest()
    encoded = canonical_json(payload)
    deserialize_feature_state(encoded)
    return encoded


def _exact_payload_to_state(root: dict[str, object]) -> CatBoostFeatureState:
    if set(root) != _TOP_KEYS:
        raise DeploymentStateError("feature state keys differ")
    declared_sha = root["state_sha256"]
    body = {key: value for key, value in root.items() if key != "state_sha256"}
    try:
        observed_sha = sha256(canonical_json(body)).hexdigest()
    except (TypeError, ValueError) as error:
        raise DeploymentStateError("feature state body is not canonical") from error
    if (
        type(declared_sha) is not str
        or len(declared_sha) != 64
        or observed_sha != declared_sha
    ):
        raise DeploymentStateError("feature state SHA-256 differs")
    if root["schema_version"] != 1 or root["artifact_kind"] != "catboost_feature_state":
        raise DeploymentStateError("feature state identity differs")
    preprocessing_value = root["preprocessing"]
    if type(preprocessing_value) is not dict or set(preprocessing_value) != _PREPROCESSING_KEYS:
        raise DeploymentStateError("preprocessing state keys differ")
    spec_value = preprocessing_value["spec"]
    if type(spec_value) is not dict or set(spec_value) != {"profile", "components"}:
        raise DeploymentStateError("preprocessing spec differs")
    components = _string_tuple(spec_value["components"], "components")
    if type(spec_value["profile"]) is not str:
        raise DeploymentStateError("preprocessing profile differs")
    try:
        spec = normalize_spec(PreprocessingSpec(spec_value["profile"], components))
    except Exception as error:
        raise DeploymentStateError(f"preprocessing spec is invalid: {error}") from error
    if spec.profile != "tree_native" or spec.components != ("hand_matchup",):
        raise DeploymentStateError("deployment preprocessing spec differs")

    source_columns = _string_tuple(
        preprocessing_value["source_columns"], "source columns", nonempty=True
    )
    output_columns = _string_tuple(
        preprocessing_value["output_columns"], "output columns", nonempty=True
    )
    categorical = _string_tuple(
        preprocessing_value["categorical_columns"], "categorical columns"
    )
    numeric = _string_tuple(preprocessing_value["numeric_columns"], "numeric columns")
    if set(categorical).intersection(numeric) or set(categorical) | set(numeric) != set(output_columns):
        raise DeploymentStateError("preprocessing column partition differs")
    if "control_success" in source_columns or "control_success" in output_columns:
        raise DeploymentStateError("target leaked into preprocessing columns")

    numeric_median = _finite_mapping(preprocessing_value["numeric_median"], "numeric median")
    numeric_mean = _finite_mapping(preprocessing_value["numeric_mean"], "numeric mean")
    numeric_std = _finite_mapping(preprocessing_value["numeric_std"], "numeric std")
    lambdas = _finite_mapping(preprocessing_value["yeo_johnson_lambda"], "Yeo-Johnson lambda")
    frequency = _frequency_mapping(preprocessing_value["entity_frequency"])
    if any((numeric_median, numeric_mean, numeric_std, lambdas, frequency)):
        raise DeploymentStateError("tree-native fitted numeric state must be empty")
    prior_value = preprocessing_value["target_prior"]
    if type(prior_value) not in (int, float) or type(prior_value) is bool:
        raise DeploymentStateError("target prior is invalid")
    prior = float(prior_value)
    if not math.isfinite(prior) or not 0.0 <= prior <= 1.0:
        raise DeploymentStateError("target prior is invalid")

    outer_categorical = _string_tuple(root["categorical_columns"], "outer categorical columns")
    if outer_categorical != categorical:
        raise DeploymentStateError("categorical column binding differs")
    category_value = root["category_values"]
    if type(category_value) is not dict or set(category_value) != set(categorical):
        raise DeploymentStateError("category value keys differ")
    categories: dict[str, tuple[str, ...]] = {}
    for column in categorical:
        values = _string_tuple(category_value[column], f"category values for {column}")
        if values != tuple(sorted(values)):
            raise DeploymentStateError(f"category values are not canonical: {column}")
        categories[column] = values

    preprocessing = PreprocessingState(
        spec=spec,
        source_columns=source_columns,
        output_columns=output_columns,
        categorical_columns=categorical,
        numeric_columns=numeric,
        numeric_median=numeric_median,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        yeo_johnson_lambda=lambdas,
        entity_frequency=frequency,
        target_prior=prior,
    )
    return CatBoostFeatureState(
        preprocessing=preprocessing,
        categorical_columns=outer_categorical,
        category_values=MappingProxyType(categories),
    )


def deserialize_feature_state(payload: bytes) -> CatBoostFeatureState:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise DeploymentStateError(f"feature state is unreadable: {error}") from error
    if type(value) is not dict:
        raise DeploymentStateError("feature state must be an object")
    return _exact_payload_to_state(value)


def feature_state_sha256(state: CatBoostFeatureState) -> str:
    return sha256(serialize_feature_state(state)).hexdigest()
