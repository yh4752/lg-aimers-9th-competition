"""Official RTDL FT-Transformer adapter."""

from __future__ import annotations

from typing import Mapping

from .common import ModelMetadata, import_runtime_module, make_two_input_checkpoint_wrapper


def ft_transformer_kwargs(
    model_config: Mapping[str, object], metadata: ModelMetadata
) -> dict[str, object]:
    return {
        "n_cont_features": metadata.n_num_features,
        "cat_cardinalities": list(metadata.categorical_cardinalities),
        "d_out": 1,
        "n_blocks": int(model_config["layers"]),
        "d_block": int(model_config["dim"]),
        "attention_n_heads": int(model_config["heads"]),
        "attention_dropout": float(model_config["attention_dropout"]),
        "ffn_d_hidden": None,
        "ffn_d_hidden_multiplier": 4 / 3,
        "ffn_dropout": float(model_config["ffn_dropout"]),
        "residual_dropout": 0.0,
    }


class FTTransformerAdapter:
    def build(
        self,
        model_config: Mapping[str, object],
        metadata: ModelMetadata,
        device: str,
    ) -> object:
        torch = import_runtime_module("torch")
        rtdl = import_runtime_module("rtdl_revisiting_models")
        model = rtdl.FTTransformer(**ft_transformer_kwargs(model_config, metadata))
        return make_two_input_checkpoint_wrapper(torch, model).to(device)

    def loss(self, model: object, x_num: object, x_cat: object, y: object) -> object:
        torch = import_runtime_module("torch")
        logits = model(x_num, x_cat).squeeze(-1)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())

    def probabilities(self, model: object, x_num: object, x_cat: object) -> object:
        return model(x_num, x_cat).squeeze(-1).sigmoid()

    def optimizer(
        self, model: object, training_config: Mapping[str, object]
    ) -> object:
        torch = import_runtime_module("torch")
        parameter_groups = model.model.make_parameter_groups()
        return torch.optim.AdamW(
            parameter_groups,
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
