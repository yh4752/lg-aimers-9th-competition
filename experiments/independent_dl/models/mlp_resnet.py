"""Official RTDL MLP and ResNet adapter with shared categorical embeddings."""

from __future__ import annotations

from typing import Mapping

from .common import (
    ModelMetadata,
    import_runtime_module,
    make_categorical_backbone_wrapper,
)


def mlp_resnet_kwargs(
    model_config: Mapping[str, object], metadata: ModelMetadata
) -> dict[str, object]:
    embedding_dim = int(model_config["embedding_dim"])
    return {
        "d_in": metadata.n_num_features
        + embedding_dim * len(metadata.categorical_cardinalities),
        "d_out": 1,
        "n_blocks": int(model_config["blocks"]),
        "d_block": int(model_config["width"]),
    }


class MLPResNetAdapter:
    def build(
        self,
        model_config: Mapping[str, object],
        metadata: ModelMetadata,
        device: str,
    ) -> object:
        torch = import_runtime_module("torch")
        rtdl = import_runtime_module("rtdl_revisiting_models")
        architecture = str(model_config["architecture"])
        dropout = float(model_config["dropout"])
        kwargs = mlp_resnet_kwargs(model_config, metadata)
        if architecture == "mlp":
            backbone = rtdl.MLP(
                **kwargs,
                dropout=dropout,
            )
        elif architecture == "resnet":
            backbone = rtdl.ResNet(
                **kwargs,
                d_hidden=None,
                d_hidden_multiplier=2.0,
                dropout1=dropout,
                dropout2=0.0,
            )
        else:
            raise ValueError(f"unsupported MLP/ResNet architecture: {architecture}")
        return make_categorical_backbone_wrapper(
            torch,
            backbone,
            metadata.categorical_cardinalities,
            int(model_config["embedding_dim"]),
        ).to(device)

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
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
