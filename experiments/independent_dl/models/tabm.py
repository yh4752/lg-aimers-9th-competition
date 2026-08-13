"""Official TabM adapter with the package-prescribed ensemble semantics."""

from __future__ import annotations

from typing import Mapping

from .common import (
    ModelMetadata,
    import_runtime_module,
    make_numeric_embeddings,
    make_two_input_checkpoint_wrapper,
)


SUPPORTED_NUM_EMBEDDINGS = {
    "linear_relu",
    "piecewise_linear",
    "periodic",
}


class TabMAdapter:
    def __init__(self, loss_name: str = "bce") -> None:
        if loss_name not in {"bce", "brier"}:
            raise ValueError(f"unsupported TabM loss: {loss_name}")
        self.loss_name = loss_name

    def build(
        self,
        model_config: Mapping[str, object],
        metadata: ModelMetadata,
        device: str,
    ) -> object:
        torch = import_runtime_module("torch")
        tabm = import_runtime_module("tabm")
        rtdl_num_embeddings = import_runtime_module("rtdl_num_embeddings")
        embedding_mode = str(model_config["num_embedding"])
        if embedding_mode not in SUPPORTED_NUM_EMBEDDINGS:
            raise ValueError(f"unsupported numerical embedding: {embedding_mode}")
        num_embeddings = make_numeric_embeddings(
            embedding_mode, metadata, torch, rtdl_num_embeddings
        )
        architecture = str(model_config["architecture"]).replace("_", "-")
        model = tabm.TabM.make(
            n_num_features=metadata.n_num_features,
            cat_cardinalities=list(metadata.categorical_cardinalities),
            d_out=1,
            num_embeddings=num_embeddings,
            n_blocks=int(model_config["blocks"]),
            d_block=int(model_config["width"]),
            dropout=float(model_config["dropout"]),
            k=int(model_config["k"]),
            arch_type=architecture,
        )
        return make_two_input_checkpoint_wrapper(torch, model).to(device)

    def loss(
        self,
        model: object,
        x_num: object,
        x_cat: object,
        y: object,
        *,
        row_indices: object,
    ) -> object:
        del row_indices
        torch = import_runtime_module("torch")
        member_logits = model(x_num, x_cat).squeeze(-1)
        if self.loss_name == "brier":
            probabilities = member_logits.sigmoid().mean(dim=1)
            return ((probabilities - y.float()) ** 2).mean()
        member_targets = y.float().unsqueeze(1).expand_as(member_logits)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            member_logits, member_targets, reduction="mean"
        )

    def probabilities(self, model: object, x_num: object, x_cat: object) -> object:
        member_logits = model(x_num, x_cat).squeeze(-1)
        return member_logits.sigmoid().mean(dim=1)

    def optimizer(
        self, model: object, training_config: Mapping[str, object]
    ) -> object:
        torch = import_runtime_module("torch")
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
