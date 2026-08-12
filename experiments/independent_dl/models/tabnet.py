"""TabNet adapter for the shared two-input binary trainer."""

from __future__ import annotations

from typing import Mapping

from .common import ModelMetadata, import_runtime_module


class TabNetAdapter:
    def __init__(self) -> None:
        self.lambda_sparse = 0.0

    def build(
        self,
        model_config: Mapping[str, object],
        metadata: ModelMetadata,
        device: str,
    ) -> object:
        torch = import_runtime_module("torch")
        tab_network = import_runtime_module("pytorch_tabnet.tab_network")
        nn = torch.nn
        embedding_dim = int(model_config["embedding_dim"])
        self.lambda_sparse = float(model_config["lambda_sparse"])

        class _TwoInputTabNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings = nn.ModuleList(
                    nn.Embedding(cardinality, embedding_dim)
                    for cardinality in metadata.categorical_cardinalities
                )
                input_dim = metadata.n_num_features + embedding_dim * len(
                    metadata.categorical_cardinalities
                )
                self.tabnet = tab_network.TabNetNoEmbeddings(
                    input_dim=input_dim,
                    output_dim=1,
                    n_d=int(model_config["n_d"]),
                    n_a=int(model_config["n_a"]),
                    n_steps=int(model_config["n_steps"]),
                    gamma=float(model_config["gamma"]),
                    momentum=float(model_config["momentum"]),
                    mask_type=str(model_config["mask_type"]),
                )

            def forward(self, x_num: object, x_cat: object) -> tuple[object, object]:
                pieces = [x_num]
                pieces.extend(
                    embedding(x_cat[:, index])
                    for index, embedding in enumerate(self.embeddings)
                )
                return self.tabnet(torch.cat(pieces, dim=1))

        return _TwoInputTabNet().to(device)

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
        logits, sparse_loss = model(x_num, x_cat)
        binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(-1), y.float()
        )
        return binary_loss - self.lambda_sparse * sparse_loss

    def probabilities(self, model: object, x_num: object, x_cat: object) -> object:
        logits, _ = model(x_num, x_cat)
        return logits.squeeze(-1).sigmoid()

    def optimizer(
        self, model: object, training_config: Mapping[str, object]
    ) -> object:
        torch = import_runtime_module("torch")
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
        )
