from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger


@dataclass
class DatasetMetadata:
    """
    Separating simple metrics from metadata.
    - summary: Scalars (int, float, str, bool) logged as MLflow params.
    - details: Lists/dicts (e.g., file IDs) saved as a JSON file.
    """

    summary: dict[str, int | float | str | bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LoggableDataModule(Protocol):
    """Any DataModule that defines this method will be automatically logged."""

    def get_dataset_metadata(self) -> DatasetMetadata: ...


class DatasetLoggingCallback(pl.Callback):
    """Logs dataset metadata to MLflow once, right before training begins."""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # 1. In multi-GPU training, only log from the main process
        if not trainer.is_global_zero:
            return

        # 2. Check that we actually have an MLFlowLogger attached
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        # 3. Check if the client's DataModule implements get_dataset_metadata()
        datamodule = getattr(trainer, "datamodule", None)
        if not isinstance(datamodule, LoggableDataModule):
            return

        # 4. Fetch the metadata from the client's DataModule
        metadata = datamodule.get_dataset_metadata()
        mlflow_logger = trainer.logger

        # 5. Log summary scalars as searchable MLflow params (prefixed with "dataset.")
        if metadata.summary:
            prefixed_summary = {f"dataset.{k}": v for k, v in metadata.summary.items()}
            mlflow_logger.log_hyperparams(prefixed_summary)

        # 6. Save rich details as a JSON artifact attached to this MLflow run.
        # mlflow_logger.experiment is the underlying MLflow client object.
        if metadata.details and mlflow_logger.run_id is not None:
            mlflow_logger.experiment.log_dict(
                run_id=mlflow_logger.run_id,
                dictionary=metadata.details,
                artifact_file="dataset_metadata.json",
            )