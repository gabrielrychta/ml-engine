from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from ml_engine.configs.base import ExperimentConfig
from ml_engine.core.task import BaseTask


class EngineModule(pl.LightningModule):
    """
    Universal LightningModule wrapper.
    Binds together the client's architecture (nn.Module), task logic (BaseTask), 
    and experiment configuration (ExperimentConfig).
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        task: BaseTask, 
        config: ExperimentConfig
    ):
        super().__init__()
        self.model = model
        self.task = task
        self.config = config
        
        # Save the config in the checkpoint for traceability.
        # We ignore 'model' and 'task' so Lightning doesn't try to save their entire state in hparams.
        self.save_hyperparameters({"config": config.model_dump()})

    def forward(self, x: Any) -> Any:
        """Forward pass for pure inference."""
        return self.model(x)

    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Tensor:
        """Shared logic for train, val, and test stages."""
        loss, metrics = self.task.compute_loss(
            model=self.model, 
            batch=batch, 
            batch_idx=batch_idx, 
            stage=stage
        )
        
        # Log all metrics (batch_size is handled automatically by Lightning)
        self.log_dict(
            metrics, 
            on_step=(stage == "train"), 
            on_epoch=True, 
            prog_bar=True,
            sync_dist=(stage != "train")  # Sync across GPUs during val/test
        )
        
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self) -> Any:
        """
        Dynamically builds the optimizer based on the Pydantic configuration.
        """
        opt_config = self.config.optimizer
        
        # Fetch the class from torch.optim (e.g., torch.optim.AdamW)
        optimizer_cls = getattr(torch.optim, opt_config.name)
        
        optimizer = optimizer_cls(
            self.model.parameters(),
            lr=opt_config.lr,
            weight_decay=opt_config.weight_decay,
            **opt_config.optimizer_kwargs
        )
        
        # If no scheduler is defined, just return the optimizer
        if opt_config.scheduler_name is None:
            return optimizer
            
        # Dynamic handling of the scheduler (e.g., CosineAnnealingLR)
        scheduler_cls = getattr(torch.optim.lr_scheduler, opt_config.scheduler_name)
        
        # NOTE: In a more advanced setup, we would expand the config to include scheduler_kwargs
        scheduler = scheduler_cls(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",  # Default value in case we use ReduceLROnPlateau
                "interval": "epoch",
                "frequency": 1
            }
        }