import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn

from ml_engine.configs.base import ExperimentConfig
from ml_engine.core.callbacks import build_callbacks
from ml_engine.core.task import BaseTask
from ml_engine.core.module import EngineModule


def run_training(
    config: ExperimentConfig,
    model: nn.Module,
    task: BaseTask,
    datamodule: pl.LightningDataModule,
) -> EngineModule:
    """
    Core entry point for training a model using ml-engine.
    
    Args:
        config: Pydantic configuration for the experiment.
        model: The client's raw PyTorch neural network.
        task: The task logic (how to compute the loss).
        datamodule: The client's PyTorch Lightning DataModule.
        
    Returns:
        The trained EngineModule (wrapper containing the weights).
    """
    # 1. Enforce absolute reproducibility globally
    pl.seed_everything(config.seed, workers=True)

    # 2. Wrap the architecture in the EngineModule
    engine = EngineModule(model=model, task=task, config=config)

    # 3. Build standard callbacks (checkpointing, early stopping, dataset logging)
    callbacks = build_callbacks(config=config)

    # 4. Initialize the Lightning Trainer based on Pydantic config
    trainer_config = config.trainer
    trainer = pl.Trainer(
        max_epochs=trainer_config.max_epochs,
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        strategy=trainer_config.strategy,
        precision=trainer_config.precision,
        callbacks=callbacks,
        deterministic=True, # Enforces deterministic CUDA algorithms if available
    )

    # 5. Execute Training
    trainer.fit(model=engine, datamodule=datamodule)
    
    return engine