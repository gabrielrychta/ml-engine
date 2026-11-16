from pydantic import BaseModel, Field
from typing import Literal, Any

class OptimizerConfig(BaseModel):
    name: str = Field(default="AdamW", description="Name of the torch.optim optimizer")
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    scheduler_name: str | None = Field(default=None, description="Optional lr_scheduler")

class DataConfig(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset")
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)
    data_dir: str | None = Field(default=None, description="Optional path to the dataset directory")
    transform_kwargs: dict[str, Any] = Field(default_factory=dict, description="Optional kwargs for data transforms")

class TrainerConfig(BaseModel):
    max_epochs: int = Field(default=100, gt=0)
    accelerator: Literal["cpu", "gpu", "auto"] = Field(default="auto")
    devices: int | Literal["auto"] = Field(default="auto")
    strategy: Literal["auto", "ddp"] = Field(default="auto")
    precision: Literal["32-true", "16-mixed"] = Field(default="32-true")
    
class ExperimentConfig(BaseModel):
    experiment_name: str = Field(..., description="Name of the experiment")
    seed: int = Field(default=67, description="Global seed")
    
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    data: DataConfig = Field(default_factory=DataConfig)