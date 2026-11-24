import pytorch_lightning as pl
import torch
from torch import nn

from ml_engine.configs.base import DataConfig, ExperimentConfig, OptimizerConfig, TrainerConfig
from ml_engine.core.callbacks import DatasetMetadata
from ml_engine.core.task import StandardSupervisedTask
from ml_engine.runners.train import run_training


class DummyModel(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, data_config: DataConfig, num_samples: int = 100, input_dim: int = 10):
        super().__init__()
        self.data_config = data_config
        self.num_samples = num_samples
        self.input_dim = input_dim

    def setup(self, stage: str | None = None):
        # Generate synthetic data: y = sum(x) + noise
        x = torch.randn(self.num_samples, self.input_dim)
        y = torch.sum(x, dim=1, keepdim=True) + torch.randn(self.num_samples, 1) * 0.1    

        train_split = int(0.7 * self.num_samples)
        val_split = int(0.85 * self.num_samples)
        self.train_data = torch.utils.data.TensorDataset(x[:train_split], y[:train_split])
        self.val_data = torch.utils.data.TensorDataset(x[train_split + 1 : val_split], y[train_split + 1 : val_split])
        self.test_data = torch.utils.data.TensorDataset(x[val_split + 1:], y[val_split + 1:])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=self.data_config.batch_size, 
            shuffle=True,
            num_workers=self.data_config.num_workers
            )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=self.data_config.batch_size, 
            shuffle=False,
            num_workers=self.data_config.num_workers
            )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=self.data_config.batch_size, 
            shuffle=False,
            num_workers=self.data_config.num_workers
            )


    def get_dataset_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            summary={
                "num_train_samples": len(self.train_data),
                "num_val_samples": len(self.val_data),
                "input_dim": self.data_config.input_dim,
            },
            details={
                "details": "I can write whatever I want here, hell yeah moment",
                "train_indices": list(range(len(self.train_data))),
                "val_indices": list(range(len(self.val_data))),
            }
        )


def main() -> None:
    # 1. Simple experiment configuration
    config = ExperimentConfig(
        experiment_name="tiny_test",
        seed=67,
        trainer=TrainerConfig(max_epochs=100, accelerator="cpu", devices=1, log_every_n_steps=10),
        optimizer=OptimizerConfig(lr=1e-3),
        data=DataConfig(dataset_name="dummy_dataset", batch_size=32, num_workers=0)
    )

    # 2. Initialize the dummy model and data module
    model = DummyModel(input_dim=10, hidden_dim=20, output_dim=1)
    datamodule = DummyDataModule(data_config=config.data, num_samples=100, input_dim=10)

    # 3. Define a simple task (mean squared error)
    task = StandardSupervisedTask(loss_fn=torch.nn.MSELoss())

    run_training(config=config, model=model, task=task, datamodule=datamodule)

if __name__ == "__main__":
    main()



