import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod
from typing import Any
from jaxtyping import Shaped

class BaseTask(ABC):
    """
    Abstract base class for a task. A task defines how to compute the loss and metrics for a given model and batch of data.
    """
    
    @abstractmethod
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Any, 
        batch_idx: int,
        stage: str = "train"
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Computes the loss and metrics for a given model and batch of data.

        Args:
            model: The client's PyTorch model.
            batch: A batch from the dataloader.
            batch_idx: The current batch index.
            stage: "train", "val" or "test" (useful for logging prefixes).

        Returns:
            loss: Scalar tensor that autograd can call .backward() on.
            metrics: Dictionary with metrics that Lightning should log (e.g., MAE, accuracy).
        """
        pass


class StandardSupervisedTask(BaseTask):
    """
    Standard implementation for typical supervised learning problems
    (e.g., regression or classification).
    Assumes that a batch is a tuple (x, y).
    """
    
    def __init__(self, loss_fn: nn.Module):
        """
        Args:
            loss_fn: A PyTorch loss function, e.g., nn.MSELoss() or nn.CrossEntropyLoss().
        """
        super().__init__()
        self.loss_fn = loss_fn

    def compute_loss(
        self, 
        model: nn.Module, 
        batch: tuple[Shaped[Tensor, "batch ..."], Shaped[Tensor, "batch ..."]], 
        batch_idx: int,
        stage: str = "train"
    ) -> tuple[Tensor, dict[str, Tensor]]:
        
        x, y = batch
        
        # forward pass
        y_hat = model(x)
        
        # Calculate loss
        loss = self.loss_fn(y_hat, y)
        
        # Collect metrics to send to MLflow (loss.detach() is created to avoid memory leaks)
        metrics = {
            f"{stage}/loss": loss.detach(),
        }
        
        return loss, metrics