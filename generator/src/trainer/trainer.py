import torch
import torch.nn.functional as F
from typing import Iterable
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

import src.model.model as model
import src.data_processing.data_processing as data_processing
import src.transformations.transformations as transformations
import src.visualizer.visualizer as visualizer

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


BATCH_NORM_TYPES = (
    torch.nn.BatchNorm1d
    | torch.nn.BatchNorm2d
    | torch.nn.BatchNorm3d
    | torch.nn.SyncBatchNorm
    | torch.nn.LazyBatchNorm1d
    | torch.nn.LazyBatchNorm2d
    | torch.nn.LazyBatchNorm3d
)


class Trainer:
    def __init__(self, lr: float = 2e-4, weight_decay: float = 3e-5,
                 batch_norm_momentum: float | None = 0.002, n_epochs: int = 10,
                 device: str = DEVICE, extra_augmentation: v2.Transform | None = None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_norm_momentum = batch_norm_momentum
        self.n_epochs = n_epochs
        self.device = device
        self.extra_augmentation = extra_augmentation  # TODO: Add reasonable extra augmentation

    def get_optimizer_and_scheduler(
            self, parameters: Iterable[torch.nn.Parameter]
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay, fused=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
        return optimizer, lr_scheduler

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets)

    def train(self, model: torch.nn.Module, train_loader: DataLoader):
        model = model.to(self.device)

        if self.batch_norm_momentum is not None:
            # Default torch.nn.BatchNorm2D.momentum is 0.1, but it's often too high.
            for m in model.modules():
                if isinstance(m, BATCH_NORM_TYPES):
                    m.momentum = self.batch_norm_momentum

        optimizer, lr_scheduler = self.get_optimizer_and_scheduler(model.parameters())

        for epoch in range(1, self.n_epochs + 1):
            self.train_epoch(model, train_loader, optimizer, epoch)
            lr_scheduler.step()

    def train_epoch(self, model: torch.nn.Module, train_loader: DataLoader,
                    test_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int):
        model.train()
        total_loss = 0.0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Train epoch {epoch:>3}")
        for batch in progress_bar:
            batch = batch.to(self.device)
            batch = transformations.transform_image_to_trainable_form(batch)
            optimizer.zero_grad()
            predictions = model(batch[:, :-1])
            loss = self.compute_loss(predictions, batch[:, 1:])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")


if __name__ == "__main__":
    trainer = Trainer(n_epochs=100)
    args = model.ModelArgs()
    model = model.SpatioTemporalTransformer(args).to(DEVICE)
    train_loader, test_loader = data_processing.get_dataloader(batch_size=1)
    trainer.train(model, train_loader)

    # get the first batch of the loader
    batch = next(iter(train_loader)).to(DEVICE)
    batch = transformations.transform_image_to_trainable_form(batch)
    predictions = model(batch[:, :-1])
    predictions_unnormalized = transformations.unnormalize_image(predictions)
    visualizer.visualize_tensor_image(predictions_unnormalized[0][-1])






