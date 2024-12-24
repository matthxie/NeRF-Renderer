import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple
import wandb
from tqdm import tqdm
from reflectance_dataset import ReflectanceDataset


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.norm(vectors, dim=-1, keepdim=True)


def train(
    model: nn.Module,
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    num_epochs: int = 100,
    batch_size: int = 1024,
    learning_rate: float = 1e-4,
    use_wandb: bool = False,
    scheduler_patience: int = 10,
    early_stopping_patience: int = 20,
) -> dict:
    """
    Args:
        model: The NeRF model
        data: Tuple of (positions, view_dirs, light_dirs, colors)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        use_wandb: Whether to log metrics to Weights & Biases
        scheduler_patience: Epochs to wait before reducing learning rate
        early_stopping_patience: Epochs to wait before early stopping

    Returns:
        dict: Training history
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ReflectanceDataset(*data)

    train_loader = DataLoader(
        dataset.train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(dataset.val_dataset, batch_size=batch_size)

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience, verbose=True
    )

    mse_loss = nn.MSELoss()

    if use_wandb:
        wandb.init(
            project="nerf-reflectance",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
            },
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": float("inf"),
        "best_model_state": None,
    }

    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                positions = batch["position"].to(device)
                view_dirs = normalize_vectors(batch["view_dir"]).to(device)
                light_dirs = normalize_vectors(batch["light_dir"]).to(device)
                target_colors = batch["color"].to(device)

                predicted_colors = model(positions, view_dirs, light_dirs)

                loss = mse_loss(predicted_colors, target_colors)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})

        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                positions = batch["position"].to(device)
                view_dirs = normalize_vectors(batch["view_dir"]).to(device)
                light_dirs = normalize_vectors(batch["light_dir"]).to(device)
                target_colors = batch["color"].to(device)

                predicted_colors = model(positions, view_dirs, light_dirs)
                loss = mse_loss(predicted_colors, target_colors)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if use_wandb:
            wandb.log(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < history["best_val_loss"]:
            history["best_val_loss"] = avg_val_loss
            history["best_model_state"] = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    model.load_state_dict(history["best_model_state"])

    if use_wandb:
        wandb.finish()

    return history
