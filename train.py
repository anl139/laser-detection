import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset_loader import LaserHeatmapDataset

from model import SimpleUNet
from model import CoarseDetector
from model import FineDetector


def train_model(

    image_dir="dataset/images",

    csv_path="dataset/labels.csv",

    model_type="unet",

    image_size=256,

    batch_size=8,

    epochs=50,

    lr=1e-4,

    save_path="models/laser_heatmap.pth"

):

    os.makedirs("models", exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    dataset = LaserHeatmapDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        image_size=image_size,
        augment=True
    )

    n_val = max(1, int(0.1 * len(dataset)))

    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    if model_type == "coarse":
        model = CoarseDetector()

    elif model_type == "fine":
        model = FineDetector()

    else:
        model = SimpleUNet()

    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    best_val = float("inf")

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0

        for images, heatmaps in train_loader:

            images = images.to(device)
            heatmaps = heatmaps.to(device)

            preds = model(images)

            loss = criterion(preds, heatmaps)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for images, heatmaps in val_loader:

                images = images.to(device)
                heatmaps = heatmaps.to(device)

                preds = model(images)

                loss = criterion(preds, heatmaps)

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs}"
            f" train={train_loss:.5f}"
            f" val={val_loss:.5f}"
        )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": model_type
                },
                save_path
            )

            print("Saved:", save_path)


if __name__ == "__main__":

    train_model()