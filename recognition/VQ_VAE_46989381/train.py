import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from modules import VQVAE
from dataset import load_our_data
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Utilities ----------------------
def compute_metrics(x, recon):
    """Compute MSE and SSIM for a batch (x and recon in [0,1])."""
    mse = nn.functional.mse_loss(recon, x)
    ssim = ssim_fn(recon, x)  # returns mean over batch
    return mse.item(), ssim.item()

def plot_metrics(train_losses, val_losses, val_ssims, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses)+1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "losses.png"))

    plt.figure()
    plt.plot(epochs, val_ssims, label="Val SSIM", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM")
    plt.savefig(os.path.join(save_dir, "ssim.png"))

# ---------------------- Training ----------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    mse_fn = nn.MSELoss()
    for x, _ in loader:
        x = x.to(device, dtype=torch.float32) / 255.0
        optimizer.zero_grad()
        recon, vq_loss = model(x)
        loss = mse_fn(recon, x) + vq_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss, total_ssim = 0.0, 0.0
    mse_fn = nn.MSELoss()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, dtype=torch.float32) / 255.0
            recon, _ = model(x)
            loss, ssim_val = compute_metrics(x, recon)
            total_loss += loss
            total_ssim += ssim_val
    return total_loss / len(loader), total_ssim / len(loader)

def test(model, loader):
    model.eval()
    total_ssim = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, dtype=torch.float32) / 255.0
            recon, _ = model(x)
            _, ssim_val = compute_metrics(x, recon)
            total_ssim += ssim_val
    return total_ssim / len(loader)

# ---------------------- Main Script ----------------------
def main():
    base_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    batch_size = 32
    epochs = 25
    learning_rate = 1e-4

    train_loader, val_loader, test_loader, _ = load_our_data(base_path, batch_size=batch_size, normImage=False)

    model = VQVAE(latent_dim=128, num_embeddings=512, commitment_cost=0.25, output_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, val_ssims = [], [], []

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_ssim = validate(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ssims.append(val_ssim)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SSIM: {val_ssim:.3f}")

    # Final evaluation
    test_ssim = test(model, test_loader)
    print(f"Test SSIM: {test_ssim:.3f}")

    # Save metrics plots
    plot_metrics(train_losses, val_losses, val_ssims)

    # Save final model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/vqvae_hipmri_final.pth")
    print("Training complete. Model and plots saved.")

if __name__ == "__main__":
    main()