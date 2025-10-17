import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import VQVAE2
from dataset import load_our_data
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

# ---------------------- Configuration ----------------------
SAVE_DIR = "train_dir"
BASE_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)  # create save directory if it doesn't exist

# ---------------------- Utilities ----------------------
def compute_metrics(x, recon):
    """Compute MSE and SSIM for a batch (x and recon in [0,1])."""
    mse = nn.functional.mse_loss(recon, x)
    ssim = ssim_fn(recon, x)  # returns mean over batch
    return mse.item(), ssim.item()

def plot_metrics(train_losses, val_losses, val_ssims, save_dir=SAVE_DIR):
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(1, len(train_losses)+1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "losses.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, val_ssims, label="Val SSIM", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM")
    plt.savefig(os.path.join(plot_dir, "ssim.png"))
    plt.close()

# ---------------------- Training ----------------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    mse_fn = nn.MSELoss()
    for x, _ in loader:
        x = x.to(DEVICE, dtype=torch.float32) / 255.0
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
            x = x.to(DEVICE, dtype=torch.float32) / 255.0
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
            x = x.to(DEVICE, dtype=torch.float32) / 255.0
            recon, _ = model(x)
            _, ssim_val = compute_metrics(x, recon)
            total_ssim += ssim_val
    return total_ssim / len(loader)

# ---------------------- Main Script ----------------------
def main():
    batch_size = 32
    epochs = 80
    learning_rate = 1e-4

    train_loader, val_loader, test_loader = load_our_data(BASE_PATH, batch_size=batch_size, normImage=False)

    model = VQVAE2(latent_dim=256, num_embeddings=1024, commitment_cost=0.25, output_channels=1).to(DEVICE)
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

    # Save metrics plots in SAVE_DIR
    plot_metrics(train_losses, val_losses, val_ssims, save_dir=SAVE_DIR)

    # Save final model in SAVE_DIR
    model_path = os.path.join(SAVE_DIR, "vqvae_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model and plots saved in {SAVE_DIR}")

if __name__ == "__main__":
    main()