import os
import torch
import matplotlib.pyplot as plt
from modules import VQVAE2
from dataset import load_our_data

# ---------------------- Configuration ----------------------
BASE_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
MODEL_PATH = "train_dir/vqvae_model.pth"
SAVE_DIR = "predict_dir"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Utilities ----------------------
def load_model(model_path, device):
    """
    Load a trained VQVAE2 model from the given file path.

    Args:
        model_path (str): Path to the saved model (.pth file).
        device (torch.device): Device to load the model onto (CPU or GPU).

    Returns:
        nn.Module: Loaded VQVAE2 model in evaluation mode.
    """
    print("Loading model...")
    model = VQVAE2(latent_dim=256, num_embeddings=1024, commitment_cost=0.25, output_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def get_sample(base_path, device):
    """
    Load a single test sample from the HipMRI dataset.

    Args:
        base_path (str): Path to the dataset directory.
        device (torch.device): Device to move the sample to.

    Returns:
        Tensor: Single input sample ready for prediction.
    """
    print("Loading data...")
    _, _, test_loader = load_our_data(base_path, batch_size=1, normImage=False)
    print("Test data loaded successfully.")
    sample_batch = next(iter(test_loader))
    x = sample_batch[0].to(device)  # assuming dataloader returns (images, labels)
    return x[:1]  # take one sample


def predict_and_reconstruct(model, x):
    """
    Perform a forward pass through the model to get reconstruction and optional latent indices.

    Args:
        model (nn.Module): Trained VQVAE2 model.
        x (Tensor): Input batch/sample.

    Returns:
        Tuple[Tensor, Optional[Tensor]]: Reconstructed image and latent codebook indices (if available).
    """
    with torch.no_grad():
        if hasattr(model, "encode") and hasattr(model, "decode"):
            z_e, z_q, indices = model.encode(x)
            recon = model.decode(z_q)
        else:
            output = model(x)
            if isinstance(output, tuple):
                recon = output[0]
            else:
                recon = output
            z_e, z_q, indices = None, None, None
    return recon, indices


def save_visualizations(x, recon, indices, save_dir):
    """
    Save visualizations of the original input, reconstruction, and optionally latent indices.

    Args:
        x (Tensor): Original input image.
        recon (Tensor): Reconstructed output image.
        indices (Optional[Tensor]): Latent codebook indices.
        save_dir (str): Directory to save the figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    x_np = x.squeeze().cpu().numpy()
    recon_np = recon.squeeze().cpu().numpy()

    # Original vs Reconstruction
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Input")
    plt.imshow(x_np, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(recon_np, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "reconstruction.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction figure to {save_path}")

    # Latent Codebook Indices (optional)
    if indices is not None:
        plt.figure(figsize=(5, 5))
        plt.title("Latent Codebook Indices")
        plt.imshow(indices.squeeze().cpu().numpy(), cmap='viridis')
        plt.axis("off")
        latent_path = os.path.join(save_dir, "latent_indices.png")
        plt.savefig(latent_path, bbox_inches="tight")
        plt.close()
        print(f"Saved latent index visualization to {latent_path}")

# ---------------------- Main ----------------------
def main():
    """
    Main script to load model, select a test sample, perform reconstruction, and save visualizations.
    """
    model = load_model(MODEL_PATH, DEVICE)
    x = get_sample(BASE_PATH, DEVICE)
    recon, indices = predict_and_reconstruct(model, x)
    save_visualizations(x, recon, indices, SAVE_DIR)
    print("Prediction complete.")


if __name__ == "__main__":
    main()