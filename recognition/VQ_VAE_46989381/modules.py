import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- Encoder ----------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=128, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(64)
        self.drop1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(128)
        self.drop2 = nn.Dropout2d(dropout)

        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(256)
        self.drop3 = nn.Dropout2d(dropout)

        self.conv4 = nn.Conv2d(256, latent_dim, 1, stride=1)

    def forward(self, x):
        x = F.relu(self.in1(self.conv1(x)))
        x = self.drop1(x)
        x = F.relu(self.in2(self.conv2(x)))
        x = self.drop2(x)
        x = F.relu(self.in3(self.conv3(x)))
        x = self.drop3(x)
        x = self.conv4(x)
        return x

# ---------------------- Decoder ----------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=1, dropout=0.1):
        super().__init__()
        # mirror encoder with upsampling
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(256)
        self.drop1 = nn.Dropout2d(dropout)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(128)
        self.drop2 = nn.Dropout2d(dropout)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(64)
        self.drop3 = nn.Dropout2d(dropout)

        self.out_conv = nn.Conv2d(64, output_channels, 1)

    def forward(self, z):
        x = F.relu(self.in1(self.deconv1(z)))
        x = self.drop1(x)
        x = F.relu(self.in2(self.deconv2(x)))
        x = self.drop2(x)
        x = F.relu(self.in3(self.deconv3(x)))
        x = self.drop3(x)
        x = torch.sigmoid(self.out_conv(x))
        return x

# ---------------------- Vector Quantizer ----------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, inputs):
        B, D, H, W = inputs.shape
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, D)

        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, self.embeddings.t())
            + torch.sum(self.embeddings**2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.embeddings)
        quantized = quantized.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices

# ---------------------- VQ-VAE ----------------------
class VQVAE(nn.Module):
    def __init__(self, latent_dim=128, num_embeddings=512, commitment_cost=0.25, output_channels=1):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim=latent_dim, output_channels=output_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _ = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss