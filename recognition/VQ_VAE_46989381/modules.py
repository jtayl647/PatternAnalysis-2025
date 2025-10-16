import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- Residual Block ----------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return F.relu(out + x)

# ---------------------- Encoder ----------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(64)
        self.drop1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(128)
        self.drop2 = nn.Dropout2d(dropout)
        self.res1 = ResidualBlock(128)

        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(256)
        self.drop3 = nn.Dropout2d(dropout)
        self.res2 = ResidualBlock(256)

        self.conv4 = nn.Conv2d(256, latent_dim, 1, stride=1)

    def forward(self, x):
        x = F.relu(self.in1(self.conv1(x)))
        x = self.drop1(x)
        x = F.relu(self.in2(self.conv2(x)))
        x = self.drop2(x)
        x = self.res1(x)
        x = F.relu(self.in3(self.conv3(x)))
        x = self.drop3(x)
        x = self.res2(x)
        x = self.conv4(x)
        return x

# ---------------------- Decoder ----------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=1, dropout=0.1):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(256)
        self.drop1 = nn.Dropout2d(dropout)
        self.res1 = ResidualBlock(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(128)
        self.drop2 = nn.Dropout2d(dropout)
        self.res2 = ResidualBlock(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(64)
        self.drop3 = nn.Dropout2d(dropout)

        self.refine = nn.Conv2d(64, 64, 3, padding=1)
        self.in_refine = nn.InstanceNorm2d(64)
        self.drop_refine = nn.Dropout2d(dropout)

        self.out_conv = nn.Conv2d(64, output_channels, 1)

    def forward(self, z):
        x = F.relu(self.in1(self.deconv1(z)))
        x = self.drop1(x)
        x = self.res1(x)
        x = F.relu(self.in2(self.deconv2(x)))
        x = self.drop2(x)
        x = self.res2(x)
        x = F.relu(self.in3(self.deconv3(x)))
        x = self.drop3(x)
        x = F.relu(self.in_refine(self.refine(x)))
        x = self.drop_refine(x)
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

# ---------------------- Improved VQ-VAE-2 ----------------------
class VQVAE2(nn.Module):
    def __init__(self, latent_dim=128, num_embeddings=512, commitment_cost=0.25, output_channels=1):
        super().__init__()
        # Bottom-level encoder
        self.encoder_b = Encoder(in_channels=1, latent_dim=latent_dim)
        
        # Top-level encoder
        self.encoder_t = nn.Sequential(
            nn.AvgPool2d(2),
            Encoder(in_channels=latent_dim, latent_dim=latent_dim)
        )
        
        # Vector quantizers
        self.vq_top = VectorQuantizer(num_embeddings, embedding_dim=latent_dim, commitment_cost=commitment_cost)
        self.vq_bottom = VectorQuantizer(num_embeddings, embedding_dim=latent_dim*2, commitment_cost=commitment_cost)

        # Residual after concatenation
        self.res_after_concat = ResidualBlock(latent_dim*2)

        # Decoder
        self.decoder = Decoder(latent_dim=latent_dim*2, output_channels=output_channels)
    
    def forward(self, x):
        # Bottom latent
        z_b = self.encoder_b(x)
        
        # Top latent
        z_t = self.encoder_t(z_b)
        
        # Top quantization
        z_t_q, vq_loss_t, _ = self.vq_top(z_t)
        
        # Upsample top latent
        z_t_q_up = F.interpolate(z_t_q, size=z_b.shape[-2:], mode='nearest')
        
        # Combine bottom latent with top latent
        z_b_combined = torch.cat([z_b, z_t_q_up], dim=1)
        z_b_combined = self.res_after_concat(z_b_combined)
        
        # Bottom quantization
        z_b_q, vq_loss_b, _ = self.vq_bottom(z_b_combined)
        
        # Decode
        x_recon = self.decoder(z_b_q)
        
        # Total VQ loss
        vq_loss = vq_loss_t + vq_loss_b
        return x_recon, vq_loss