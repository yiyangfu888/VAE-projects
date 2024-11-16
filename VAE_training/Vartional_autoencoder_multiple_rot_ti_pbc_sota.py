##warning!!!
##this code only works under gpu condition

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch.optim import Adam


class RotationalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1,padding_mode='circular'):
        super(RotationalConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode)

    def forward(self, x):
        # Apply convolution to original, 90, 180, and 270 degree rotated inputs
        x0 = self.conv(x)
        x90 = self.conv(torch.rot90(x, k=1, dims=[2, 3]))
        x180 = self.conv(torch.rot90(x, k=2, dims=[2, 3]))
        x270 = self.conv(torch.rot90(x, k=3, dims=[2, 3]))
        
        # Combine the results (you can choose different ways to combine, here we use max)
        x_combined = torch.max(torch.max(x0, x90), torch.max(x180, x270))
        
        return x_combined


    
class VAE(nn.Module):
    def __init__(self, latent_dim=60):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            RotationalConv3d(1, 40, kernel_size=3, padding=1, padding_mode='circular'), # Adjust parameters as needed
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            RotationalConv3d(40, 120, kernel_size=3, padding=1, padding_mode='circular'), # Adjust parameters as needed
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            RotationalConv3d(120, 320, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            # nn.Linear(392*4, latent_dim*2),
            nn.Linear(320, latent_dim*2),
            # nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3*3*5*40),
            nn.ReLU(),
            nn.Unflatten(1, (40, 3, 3, 5)),
            nn.ConvTranspose3d(40, 60, kernel_size=(5,5,4), stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(60, 80, kernel_size=(5,5,4), stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(80, 1, kernel_size=4, stride=2,padding=1),
            nn.Sigmoid()  # Sigmoid for values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # 0. Input size (preprocess)
        N, C, W, L, H= x.size()

        # Encode
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.decoder(z)


        # 4 .Up-sampling (adaptive postprocess)
        upsampling = nn.Upsample(size=(W, L, H))
        x = upsampling(x)
        
        return x, mu, logvar

    def get_latent_space(self, x):
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    # def get_latent_space(self):
    #     return self.latent_space
    def no_grad(self):
        for param in self.parameters():
            param.requires_grad = False
        return self

reconstruction_function = nn.MSELoss(reduction='sum')
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss
    BCE = reconstruction_function(recon_x, x)

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD


def train_vae(model, dataloader ,epochs=10, learning_rate=0.01, beta=1.0):
    model = model()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        train_loss = 0
        for batch_data in dataloader:
            # If your dataset provides data in the form (data, labels), use batch_data[0]
            # If not, just use batch_data
            data = batch_data[0].float()

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(dataloader.dataset)}")



if __name__ == "__main__":
    import h5py
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from torchsummary import summary

    # Load data from the HDF5 file
    file_path = "wfs_merged.h5"
    with h5py.File(file_path, "r") as file:
        data = file["data"][:]
        energy = file['energy'][:]
        information = file['information'][:]

    ######### find states near Fermi Level
    energy_index = list(np.where((energy < 4) & (-4 < energy))[0])
    energy = energy[energy_index]
    data = data[energy_index]
    information = information[energy_index]
    information[...,-1] = energy
    ######### find states near Fermi Level

    # Normalize the data to values between 0 and 1
    # data = data / data.max()

    max_image = np.max(data, axis=(2, 3, 4))
    data = data / max_image[:, np.newaxis, np.newaxis, np.newaxis]

    # Convert the data to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)
    print(type(data),data.dtype)
    information = torch.tensor(information, dtype=torch.float32)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # Create DataLoaders for batch processing
    batch_size = 128
    train_data_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False)

    # train_vae(VAE, train_data_loader)
    epochs= 400
    beta = 0.5

    model = VAE().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    model.train()
    train_error = np.zeros(epochs)
    test_error = np.zeros(epochs)
    variance_data = np.var(data.numpy())
    # test_var = np.var()

    criterion_image = nn.MSELoss()
    criterion_image = criterion_image.to(device)


    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for batch_data in train_data_loader:
            # If your dataset provides data in the form (data, labels), use batch_data[0]
            # If not, just use batch_data
            data = batch_data[0].to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta).to(device)
            loss_image = criterion_image(recon_batch, data)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch + 1}/{epochs}], R^2(train): {1 - (loss_image.item() / variance_data):.4f}")
        train_error[epoch] = loss_image.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_data in train_data_loader:
                data = batch_data[0].to(device)
                recon_batch, mu, logvar = model(data)
                loss_image = criterion_image(recon_batch, data)

        print(f"Epoch [{epoch + 1}/{epochs}], R^2(test): {1 - (loss_image.item() / variance_data):.4f}")

        test_error[epoch] = loss_image.item()

    print("Training complete.")
    torch.save(model.state_dict(), "VAE_model_rot.pth")