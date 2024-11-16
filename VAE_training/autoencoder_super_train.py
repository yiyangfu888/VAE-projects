import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


class Autoencoder_super(nn.Module):
    def __init__(self, latent_dim=10):
        super(Autoencoder_super, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 40, kernel_size=3, padding=1),  # Adjust parameters as needed
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(40, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(24, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Flatten(),
            nn.Linear(8*3*3*5, latent_dim*2),
            # nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3*3*5*24),
            nn.ReLU(),
            nn.Unflatten(1, (24, 3, 3,5)),
            nn.ConvTranspose3d(24, 32, kernel_size=(5,5,4), stride=2, padding=1),  # (24, 3, 3, 5) -> (32, 7, 7, 10)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 60, kernel_size=(5,5,4), stride=2, padding=1),  # (32, 7, 7, 10) -> (60, 15, 15, 20)
            nn.ReLU(),
            nn.ConvTranspose3d(60, 1, kernel_size=4, stride=2, padding=1),  # (60, 15, 15, 20) -> (1, 30, 30, 40)
            nn.Sigmoid()  # Sigmoid for values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        mu, logvar = x.chunk(2, dim=1)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        x = self.decoder(z)
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

if __name__ == "__main__":
    import h5py
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    # from torchinfo import summary

    # Load data from the HDF5 file
    file_path = "wfs_merged.h5"
    with h5py.File(file_path, "r") as file:
        data = file["super_band_unique"][:]
        # data = file["super_density"][:]
        # energy = file['energy'][:]

    data = np.vstack((data[:,0], data[:,1], data[:,2]))
    ######### find states near Fermi Level
    # energy_index = list(np.where((energy < 3) & (-3 < energy))[0])
    # energy = energy[energy_index]
    # data = data[energy_index]
    ######### find states near Fermi Level

    # Normalize the data to values between 0 and 1
    # data = data / data.max()
    max_image = np.max(data, axis=(2, 3, 4))
    data = data / max_image[:, np.newaxis, np.newaxis, np.newaxis]

    # Convert the data to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # Create DataLoaders for batch processing
    batch_size = 125
    train_data_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False)

    autoencoder = Autoencoder_super().to(device)  # Move model to the selected device

    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)



    criterion_image = nn.MSELoss()
    criterion_image = criterion_image.to(device)

    num_epochs = 3000
    beta = 0.5
    # summary(autoencoder, input_size=(2, 30, 30))

    train_error = np.zeros(num_epochs)
    test_error = np.zeros(num_epochs)

    variance_data = np.var(data.numpy())
    for epoch in range(num_epochs):
        train_loss = 0
        autoencoder.train()
        for batch in train_data_loader:
            data = batch[0].to(device)  # Move batch to the selected device
            optimizer.zero_grad()

            recon_batch, mu, logvar = autoencoder(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta).to(device)
            loss_image = criterion_image(recon_batch, data)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], R^2(train): {1 - (loss_image.item() / variance_data):.4f}")
        train_error[epoch] = loss_image.item()

        # Evaluate on the test set
        autoencoder.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_data_loader:
                data = batch[0].to(device)
                recon_batch, mu, logvar = autoencoder(data)
                loss_image = criterion_image(recon_batch, data)

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss / len(test_data_loader):.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], R^2(test): {1 - (loss_image.item() / variance_data):.4f}")
        test_error[epoch] = loss_image.item()

    print("Training complete.")

    torch.save(autoencoder.state_dict(), "autoencoder_model_super.pth")
