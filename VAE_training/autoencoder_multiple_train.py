import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, encode_layer=30, info_layer=1, quantum_number_later=1):
        super(Autoencoder, self).__init__()

        self.info_layer = info_layer
        self.encode_layer = encode_layer
        self.quantum_number_layer = quantum_number_later

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 60, kernel_size=3, padding=1),  # Adjust parameters as needed
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(60, 40, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(40, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Flatten(),
            nn.Linear(3*3*5*32, encode_layer),
            nn.ReLU()
        )

        self.encoder_info = nn.Sequential(
            nn.Linear(3, 8),  # Encode crucial information to the same size as the latent space
            nn.ReLU(),
            nn.Linear(8, info_layer)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encode_layer, 3*3*5*32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 3, 3,5)),
            nn.ConvTranspose3d(32, 40, kernel_size=(5,5,4), stride=2, padding=1),  # (32, 3, 3, 5) -> (40, 7, 7, 10)
            nn.ReLU(),
            nn.ConvTranspose3d(40, 60, kernel_size=(5,5,4), stride=2, padding=1),  # (40, 7, 7, 10) -> (60, 15, 15, 20)
            nn.ReLU(),
            nn.ConvTranspose3d(60, 1, kernel_size=4, stride=2, padding=1),  # (60, 15, 15, 20) -> (1, 30, 30, 40)
            nn.Sigmoid()  # Sigmoid for values between 0 and 1
        )

        self.decoder_info = nn.Sequential(
            nn.Linear(info_layer, 8),  # Decode to the size of crucial information
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x_image, x_info):

        ####### partition info layer
        reciprocal_lattice = x_info[...,:3]
        quantum_number = x_info[...,3:]
        ####### partition info layer

        x_image = self.encoder(x_image)
        reciprocal_lattice = self.encoder_info(reciprocal_lattice)

        # print('x_image.shape', x_image.shape)
        # print('x_info.shape', x_info.shape)
        self.combined_latent = torch.cat((quantum_number, reciprocal_lattice, x_image), dim=1)
        # print('combined_latent.shape', combined_latent.shape)

        ####### partition latent space
        info_latent = self.combined_latent[:, self.quantum_number_layer:self.quantum_number_layer+self.info_layer]
        image_latent = self.combined_latent[:,self.info_layer+self.quantum_number_layer:]


        decoded_info = self.decoder_info(info_latent)  # You can also decode the image-encoded info if needed
        decoded_image = self.decoder(image_latent)

        ###### merge info output
        decoded_info = torch.cat((decoded_info, quantum_number), dim=1)

        return decoded_image, decoded_info

    def get_latent_space(self):
        return self.combined_latent


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
        print(data.shape)

    ######### find states near Fermi Level
    energy_index = list(np.where((energy < 3) & (-3 < energy))[0])
    energy = energy[energy_index]
    data = data[energy_index]
    information = information[energy_index]
    information[...,-1] = energy
    ######### find states near Fermi Level

    # Normalize the data to values between 0 and 1
    data = data / data.max()

    # Convert the data to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)
    information = torch.tensor(information, dtype=torch.float32)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Split the data into training and testing sets
    train_data, test_data, train_info, test_info = train_test_split(data, information, test_size=0.2, random_state=42)


    # Create DataLoaders for batch processing
    batch_size = 128
    train_data_loader = DataLoader(TensorDataset(train_data, train_info), batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(TensorDataset(test_data, test_info), batch_size=batch_size, shuffle=False)


    autoencoder = Autoencoder().to(device)  # Move model to the selected device

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    criterion = criterion.to(device)

    num_epochs = 300

    # summary(autoencoder, input_size=(2, 30, 30))

    train_error = np.zeros(num_epochs)
    test_error = np.zeros(num_epochs)

    variance_data = np.var(data.numpy())
    # variance_info = np.var()
    for epoch in range(num_epochs):
        autoencoder.train()
        for batch, batch_info in train_data_loader:
            inputs = batch.to(device)  # Move batch to the selected device
            inputs_info = batch_info[...,9:].to(device)
            optimizer.zero_grad()
            outputs, outputs_info = autoencoder(inputs, inputs_info)
            loss_image = criterion(outputs, inputs)
            loss_info = criterion(outputs_info, inputs_info)
            loss = loss_info + loss_image
            loss.backward()
            optimizer.step()

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], R^2(train): {1 - (loss_image.item() / variance_data):.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], info error (train): {(   loss_info.item() ):.4f}")
        train_error[epoch] = loss_image.item()

        # Evaluate on the test set
        autoencoder.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch, batch_info in test_data_loader:
                inputs = batch.to(device)
                inputs_info = batch_info[...,9:].to(device)
                outputs, outputs_info = autoencoder(inputs, inputs_info)
                loss_image_test = criterion(outputs, inputs)
                loss_info_test = criterion(outputs_info, inputs_info)

                test_loss = loss_image_test + loss_info_test

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss / len(test_data_loader):.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], R^2(test): {1 - (loss_image_test.item() / variance_data):.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], info error (test): {loss_info_test:.4f}")
        test_error[epoch] = test_loss.item()

    print("Training complete.")

    torch.save(autoencoder.state_dict(), "autoencoder_multiple_model.pth")
