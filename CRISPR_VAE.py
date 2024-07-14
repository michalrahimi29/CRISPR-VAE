import torch.nn.functional as F
from torch import nn, optim
from DataLoader import train_set, test_creator
from HeatMap import *
from Evaluation import *

"""Implementation of CRISPR-VAE framework."""


class CVAE(nn.Module):
    def __init__(self, latent_size, class_size, input_size):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.class_size = class_size
        self.input_size = input_size
        self.features = 600

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(31, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(512 * input_size + class_size, self.features)
        self.fc_mu = nn.Linear(self.features, latent_size)
        self.fc_logvar = nn.Linear(self.features, latent_size)

        # Decoder layers
        self.fc2 = nn.Linear(latent_size + class_size, self.features)
        self.fc3 = nn.Linear(self.features, 512 * input_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 31, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        """Encoder network that maps input sequences and class information to latent space parameters."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, c], 1)  # Concatenate with class information
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z, c):
        """Decoder network that reconstructs sequences from latent space and class information."""
        z = torch.cat([z, c], 1)  # Concatenate with class information
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = z.view(z.size(0), -1, nucleotides_num)  # Reshape into feature maps
        x = self.decoder(z)
        return x

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from the latent space distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        """Forward pass through the CVAE."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, c)
        return reconstruction, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """Calculate the loss function for training the CVAE."""
        BCE = F.binary_cross_entropy(recon_x.view(-1, nucleotides_num * LEN), x.view(-1, nucleotides_num * LEN),
                                     reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def train(epoch):
    """Train the model and display the loss for every epoch."""
    model.train()
    total_loss = 0.0
    for batch_idx, (x_batch, efficiency_batch) in enumerate(train_dataset):
        x_batch = x_batch.float()
        efficiency_batch = one_hot(efficiency_batch, class_size)
        recon, mu, logvar = model(x_batch, efficiency_batch)
        optimizer.zero_grad()
        loss = model.loss_function(recon, x_batch, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print('Train Epoch {}, Loss: {}'.format(epoch, total_loss / len(train_dataset)))
    return total_loss / len(train_dataset)


def run_heatmap(test_sequences, test_labels):
    """Generate and display heat maps for latent vectors extracted from test sequences."""
    test_indeices = np.where(test_labels == 99)[0]
    label = one_hot(torch.tensor([99]), class_size)
    all_sequences = test_sequences[test_indeices].clone().detach()
    latent_vectors = extract_latent_vectors(model, all_sequences, label)
    run(latent_vectors)


if __name__ == '__main__':
    # Create a CVAE model
    latent_size = 10000  # Total number of latent variables
    class_size = 100
    nucleotides_num = 4
    LEN = 31
    model = CVAE(latent_size, class_size, nucleotides_num).to(device)
    num_epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = train_set()
    test_seqs, test_labels = test_creator()

    encoded_labels = one_hot(torch.tensor(test_labels), class_size).to(device)
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        
    avg_error = calculate_errors(model, test_seqs, encoded_labels)
    run_heatmap(test_seqs, np.array(test_labels))

    class_0_sequences = generate_sequences(model, 0, latent_size, latent_size, class_size)
    class_99_sequences = generate_sequences(model, 99, latent_size, latent_size, class_size)
