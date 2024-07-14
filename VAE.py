import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DataLoader import train_set, test_creator
from Evaluation import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""VAE framework, used for comparison to CVAE"""


class VAE(nn.Module):
    def __init__(self, latent_size, input_size):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.features = 600

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(31, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(512 * input_size, self.features)
        self.fc_mu = nn.Linear(self.features, latent_size)
        self.fc_logvar = nn.Linear(self.features, latent_size)

        # Decoder layers
        self.fc2 = nn.Linear(latent_size, self.features)
        self.fc3 = nn.Linear(self.features, 512 * input_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 31, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = z.view(z.size(0), -1, nucleotides_num)
        x = self.decoder(z)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, nucleotides_num * LEN), x.view(-1, nucleotides_num * LEN), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def train(epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (x_batch, efficiency_batch) in enumerate(train_dataset):
        x_batch = x_batch.float()
        recon, mu, logvar = model(x_batch)
        optimizer.zero_grad()
        loss = model.loss_function(recon, x_batch, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print('Train Epoch {}, Loss: {}'.format(epoch, total_loss / len(train_dataset)))
    return total_loss / len(train_dataset)


if __name__ == '__main__':
    # Create a VAE model
    latent_size = 10000  # Total number of latent variables
    nucleotides_num = 4
    LEN = 31
    model = VAE(latent_size, nucleotides_num).to(device)
    num_epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = train_set()
    test_seqs, test_labels = test_creator()

    for epoch in range(1, num_epochs + 1):
        train(epoch)
    avg_error = calculate_errors(model, test_seqs, [])
