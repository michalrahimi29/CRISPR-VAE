import matplotlib.pyplot as plt
import numpy as np
import torch

delta_values = [3, 11, 29]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""In this file we used distance heat maps to test the structure of the latent space."""


def calculate_hamming_distance(vec1, vec2):
    """Calculate the hamming distance between two vectors."""
    return np.sum(vec1 != vec2)


def H(a, b, delta):
    """Calculate Hamming distance and check if it equals delta."""
    return 1 if calculate_hamming_distance(a, b) == delta else 0


def create_heat_map(latent_vectors, delta, L=10):
    """Create a heat map for latent vectors with a specific delta value."""
    num_sequences = latent_vectors.shape[0]
    heat_map = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(num_sequences):
            if i != j:
                a_ij = latent_vectors[i]
                # L is the number of seeds in the latent space that satisfy the Minkowski distance condition
                b_l_candidates = latent_vectors[np.random.choice(num_sequences, L, replace=False)]
                H_sum = 0
                for b_l in b_l_candidates:
                    if calculate_hamming_distance(a_ij, b_l) == delta:
                        H_sum += H(a_ij, b_l, delta)
                heat_map[i, j] = H_sum / L

    return heat_map


def extract_latent_vectors(model, sequences, label):
    """Extract latent vectors from sequences using the model."""
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for sequence in sequences:
            sequence = sequence.reshape(1, sequences.size(1), sequences.size(2))
            mu, logvar = model.encode(sequence, label)
            z = model.reparameterize(mu, logvar)
            latent_vectors.append(z.cpu().numpy()[0])
    latent_vectors = np.array(latent_vectors)
    return latent_vectors


def run(latent_vectors):
    """Display heat maps for different delta values."""
    heat_maps = {}
    # delta indicates the distance threshold for the heat map construction.
    for delta in delta_values:
        heat_maps[delta] = create_heat_map(latent_vectors, delta)
    fig, axes = plt.subplots(1, len(delta_values), figsize=(15, 5))
    for i, delta in enumerate(delta_values):
        ax = axes[i]
        cax = ax.matshow(heat_maps[delta], cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f"Heat Map (δ = {delta})")
    plt.show()
