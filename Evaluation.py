import numpy as np
import torch
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""Here I implement the functions for evaluation as described in paper"""


def one_hot(labels, class_size):
    """This function does one-hot encoding for labels"""
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


def reconstruct_sequences(model, test_sequences, test_labels):
    """Reconstructing sequences using the trained model based on its learned patterns during the training stage."""
    model.eval()  # set to inference mode
    with torch.no_grad():
        reconstructed_sequences = []
        for i in range(len(test_sequences)):
            seq = test_sequences[i].reshape(1, test_sequences.size(1), test_sequences.size(2))
            if test_labels != []:
                label = test_labels[i].reshape(1, test_labels.size(1))
                reconstructed_sequence, mu, logvar = model(seq, label)
            else:
                reconstructed_sequence, mu, logvar = model(seq)
            reconstructed_sequence = reconstructed_sequence[0]
            reconstructed_sequences.append(create_seq(reconstructed_sequence))
    return np.array(reconstructed_sequences)


def create_seq(reconstructed):
    """Convert a reconstructed sequence tensor into a nucleotide sequence."""
    nucleotide_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq = ''
    for i in range(len(reconstructed)):
        index = np.argmax(reconstructed[i].detach().numpy())
        seq += nucleotide_map[index]
    return seq


def calculate_seed_error(originals, reconstructed):
    """Calculating the percent of reconstructed sequences that have errors in the seed region."""
    percantage = 0
    for i in range(len(originals)):
        count = 0
        original_seed = originals[i][5:11]
        reconstructed_seed = reconstructed[i][5:11]
        for j in range(len(original_seed)):
            if original_seed[j] != reconstructed_seed[j]:
                count += 1
        if count >= 2:
            percantage += 1
    percantage /= len(originals)
    return percantage


def calculate_errors(model, test_sequences, test_labels):
    """Calculating the amount of errors in the reconstructed sequence and the percent of sequence with more than 10 errors."""
    recon = reconstruct_sequences(model, test_sequences, test_labels)
    test_seqs = [create_seq(seq) for seq in test_sequences]
    sum_errors = 0
    overall_mistakes_percantage = 0
    for i in range(len(test_seqs)):
        count = 0
        origin = test_seqs[i]
        for j in range(len(origin)):
            if origin[j] != recon[i][j]:
                count += 1
        if count >= 10:
            overall_mistakes_percantage += 1
        sum_errors += count
    avg_error_count = sum_errors / len(recon)
    overall_mistakes_percantage /= len(recon)
    seed_error_percantage = calculate_seed_error(test_seqs, recon)
    print(f"The average number of errors after reconstruction: {avg_error_count}")
    print(f"Percentage of sequences after reconstruction with more than 10 mistakes overall:"
          f" {overall_mistakes_percantage * 100}")
    print(f"Percentage of sequences after reconstruction with more than 2 mistakes in seed region:"
          f" {overall_mistakes_percantage * 100}")
    return avg_error_count, overall_mistakes_percantage, seed_error_percantage


def generate_sequences(model, class_label, num_sequences, latent_size, class_size):
    """Generating sequences using the trained model based on its learned patterns during the training stage."""
    model.eval()
    sequences = []
    labels = torch.tensor([class_label] * num_sequences).to(next(model.parameters()).device)
    encoded_labels = one_hot(labels, class_size)
    with torch.no_grad():
        for i in range(num_sequences):
            z = torch.randn(1, latent_size).to(next(model.parameters()).device)
            label = encoded_labels[i].reshape(1, encoded_labels.size(1))
            generated_seq = model.decode(z, label)
            generated_seq = create_seq(generated_seq[0])
            sequences.append(generated_seq)
    # creates the files for DeepCpf1
    with open(f'{class_label}_sequences.txt', 'w') as file:
        file.write("Target number + '\t + 34 bp target sequence (4 bp + PAM + 23 bp protospacer + 3 bp) + '\t' + "
                   "Chromatin accessibility (1= DNase I hypersensitive sites, 0 = Dnase I non-sensitive sites)")
        chromatin = '0' if class_label == 0 else '1'  # usually hypersensitive sites will indicate high efficiency
        # hence, 0-efficiency class will get 0 and 99-efficiency class will get 1
        for i in range(len(sequences)):
            new_seq = sequences[i][:4] + 'TTT' + sequences[i][4:]  # adding back the 'TTT' sequence that was removed
            file.write(str(i + 1) + '\t' + new_seq + '\t' + chromatin + '\n')


def extract_efficiency_scores(file):
    """Takes only the efficiency scores from a file."""
    efficiencies = []
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            lst = line.strip().split('\t')
            efficiencies.append(float(lst[-1]))
    return efficiencies


def hist_efficiency():
    """Create histogram of predicted efficiency scores from DeepCpf1."""
    zero_efficiencies = extract_efficiency_scores('zero_results.txt')
    ninetynine_efficiencies = extract_efficiency_scores('ninetynine_results.txt')
    plt.hist(zero_efficiencies, bins=30, alpha=0.7, color='blue', label='0-efficiency')
    plt.hist(ninetynine_efficiencies, bins=30, alpha=0.7, color='green', label='99-efficiency')
    plt.legend()
    plt.xlabel('Predicted Efficiency')
    plt.ylabel('Frequency')
    plt.title('Efficiency Prediction by seq-deepCpf1')
    plt.show()


if __name__ == '__main__':
    hist_efficiency()