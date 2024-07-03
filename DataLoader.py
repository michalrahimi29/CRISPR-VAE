import numpy as np
import pandas as pd
from Dataset import *
from torch.utils.data import DataLoader

batch_size = 256


def oneHot(string, SIZE=4):
    """One-hot encoding for DNA sequence"""
    trantab = str.maketrans('ACGT', '0123')
    string = str(string)
    data = [int(x) for x in list(string.translate(trantab))]
    ret = torch.eye(SIZE)[torch.tensor(data)]
    return ret


def load(file):
    """Extract from file only the sequences of 99-efficiency and 0-efficiency"""
    a = pd.read_csv(file)
    seqs = a['34 bp synthetic target and target context sequence(4 bp + PAM + 23 bp protospacer + 3 bp)'].to_numpy()
    labels = a['Indel freqeuncy(Background substracted, %)'].to_numpy().astype(int)
    labels[labels == 100] = 99  # 99-efficiency class includes the sequences labeled 99 <= x <=100
    labels[labels <= -1] = 0   # changing the labels lower than zero to zero
    zero_efficiency_indexes = np.where(labels <= 1)[0]
    labels[labels == 1] = 0  # 0-efficiency class includes the sequences labeled <= 1
    ninety_nine_efficiency_indexes = np.where(labels >= 99)[0]
    indexes = np.concatenate((zero_efficiency_indexes, ninety_nine_efficiency_indexes), axis=0)
    return seqs[indexes], labels[indexes]


def change_double_nucleotide(sequence):
    """Creates two changes in promiscuous in non-adjacent indexes. Only for 99-efficiency sequences"""
    loc = [(26, 28), (26, 29), (26, 30), (27, 29), (27, 30), (28, 30)]
    bases = ['A', 'C', 'G', 'T']
    new_seqs = []
    for l in loc:
        new_seq = list(sequence)
        first_bases = bases.copy()
        first_bases.remove(sequence[l[0]])
        second_bases = bases.copy()
        second_bases.remove(sequence[l[1]])
        for nuc_first in first_bases:
            for nuc_second in second_bases:
                new_seq[l[0]] = nuc_first
                new_seq[l[1]] = nuc_second
                new_seqs.append((''.join(new_seq), 99))
    return new_seqs


def dimer_change(sequence, label):
    """Creates two changes in promiscuous in adjacent indexes. Only for 99-efficiency sequences"""
    dimers = ['AT', 'AA', 'AC', 'AG', 'CC', 'CA', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    start, end = 26, 30
    new_seqs = []
    sequence_list = list(sequence)
    for i in range(start, end):
        for dimer in dimers:
            # validation the both nucleotides will be changed
            if sequence[i] != dimer[0] and sequence[i + 1] != dimer[1]:
                sequence_list[start:start + 2] = list(dimer)
                seq = ''.join(sequence_list)
                new_seqs.append((seq, label))
    return new_seqs


def augment_sequence(sequence, label):
    """Apply augmentation for both classes: 0-efficiency and 99-efficiency"""
    start, end = 26, 31
    new_seqs = [(sequence, label)]
    bases = ['A', 'C', 'G', 'T']
    # change only one nucleotide
    for i in range(start, end):
        sequence_list = list(sequence)
        curr = sequence[i]
        for base in bases:
            if base != curr:
                sequence_list[i] = base
                augmented_sequence = ''.join(sequence_list)
                new_seqs.append((augmented_sequence, label))
    # this part will change two nucleotides for every sequence, only for 99-efficiency class due to unbalanced data
    if label == 99:
        dimers_sequences = dimer_change(sequence, label)
        new_seqs.extend(dimers_sequences)
        randoms = change_double_nucleotide(sequence)
        new_seqs.extend(randoms)
    return new_seqs


def train_creater():
    """Return the sequences and corresponding labels to train CRISPR-VAE"""
    dataset = []
    seqs, labels = load('train.csv')
    ninetynine_indices = np.where(labels == 99)[0]
    zero_indices = np.where(labels == 0)[0]
    zero_sequences = seqs[zero_indices]
    ninetynine_sequences = seqs[ninetynine_indices]
    for i in range(len(ninetynine_sequences)):
        augmentation = augment_sequence(ninetynine_sequences[i], 99)[:69]
        dataset.extend(augmentation)
    for i in range(len(zero_sequences)):
        augmentation = augment_sequence(zero_sequences[i], 0)
        dataset.extend(augmentation)
    sequences, labels = zip(*dataset)
    # removal of 'TTT' sequence in PAM
    start_index, length = 4, 3
    seqs = [s[:start_index] + s[start_index + length:] for s in sequences]
    encoded_seqs = torch.stack([oneHot(seq) for seq in seqs])
    return encoded_seqs, labels


def test_creator():
    """Return the sequences and corresponding labels to evaluate CRISPR-VAE"""
    sequences, labels = load('test.csv')
    start_index, length = 4, 3
    seqs = [s[:start_index] + s[start_index + length:] for s in sequences]
    encoded_seqs = torch.stack([oneHot(seq) for seq in seqs])
    return encoded_seqs, labels


def train_set():
    train_sequences, train_labels = train_creater()
    train_dataset = DNADataset(train_sequences, train_labels)
    training_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return training_dataset

