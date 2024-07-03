import torch
from torch.utils.data import Dataset


class DNADataset(Dataset):
    """
       This class creates datasets of sequences with their labels.

       Attributes:
       -----------
       sequences : list
           A list of DNA sequences, where each sequence is represented as a list or array of integers.
       labels : list
           A list of labels corresponding to each DNA sequence.

       Methods:
       --------
       __len__():
           Returns the total number of sequences in the dataset.
       __getitem__(idx):
           Returns the sequence and label at the specified index, converted to PyTorch tensors.
       """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # Convert sequence and label to tensors
        sequence_tensor = torch.tensor(sequence, dtype=torch.int32)  # Assuming float32 tensor
        label_tensor = torch.tensor(label, dtype=torch.long)  # Assuming long tensor for classification labels
        return sequence_tensor, label_tensor