<h1 align="center">
    CRISPR-VAE: A Method for Explaining CRISPR/Cas12a Predictions, <br>
    and an Efficiency-aware gRNA Sequence Generator
</h1>
CRISPR-VAE aims to enhance the interpretability of gRNA efficiency predictions by creating a structured latent space. This space places similar sequences in proximity, allowing for smooth transitions between different sequence phenomena and facilitating better understanding and optimization of gRNA designs.

## Overview
CRISPR-VAE employs a Variational Autoencoder (VAE) framework to create a structured latent space. The encoder projects sequences into a latent space using convolutional neural networks (CNNs) and dense layers, allowing for the generation of new sequences that bridge gaps in the data. CRISPR-VAE follows the conditional VAE paradigm, conditioning each sequence on its efficiency score. This approach creates separate latent spaces for different efficiency categories, focusing on the most distinct high (99-efficiency) and low (0-efficiency) sequences to highlight key features. The conditional VAE process involves two stages of information embedding. In the first stage, after the initial dense layer, one-hot-encoded efficiency information is blended with sequence information, embedding it into the latent space. In the second stage, after the code layer, the decoder operates as a standalone efficiency-aware sequence generator. This framework enhances the interpretability and efficiency-awareness of gRNA prediction, facilitating better understanding and optimization of CRISPR-Cas systems.
![image](https://github.com/michalrahimi29/CRISPR-VAE/assets/101520786/fb8f4da8-2947-484a-8db5-82d7b5e41e77)

## Requirements
The model is implemented with pyTorch 2.2.1 torchvision 0.17.1 and numpy 1.24.3.

## Run
For running the training and evaluation process of CRISPR-VAE as described in paper run the following code:
```python
python CRISPR_VAE.py 
```
For comparison with the VAE model run the folowing code:
```python
python VAE.py
```
In this study they used DeepCpf1 in order to predict the efficiency of the reconstructed sequences. To run DeepCpf1 you can either use the files in DeepCpf1 directory or run CRISPR_VAE.py to generate new sequences and copy them to the DeepCpf1 directory to predict their efficiency. To run DeepCpf1 see the following example where first input is the file input of the sequences you want to predict their efficiency (0_sequences.txt or 99_sequences.txt) and the second input is the name to file with predictions (for predicting the zero class sequences give the name zero_results.txt or for predicting the 99 class sequences give the name ninetynine_results.txt):
```python
python DeepCpf1/DeepCpf1.py 0_sequences.txt zero_results.txt
```
```python
python DeepCpf1/DeepCpf1.py 99_sequences.txt ninetynine_results.txt
```
