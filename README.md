<h1 align="center">
    CRISPR-VAE: A Method for Explaining CRISPR/Cas12a Predictions, <br>
    and an Efficiency-aware gRNA Sequence Generator
</h1>
CRISPR-VAE aims to enhance the interpretability of gRNA efficiency predictions by creating a structured latent space. This space places similar sequences in proximity, allowing for smooth transitions between different sequence phenomena and facilitating better understanding and optimization of gRNA designs.

## Framework
CRISPR-VAE utilizes a Variational Autoencoder (VAE) framework to establish a structured latent space. The encoder projects sequences into a latent space, implemented using convolutional neural networks (CNNs) and dense layers. Sampling from the latent space allows the generation of new sequences that bridge these gaps. In this study, 10,000 latent codes arranged in a grid of 100x100 were sampled and decoded into synthetic sequences. CRISPR-VAE follows the conditional VAE paradigm inspired by~\cite{sohn2015learning}, conditioning each sequence on its efficiency score to creates separate latent spaces for different efficiency categories, with a focus on the most distinct high (99-efficiency) and low (0-efficiency) sequences to highlight key features.
The conditional VAE process is shown in Figure~\ref{fig:model}, where one-hot-encoded efficiency information is fed to the network at two concatenation stages. The first stage, after the initial dense layer, blends efficiency information with sequence information, embedding it into the latent space. The second stage, after the code layer, allows the decoder to be a standalone efficiency aware sequence generator.
![image](https://github.com/michalrahimi29/CRISPR-VAE/assets/101520786/fb8f4da8-2947-484a-8db5-82d7b5e41e77)
