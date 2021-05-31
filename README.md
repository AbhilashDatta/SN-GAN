# SN-GAN
Spectrally Normalized Generative Adversarial Networks (SN-GAN)

## Spectral Norm

Notationally, the spectral norm of a matrix $W$ is typically represented as $\sigma(W)$. For neural network purposes, this $W$ matrix represents a weight matrix in one of the network's layers. The spectral norm of a matrix is the matrix's largest singular value, which can be obtained via singular value decomposition (SVD).

**A Quick Refresher on SVD**

SVD is a generalization of [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) and is used to factorize a matrix as $W = U\Sigma V^\top$, where $U, V$ are orthogonal matrices and $\Sigma$ is a matrix of singular values on its diagonal. Note that $\Sigma$ doesn't have to be square.

\begin{align*}
    \Sigma = \begin{bmatrix}\sigma_1 & & \\ & \sigma_2 \\ & & \ddots \\ & & & \sigma_n\end{bmatrix}
\end{align*}

where $\sigma_1$ and $\sigma_n$ are the largest and smallest singular values, respectively. Intuitively, larger values correspond to larger amounts of stretching a matrix can apply to another vector. Following this notation, $\sigma(W) = \sigma_1$.

**Applying SVD to Spectral Normalization**

To spectrally normalize the weight matrix, you divide every value in the matrix by its spectral norm. As a result, a spectrally normalized matrix $\overline{W}_{SN}$ can be expressed as

\begin{align*}
  \overline{W}_{SN} = \dfrac{W}{\sigma(W)},
\end{align*}

In practice, computing the SVD of $W$ is expensive, so the authors of the SN-GAN paper do something very neat. They instead approximate the left and right singular vectors, $\tilde{u}$ and $\tilde{v}$ respectively, through power iteration such that $\sigma(W) \approx \tilde{u}^\top W\tilde{v}$.

Starting from randomly initialization, $\tilde{u}$ and $\tilde{v}$ are updated according to

\begin{align*}
  \tilde{u} &:= \dfrac{W^\top\tilde{u}}{||W^\top\tilde{u}||_2} \\
  \tilde{v} &:= \dfrac{W\tilde{v}}{||W\tilde{v}||_2}
\end{align*}

In practice, one round of iteration is sufficient to "achieve satisfactory performance" as per the authors.

Don't worry if you don't completely follow this! The algorithm is conveniently implemented as `torch.nn.utils.spectral_norm` in PyTorch, so as long as you get the general gist of how it might be useful and when to use it, then you're all set.
