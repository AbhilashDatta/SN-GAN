# SN-GAN
Spectrally Normalized Generative Adversarial Networks (SN-GAN)

## Spectral Norm

A weight normalization technique to stabilize the training of the discriminator, as proposed in [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957) (Miyato et al. 2018).

## A Bit of History on Spectral Normalization

This isn't the first time that spectral norm has been proposed in the context of deep learning models. There's a paper called [Spectral Norm Regularization for Improving the Generalizability of Deep Learning](https://arxiv.org/abs/1705.10941) (Yoshida et al. 2017) that proposes **spectral norm regularization**, which they showed to improve the generalizability of models by adding extra loss terms onto the loss function (just as L2 regularization and gradient penalty do!). These extra loss terms specifically penalize the spectral norm of the weights. You can think of this as *data-independent* regularization because the gradient with respect to $W$ isn't a function of the minibatch.

**Spectral normalization**, on the other hand, sets the spectral norm of the weight matrices to 1 -- it's a much harder constraint than adding a loss term, which is a form of "soft" regularization. As the authors show in the paper, you can think of spectral normalization as *data-dependent* regularization, since the gradient with respect to $W$ is dependent on the mini-batch statistics (shown in Section 2.1 of the [main paper](https://arxiv.org/pdf/1802.05957.pdf)). Spectral normalization essentially prevents the transformation of each layer
from becoming to sensitive in one direction and mitigates exploding gradients.
