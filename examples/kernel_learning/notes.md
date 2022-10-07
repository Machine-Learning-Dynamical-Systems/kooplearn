## Notes and implementation details.
1. In the spirit of RBF's feature map, we would like that $\phi: \mathcal{X} \to \mathbb{S}^{n}$.
   - To do so, we would like the distribution of the output values $\phi_{\sharp \mu}$ to be mean zero and normalized. Here $\mu$ is the distribution of the inputs.
   - We therefore use **batch normalization** in our NN architecture.
   - Finally, the last layer of the NN consists in a stereographic projection to map onto $\mathbb{S}^{n}$.