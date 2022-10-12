## Notes and implementation details.
1. In the spirit of RBF's feature map, we would like that $\phi: \mathcal{X} \to \mathbb{S}^{n}$.
   - To do so, we would like the distribution of the output values $\phi_{\sharp \mu}$ to be mean zero and normalized. Here $\mu$ is the distribution of the inputs.
   - We therefore use **batch normalization** in our NN architecture.
   - Finally, the last layer of the NN consists in a stereographic projection to map onto $\mathbb{S}^{n}$. Proposal: the square root of the softmax might do the job.
2. Next, we have to define the loss functions.
   - Alignment between the dominant eigenspaces of the kernel and of the Koopman operator. Is this equivalent to maximizing the VAMP-score?
   -  Forecasting-based objective functions.