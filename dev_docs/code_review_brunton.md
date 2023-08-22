# Notes (Bruno)

1. Due to the particularities of the model (especially the way the auxiliary network is defined), I do not currently
   see how we could implement it with the same structure that we used for an Encoder(Decoder)Model. I have decided 
   to create a new model for it.
2. I am not sure about the methods `mode` and `eig` defined, someone must check it.
3. I do not know how to generalize for an arbitrary observable 
4. It seems odd to me that in the original implementation the eigenvalues are recalculate for each time step m times.
   For me, it would be more logical to calculate them once and only advance the system m times using the same 
   eigenvalues. We can discuss this and possibly change/compare the results. For the moment I have implemented it as 
   in the original code.
5. Define output shapes of predict/eig/mode
6. We need to add the sampling frequency (but maybe this is still needed for every model)