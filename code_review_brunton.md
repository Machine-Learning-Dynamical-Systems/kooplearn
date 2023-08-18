# Notes (Bruno)

1. Due to the particularities of the model (especially the way the auxiliary network is defined), I do not currently
   see how we could implement it with the same structure that we used for an Encoder(Decoder)Model. I have decided 
   to create a new model for it.
2. I am not sure about the methods `mode` and `eig` defined, someone must check it.
3. I do not know how to generalize for an arbitrary observable 