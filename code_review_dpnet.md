# First Review (Pietro, answered by Bruno)
The code is far too complex to be putted in this form. Some comments:

1. There is no need to define an additional `DPNet` model, we can just use the `encoder_decoder.EncoderModel` class.
    - I agree that DPNet could just be the EncoderModel, but It was easier to define a new model to look at the bare 
      minimum that I would need to make the model work. I think it is easier to start from a concrete example and 
      then generalize to the EncoderModel.
2. The `DPNetFeatureMap` should be a subclass of both `lightning.LightningModule` and `kooplearn._src.encoding_decoding_utils.TrainableFeatureMap`. They are compatible.
    - I am actually not a big fan of multiple inheritance, but we can discuss this. Besides, for the moment I do not 
      see the need to define a FeatureMap and a TrainableFeatureMap.
3. The functionality currently contained in `DPNetModule` should be moved inside `DPNetFeatureMap`, which is currently only handling initialization.
   - A module in Pytorch Lighting is well-defined with its attributes and methods. I do not think that it is a good 
     idea to mix the functionality that we are developing with the functionality of the LightningModule. Someone 
     that looks at the code for the LightningModule should recognize the structure of a LightningModule.
4. I provide a minimal example what I explained above, feel free to add the additional kwargs needed (although I do not really like the long list of arguments which is currently needed to initialize `DPNetFeatureMap`, + there are some arguments with the suffix `_2`, which are unintelligible to me)
    - I agree that the list of arguments is long, but I think that it is necessary to have a flexible model. 
      Essentially, to train a deep learning model you need:
      - The model architecture (nn.Module)
      - Your optimizer (torch.optim)
      - Possibly a scheduler (torch.optim.lr_scheduler)
      - Data (Dataset + LightningDataModule if we want to automate the data loading/split of data)
      - Loss function
      - The training/validation/test loop (LightningModule)
      
      Besides, you need the code that will handle the interaction between all these components, which is the role of the
      Trainer. As we are using Pytorch Lightning we can also define callbacks that will add some functionalities to the
      training loop, like early stopping, logging, gradient clipping etc. Finally, if we want to log the training 
      results, metrics, hyperparameters etc, we should also define a logger (that we can also use elsewhere in the code).
      The trick that I used separating kwargs from the functions/classes is only because I know that it will be 
      easier to save everything in a file later. 
      We can always define suitable defaults for the arguments to make it easier to the user if he/she does not want 
      to bother with one of the parts, but as it is now, it is easy to test what would happen if for example instead 
      of using the Adam optimizer we use the SGD optimizer, or if we clip the gradient, or change the learning rate 
      following a scheduler etc.
      Finally, the arguments with the suffix `_2` are the arguments that I used to define the second model that 
      would encode the state at t+1. In general, we can use different models to encode the data at t and t+1, so for 
      me, it makes sense to be able to define 2 different models, we can discuss the details, but I think that this 
      is something that will become clear with some documentation.

# Some observations (Bruno)
1. Shouldn't the second regularization form (r + tr(C*log(C) - C)) be (r + tr(C*log(C.abs()) - C))? This way we 
   avoid  the `nan` values when C has negative values.