The code is far too complex to be putted in this form. Some comments:

1. There is no need to define an additional `DPNet` model, we can just use the `encoder_decoder.EncoderModel` class.
2. The `DPNetFeatureMap` should be a subclass of both `lightning.LightningModule` and `kooplearn._src.encoding_decoding_utils.TrainableFeatureMap`. They are compatible.
3. The functionality currently contained in `DPNetModule` should be moved inside `DPNetFeatureMap`, which is currently only handling initialization.
4. I provide a minimal example what I explained above, feel free to add the additional kwargs needed (although I do not really like the long list of arguments which is currently needed to initialize `DPNetFeatureMap`, + there are some arguments with the suffix `_2`, which are unintelligible to me)