# Model Zoo

| Model | Kooplearn Implementation | Notes |
| --- | :---: | --- |
| DMD | {class}`kooplearn.models.DMD` | | 
| Extended DMD | {class}`kooplearn.models.ExtendedDMD` | | 
| Kernel DMD | {class}`kooplearn.models.KernelDMD` | | 
| Hankel DMD {footcite:p}`Arbabi2017` | {class}`kooplearn.models.DMD`<br>{class}`kooplearn.models.ExtendedDMD`<br>{class}`kooplearn.models.KernelDMD` | Hankel DMD with an history of $m$ steps is obtained by fitting these models with dataset of context length $m + 1$ |
| Reduced Rank DMD {footcite:p}`Kostic2022`|  {class}`kooplearn.models.DMD`<br>{class}`kooplearn.models.ExtendedDMD`<br>{class}`kooplearn.models.KernelDMD` | Reduced rank estimators are obtained by setting the `reduced_rank` flag to `True` at initialization |
| VAMPNets | {class}`kooplearn.models.DeepEDMD` + {class}`kooplearn.models.feature_maps.VAMPNet` | |

```{footbibliography}
```