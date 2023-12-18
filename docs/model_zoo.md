(model_zoo)=
# Model Zoo 

| Model | Kooplearn Implementation | Notes |
| --- | :---: | --- |
| DMD | {class}`kooplearn.models.DMD` | | 
| Extended DMD | {class}`kooplearn.models.ExtendedDMD` | | 
| Kernel DMD | {class}`kooplearn.models.KernelDMD` | | 
| Nyström Kernel DMD{footcite:p}`Meanti2023` | {class}`kooplearn.models.NystromKernelDMD` | | 
| Hankel DMD {footcite:p}`Arbabi2017` | {class}`kooplearn.models.DMD`<br>{class}`kooplearn.models.ExtendedDMD`<br>{class}`kooplearn.models.KernelDMD` | Hankel DMD with an history of $m$ steps is obtained by fitting these models with dataset of context length $m + 1$ |
| Reduced Rank DMD {footcite:p}`Kostic2022`|  {class}`kooplearn.models.DMD`<br>{class}`kooplearn.models.ExtendedDMD`<br>{class}`kooplearn.models.KernelDMD` | Reduced rank estimators are obtained by setting the `reduced_rank` flag to `True` at initialization |
| VAMPNets{footcite:p}`Mardt2018, Wu2019` | {class}`kooplearn.models.DeepEDMD` + {class}`kooplearn.models.feature_maps.VAMPNet` | |
| Dynamic AutoEncoder{footcite:p}`Lusch2018, Morton2018` | {class}`kooplearn.models.DynamicAE` | When `use_lstsq_for_evolution == True`, the linear evolution of the embedded state is given by a least square model as in {footcite:t}`Morton2018`.|
| Consistent AutoEncoder{footcite:p}`Azencot2020CAE` | {class}`kooplearn.models.ConsistentAE` | |

```{footbibliography}
```