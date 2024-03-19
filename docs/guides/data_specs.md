(kooplearn_data_paradigm)=
# Kooplearn's data paradigm

> _Author:_ Pietro Novelli — [@pie_novelli](https://twitter.com/pie_novelli)

This guide describes kooplearn's data paradigm, which is based on context windows. 

<p align = "center">
  <img src="../_static/_images/context_window_scheme.svg" alt="context-window-scheme" style="width:50%;"/>
</p>
<p align = "center"><em>A context window: the fundamental unit of data in kooplearn.  </em></p>

Kooplearn uses context windows as single _data points_, and expect (batches of) context windows to be provided to the fitting ({class}`kooplearn.abc.BaseModel.fit`, {class}`kooplearn.abc.TrainableFeatureMap.fit`) and inference ({class}`kooplearn.abc.BaseModel.predict`, {class}`kooplearn.abc.BaseModel.eig`, {class}`kooplearn.abc.BaseModel.modes`) methods.

A context window, whose abstract implementation can be found in the class {class}`kooplearn.abc.ContextWindow`, is a _fixed length_ sequence of observations of the system, and it may be further divided into two splits: the _lookback_ and _lookforward_ windows. In kooplearn this is as easy as calling `context_window.lookback(lookback_length)` and `context_window.lookforward(lookback_length)` on an initialized `context_window` object. The partitioning between lookback and lookforward windows is defined by the kind of model we are using. For example, in models such as {class}`kooplearn.models.Nonlinear` or {class}`kooplearn.models.Kernel`, the lookforward window is _always_ of length 1, and the states in the lookback window are stacked to create an augmented state of delay embeddings, used e.g. in Hankel DMD by {footcite:t}`Arbabi2017`. Conversely, consistent Koopman auto-encoders {footcite:t}`Azencot2020CAE` expect multiple-step forward/backward evolutions at train time, and the lookback and lookforward windows are used to provide the ground truth values for these evolutions.

In kooplearn, the lookback window length is specified by the model, and stored in the attribute {class}`kooplearn.abc.BaseModel.lookback_len`, while the context length (and hence the lookforward window length) are defined in the attribute {class}`kooplearn.abc.ContextWindow.context_length`. 

### Tutorial & Further References

Working with context windows in kooplearn is explained in the example [Working with Context Windows](context_windows_tutorial). To check out the full documentation of the context window objects, see [the API reference](data_api). 

```{footbibliography}
```