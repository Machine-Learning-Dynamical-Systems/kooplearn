(kooplearn_data_paradigm)=
# Kooplearn's data paradigm
In this guide we describe kooplearn's data paradigm, which is based on context windows. 

<p align = "center">
  <img src="../_static/_images/context_window_scheme.svg" alt="context-window-scheme" style="width:50%;"/>
</p>
<p align = "center"><em>A context window: the fundamental unit of data in kooplearn.  </em></p>

Kooplearn uses context windows as single _data points_, and expect (batches of) context windows to be provided to the fitting ({class}`kooplearn.abc.BaseModel.fit`, {class}`kooplearn.abc.TrainableFeatureMap.fit`) and inference ({class}`kooplearn.abc.BaseModel.predict`, {class}`kooplearn.abc.BaseModel.eig`, {class}`kooplearn.abc.BaseModel.modes`) methods.

A context window is a _fixed length_ sequence of observations of the system, and it may be further divided into two splits: the _lookback_ and _lookforward_ windows. The partitioning between lookback and lookforward slices is used to train models --- such as consistent Koopman auto-encoders {footcite:t}`Azencot2020CAE` --- which expect multiple-step forward/backward predictions at train time.

In kooplearn, the lookback window length is specified by the model, and stored in the attribute {class}`kooplearn.abc.BaseModel.lookback_len`, while the context length (and hence the lookforward window length) are inferenced at fitting time from the data shape. 

### Standard tensors shapes
Kooplearn expects tensors of shape `[batch_size, context_len, *features]`, where `features` can contain an arbitrary $\geq 1$ number of dimensions. 

Kooplearn **do not perform** shape inference, and will throw an error if tensors with less than $3$ dimensions are provided. This is quite important for dealing with single data points (or single features). Indeed, in a tensor of shape `[5, 10]` there is no way of tell if we are dealing with $5$ examples, a context length of $10$ and a single feature (which should be passed to kooplearn as a tensor of shape `[5, 10, 1]`) or with a single example, a context length of $5$ and $10$ features (passed in kooplearn as a tensor of shape `[1, 5, 10]`).

### Data at fitting/inference
Data pipelines in kooplearn are organized so that the look*back* slice of context windows can be accessed both at training and inference time. The look*forward* slice, on the other hand, can be accessed only at training time. {guilabel}`TODO - Add code snippet to show the data shapes at fitting and inference time`.

In kooplearn we expose several utility function to manipulate data and generate context-windows-based datasets. In this respect, see [the API reference](data_api).

```{footbibliography}
```