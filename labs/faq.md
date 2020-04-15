### `tf.data.Dataset`

- _How to look what is in a `tf.data.Dataset`?_

  The `tf.data.Dataset` is not just an array, but a description of a pipeline,
  which can produce data if requested. A simple way to run the pipeline is
  to iterate it using Python iterators:
  ```python
  dataset = tf.data.Dataset.range(10)
  for entry in dataset:
      print(entry)
  ```

- _How to use `tf.data.Dataset` with `model.fit` or `model.evaluate`?_

  To use a `tf.data.Dataset` in Keras, the dataset elements should be pairs
  `(input_data, gold_labels)`, where `input_data` and `gold_labels` must be
  already batched. For example, given `CAGS` dataset, you can preprocess
  training data for `cags_classification` as (for development data, you would
  remove the `.shuffle`):
  ```python
  train = cags.train.map(CAGS.parse)
  train = train.map(lambda example: (example["image"], example["label"]))
  train = train.shuffle(10000, seed=args.seed)
  train = train.batch(args.batch_size)
  ```

- _Is every iteration through a `tf.data.Dataset` the same?_

  No. Because the dataset is only a _pipeline_ generating data, it is called
  each time the dataset is iterated – therefore, every `.shuffle` is called
  in every iteration.

  Similarly, if you use random numbers in a `augment` method and use it in
  a `.map(augment)`, it is called on each iteration and can modify the same image
  differently in different epochs.

  ```python
  data = tf.data.Dataset.from_tensor_slices(tf.zeros(10, tf.int32))
  data = data.map(lambda x: x + tf.random.uniform([], maxval=10, dtype=tf.int32))
  for _ in range(3):
      print(*[element.numpy() for element in data])
  ```

### Finetuning

- _How to make a part of the network frozen, so that its weights are not updated?_

  Each `tf.keras.layers.Layer`/`tf.keras.Model` has a mutable `trainable`
  property indicating whether its variables should be updated – however, after
  changing it, you need to call `.compile` again (or otherwise make sure the
  list of trainable variables for the optimizer is updated).

  Note that once `trainable == False`, the insides of a layer are no longer
  considered, even if some its sub-layers have `trainable == True`. Therefore, if
  you want to freeze only some sub-layers of a layer you use in your model, the
  layer itself must have `trainable == True`.

- _How to choose whether dropout/batch normalization is executed in training or
  inference regime?_

  When calling a `tf.keras.layers.Layer`/`tf.keras.Model`, a named option
  `training` can be specified, indicating whether training or inference regime
  should be used. For a model, this option is automatically passed to its layers
  which require it, and Keras automatically passes it during
  `model.{fit,evaluate,predict}`.

  However, you can manually pass for example `training=False` to a layer when
  using Functional API, meaning that layer is executed in the inference
  regime even when the whole model is training.

- _How does `trainable` and `training` interact?_

  The only layer, which is influenced by both these options, is batch
  normalization, for which:
  - if `trainable == False`, the layer is always executed in inference regime;
  - if `trainable == True`, the training/inference regime is chosen according
    to the `training` option.
