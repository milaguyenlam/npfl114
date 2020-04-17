#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net

import tensorflow.keras.layers as layers

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=24, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--hidden_layers", default="512:relu", type=str)
    parser.add_argument("--l2", default=0.0001, type=float)
    parser.add_argument("--drop", default=0.3, type=float)
    parser.add_argument("--opt", default="adam:0.001", type=str, help='adam|rms:rate')
    parser.add_argument("--fine", default=False, type=bool, help="Finetuning.")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.hidden_layers = [(int(hidden_layer.split(':')[0]), hidden_layer.split(':')[1])
                          for hidden_layer in args.hidden_layers.split(",") if hidden_layer]
    args.opt = tuple([args.opt.split(':')[0], float(args.opt.split(':')[1])])

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)


    def train_augment(e):
        image = e["image"]
        label = e["label"]
        if tf.random.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 12, CAGS.W + 12)
        image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 24, dtype=tf.int32),
                                        tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 24, dtype=tf.int32)])
        image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])
        return image, label


    train = cags.train.map(CAGS.parse)
    train = train.map(train_augment)
    train.shuffle(10000, seed=args.seed)
    train = train.batch(args.batch_size)

    dev = cags.dev.map(CAGS.parse)
    dev = dev.map(lambda example: (example["image"], example["label"]))
    dev = dev.batch(args.batch_size)

    test = cags.test.map(CAGS.parse)
    test = test.map(lambda example: example["image"])
    test = test.batch(args.batch_size)

    # TODO: Create the model and train it
    if not args.fine:
        for layer in efficientnet_b0.layers:
            layer.trainable = False

    hidden = efficientnet_b0.outputs[0]

    for h, a in args.hidden_layers:
        hidden = layers.Dense(h, a, kernel_regularizer=tf.keras.regularizers.l2(args.l2))(hidden)
        hidden = layers.Dropout(args.drop)(hidden)

    outputs = layers.Dense(len(CAGS.LABELS), tf.nn.softmax)(hidden)

    model = tf.keras.Model(efficientnet_b0.inputs, outputs)

    if args.fine:
        model.load_weights("M-pre.h5")

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(args.opt[1]) if args.opt[1] == 'rms' else tf.keras.optimizers.Adam(
            args.opt[1]),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    model.fit(train, epochs=args.epochs,
              validation_data=dev,
              callbacks=[tb_callback],
              )
    model.save(os.path.join(args.logdir, f"M-{'fine' if args.fine else 'pre'}.h5"))

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as out_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test)
        for probs in test_probabilities:
            print(np.argmax(probs), file=out_file)
