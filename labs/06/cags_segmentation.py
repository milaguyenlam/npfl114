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

from cags_segmentation_eval import CAGSMaskIoU

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

class Network(tf.keras.Model):
    def __init__(self, args):
        efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)

        hidden = efficientnet_b0.outputs

        # MASK LAYER
        # Pyramid block 1
        #bn1 = layers.BatchNormalization()(hidden[1])
        #conv1 = layers.Conv2D(filters=64, kernel_size=(3,3), strides = 1, padding='same', use_bias=False, activation = tf.nn.relu)(bn1)
        #bn2 = layers.BatchNormalization()(conv1)
        #conv2 = layers.Conv2D(filters=64, kernel_size=(3,3), strides = 1, padding='same', use_bias=False, activation = tf.nn.relu)(bn2)
        #shortcut1 = layers.Conv2D(filters=64, kernel_size=(1,1), strides = 1, padding='same', use_bias=False, activation = tf.nn.relu)(bn1)
        #out1 = layers.add([conv2, shortcut1])

        # Pyramid block 2
        #bn3 = layers.BatchNormalization()(out1)
        #conv3 = layers.Conv2D(filters=32, kernel_size=(3,3), strides = 1, padding='same', use_bias=False, activation = tf.nn.relu)(bn3)
        #bn4 = layers.BatchNormalization()(conv3)
        #conv4 = layers.Conv2D(filters=32, kernel_size=(3,3), strides = 1, padding='same', use_bias=False, activation = tf.nn.relu)(bn4)
        #shortcut2 = layers.Conv2D(filters=32, kernel_size=(1,1), strides = 1, padding='same', use_bias=False, activation = tf.nn.relu)(bn3)
        #out2 = layers.add([conv4, shortcut2])

        #bn5 = layers.BatchNormalization()(out2)
        #conv5 = layers.Conv2D(filters=64, kernel_size=(3,3), strides = 1, padding='same', use_bias=False, activation = tf.nn.relu)(bn5)
        #bn6 = layers.BatchNormalization()(conv5)
        #conv6 = layers.Conv2D(filters=64, kernel_size=(3,3), strides = 1, padding='same', use_bias=False, activation = tf.nn.relu)(bn6)
        #shortcut3 = layers.Conv2D(filters=64, kernel_size=(1,1), strides = 1, padding='same', use_bias=False, activation = tf.nn.relu)(bn5)
        #out3 = layers.add([conv6, shortcut3])
        #tmp = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(out3)
        #tmp = layers.BatchNormalization()(tmp)
        
        
        EFNet_out = efficientnet_b0.outputs[1:][::-1]
        down_stack = tf.keras.Model(inputs=efficientnet_b0.inputs, outputs=EFNet_out)

        print(EFNet_out)

        inputs = layers.Input(shape=[224,224,3])
        
        x = inputs
        skips = down_stack(x)
        print(skips)

        x = skips[-1]
        skips = reversed(skips[:-1])

        print(skips)

        up_stack = [
            upsample(512, 3),
            upsample(256, 3),
            upsample(128, 3),
            upsample(64, 3)
        ]

        for up, skip in zip(up_stack, skips):
            x = up(x)
            layers.Concatenate()([x, skip])
        
       # x = layers.BatchNormalization()(x)
        masks_layer = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding="same", use_bias=True, activation=tf.nn.sigmoid)(x)

        super().__init__(inputs=inputs, outputs=[masks_layer])

        opt = tf.keras.optimizers.RMSprop(args.opt[1]) if args.opt[0] == 'rms' \
            else tf.keras.optimizers.Adam()


        self.compile(
            optimizer=opt,
            loss=[
                #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                tf.keras.losses.BinaryCrossentropy()
            ],
            metrics=[CAGSMaskIoU()]
        )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=48, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--hidden_layers", default="512:relu", type=str)
    parser.add_argument("--l2", default=0.0001, type=float)
    parser.add_argument("--drop", default=0.3, type=float)
    parser.add_argument("--opt", default="rms:0.001", type=str, help='adam|rms:rate')
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

    os.makedirs(args.logdir)

    # Load the data
    cags = CAGS()

    def train_augment(e):
        image = e["image"]
        mask = e["mask"]
        if tf.random.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        #image = tf.image.resize_with_crop_or_pad(image, CAGS.H + 12, CAGS.W + 12)
        #image = tf.image.resize(image, [tf.random.uniform([], minval=CAGS.H, maxval=CAGS.H + 24, dtype=tf.int32),
        #                                tf.random.uniform([], minval=CAGS.W, maxval=CAGS.W + 24, dtype=tf.int32)])
        #image = tf.image.random_crop(image, [CAGS.H, CAGS.W, CAGS.C])
        return image, mask


    train = cags.train.map(CAGS.parse)
    #train = train.map(lambda e: (e['image'], (e['label'], tf.reshape(e['mask'], [-1]))))
    train = train.map(train_augment)
    train.shuffle(10000, seed=args.seed)
    train = train.batch(args.batch_size)

    dev = cags.dev.map(CAGS.parse)
    #dev = dev.map(lambda e: (e['image'], (e['label'], tf.reshape(e['mask'], [-1]))))
    dev = dev.map(lambda e: (e['image'], e['mask']))
    dev = dev.batch(args.batch_size)
    test = cags.test.map(CAGS.parse)
    test = test.map(lambda e: e['image'])
    test = test.batch(args.batch_size)

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    # TODO: Create the model and train it
    model = Network(args)
    print(model.summary())
    model.fit(train,
              epochs=args.epochs,
              validation_data=dev,
              callbacks=[tb_callback])

    model.save(os.path.join(args.logdir, f"M-{'fine' if args.fine else 'pre'}.h5"))

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as out_file:
        # TODO: Predict the masks on the test set
        test_masks = model.predict(test)
        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=out_file)
