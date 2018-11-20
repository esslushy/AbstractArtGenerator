#tensor needs this stuff
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

def cnnModel(features, labels, mode):

    inputLayer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=inputLayer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    pool1 = tf.layers.average_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2Flatten = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense1 = tf.layers.dense(inputs=pool2Flatten, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode==tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes" : tf.argmax(input=logits, axis=1),

        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metrics = {
        "accuracy" : tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

cifar_classifier = tf.estimator.Estimator(
    model_fn=cnnModel, model_dir="/dataset/cifar_convnet_model")
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
cifar_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False)
eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)