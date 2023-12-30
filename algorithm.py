import os
import random

import numpy as np
import scipy
import tensorflow as tf


def load_model(sess, path):
    model = tf.saved_model.loader.load(sess, ['vgg16'], path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, layer3, layer4, layer7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
                                      kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
                                       kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
                                       kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    keep_prob_value = 0.5
    learning_rate_value = 0.001
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, gt_batch in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op],
                               feed_dict={input_image: X_batch, correct_label: gt_batch,
                                          keep_prob: keep_prob_value, learning_rate: learning_rate_value})

            total_loss += loss


def get_batches(batch_size, image_shape):
    image_dir = './data/image'
    label_dir = './data/label'

    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.png')]
    random.shuffle(image_paths)

    num_examples = len(image_paths)

    while True:
        for offset in range(0, num_examples, batch_size):
            batch_images = []
            batch_labels = []

            for i in range(batch_size):
                image_path = image_paths[offset + i]
                label_path = os.path.join(label_dir, os.path.basename(image_path))

                image = scipy.misc.imresize(scipy.misc.imread(image_path), image_shape)
                image = image / 255.0

                label = scipy.misc.imresize(scipy.misc.imread(label_path), image_shape)
                label = (label[:, :, 0] > 128).astype(float)

                batch_images.append(image)
                batch_labels.append(label)

            yield np.array(batch_images), np.array(batch_labels)


def run(num_classes=2, image_shape=(160, 576), epochs=40, batch_size=16, vgg_path='./data'):
    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    get_batches_fn = get_batches(batch_size, image_shape)

    with tf.Session() as session:
        image_input, keep_prob, layer3, layer4, layer7 = load_model(session, vgg_path)
        model_output = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)

        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        train_nn(session, epochs, batch_size, get_batches_fn,
                 train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)



run()
