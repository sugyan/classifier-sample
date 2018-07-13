import hashlib
import os
import random
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.framework import graph_util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', './dataset',
                           '''Path to dataset directory.''')
tf.app.flags.DEFINE_string('output_graph', './model.pb',
                           '''File name of output graph def.''')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            '''Batch size for training.''')
tf.app.flags.DEFINE_integer('training_steps', 1000,
                            '''How many training steps to run.''')


def create_image_lists(image_dir):
    if not tf.gfile.Exists(image_dir):
        tf.logging.error('Image directory "{}" not found.'.format(image_dir))
    result = {}
    sub_dirs = [x[0] for x in tf.gfile.Walk(image_dir)]
    for sub_dir in sub_dirs[1:]:
        dir_name = os.path.basename(sub_dir)

        tf.logging.info('Looking for images in "{}"'.format(dir_name))
        file_list = tf.gfile.Glob(os.path.join(image_dir, dir_name, '*.jpg'))

        training_images = []
        testing_images = []
        for file_name in file_list:
            basename = os.path.basename(file_name)
            v = int(hashlib.sha1(compat.as_bytes(file_name)).hexdigest(), 16) % 100
            if v < 90:
                training_images.append(basename)
            else:
                testing_images.append(basename)
        result[dir_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
        }
    return result


def get_inputs(sess, image_lists, jpeg_input, distorted_image, category):
    class_count = len(image_lists)
    images = []
    labels = []
    n = FLAGS.batch_size
    if category == 'testing':
        n = 0
        for v in image_lists.values():
            n += len(v['testing'])
    for _ in range(n):
        label_index = random.randrange(class_count)
        dir_name = list(image_lists)[label_index]
        basename = random.choice(image_lists[dir_name][category])
        filepath = os.path.join(FLAGS.dataset_dir, dir_name, basename)
        if not tf.gfile.Exists(filepath):
            tf.logging.fatal('File "{}" does not exist'.format(filepath))
        data = sess.run(distorted_image, feed_dict={jpeg_input: tf.gfile.FastGFile(filepath, 'rb').read()})
        images.append(data)
        labels.append(label_index)
    return images, labels


def get_image(jpeg_data, distortion=True):
    decoded = tf.image.decode_jpeg(jpeg_data, channels=3)
    image = tf.image.per_image_standardization(decoded)
    if distortion:
        image = tf.random_crop(image, [64, 64, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.5)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)
    return image


def inference(images, class_count, reuse=False, training=True):
    with tf.variable_scope('model', reuse=reuse):
        output = tf.identity(images)
        output = tf.layers.conv2d(output, 20, (3, 3), activation=tf.nn.relu, name='conv1')
        output = tf.layers.max_pooling2d(output, (3, 3), (2, 2))
        output = tf.layers.conv2d(output, 40, (3, 3), activation=tf.nn.relu, name='conv2')
        output = tf.layers.max_pooling2d(output, (3, 3), (2, 2))
        output = tf.layers.conv2d(output, 60, (3, 3), activation=tf.nn.relu, name='conv3')
        output = tf.layers.max_pooling2d(output, (3, 3), (2, 2))
        output = tf.reshape(output, [-1, 5 * 5 * 60])
        output = tf.layers.dense(output, 30, activation=tf.nn.relu)
        output = tf.layers.dropout(output, training=training)
        output = tf.layers.dense(output, class_count)
    return tf.identity(output, name='inference')


def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(cross_entropy)


def training(losses):
    train_op = tf.train.AdamOptimizer().minimize(losses)
    ema = tf.train.ExponentialMovingAverage(0.9999)
    with tf.control_dependencies([train_op]):
        op = ema.apply(tf.global_variables())
    return op


def main(argv=None):
    image_lists = create_image_lists(FLAGS.dataset_dir)
    class_count = len(image_lists)
    jpeg_data = tf.placeholder(tf.string)
    image = {
        'training': get_image(jpeg_data, distortion=True),
        'testing':  get_image(jpeg_data, distortion=False),
    }
    input_images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='input_images')
    input_labels = tf.placeholder(tf.int64, shape=[None])
    # for training
    training_logits = inference(input_images, class_count)
    losses = loss(input_labels, training_logits)
    train_op = training(losses)
    # for testing
    testing_logits = inference(input_images, class_count, reuse=True, training=False)
    correct_prediction = tf.equal(tf.argmax(testing_logits, 1), input_labels)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.training_steps):
            training_images, training_labels = get_inputs(sess, image_lists, jpeg_data, image['training'], 'training')
            loss_value, _ = sess.run([losses, train_op], feed_dict={
                input_images: training_images,
                input_labels: training_labels,
            })
            logging_message = 'step {:3d}: loss: {:.6f}'.format(i + 1, loss_value)
            if i % 10 == 0 or i == FLAGS.training_steps - 1:
                testing_images, testing_labels = get_inputs(sess, image_lists, jpeg_data, image['testing'], 'testing')
                accuracy_value = sess.run(accuracy, feed_dict={
                    input_images: testing_images,
                    input_labels: testing_labels,
                })
                logging_message += ' (accuracy: {:.3f}%)'.format(accuracy_value * 100.0)
            print(logging_message)
        output = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['inference_1'])
        with open(FLAGS.output_graph, 'wb') as f:
            f.write(output.SerializeToString())


if __name__ == '__main__':
    tf.app.run()
