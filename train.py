import hashlib
import os
import random
import tensorflow as tf
from tensorflow.python.util import compat

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', './dataset',
                           '''Path to dataset directory''')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            '''Batch size for training''')


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
            if v < 80:
                training_images.append(basename)
            else:
                testing_images.append(basename)
        result[dir_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
        }
    return result


def get_inputs(sess, image_lists, jpeg_input, distorted_image):
    class_count = len(image_lists)

    images = []
    labels = []
    for _ in range(FLAGS.batch_size):
        label_index = random.randrange(class_count)
        dir_name = list(image_lists)[label_index]
        basename = random.choice(image_lists[dir_name]['training'])
        filepath = os.path.join(FLAGS.dataset_dir, dir_name, basename)
        if not tf.gfile.Exists(filepath):
            tf.logging.fatal('File "{}" does not exist'.format(filepath))
        data = sess.run(distorted_image, feed_dict={jpeg_input: tf.gfile.FastGFile(filepath, 'rb').read()})
        images.append(data)
        labels.append(label_index)
    return images, labels


def get_distorted_image():
    jpeg_data = tf.placeholder(tf.string)
    decoded = tf.image.decode_jpeg(jpeg_data, channels=3)
    image = tf.image.per_image_standardization(decoded)
    image = tf.random_crop(image, [64, 64, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.5)
    return jpeg_data, image


def inference(images, class_count):
    output = tf.identity(images)
    output = tf.layers.conv2d(output, 20, (3, 3), activation=tf.nn.relu, name='conv1')
    output = tf.layers.max_pooling2d(output, (3, 3), (2, 2))
    output = tf.layers.conv2d(output, 40, (3, 3), activation=tf.nn.relu, name='conv2')
    output = tf.layers.max_pooling2d(output, (3, 3), (2, 2))
    output = tf.layers.conv2d(output, 60, (3, 3), activation=tf.nn.relu, name='conv3')
    output = tf.layers.max_pooling2d(output, (3, 3), (2, 2))
    output = tf.reshape(output, [-1, 5 * 5 * 60])
    output = tf.layers.dense(output, 30, activation=tf.nn.relu)
    output = tf.layers.dense(output, class_count)
    return output


def loss(labels, logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(cross_entropy)


def training(losses):
    return tf.train.AdamOptimizer().minimize(losses)


def main(argv=None):
    image_lists = create_image_lists(FLAGS.dataset_dir)
    jpeg_input, distorted_image = get_distorted_image()
    input_images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    input_labels = tf.placeholder(tf.int32, shape=[None])
    logits = inference(input_images, len(image_lists))
    losses = loss(input_labels, logits)
    train_op = training(losses)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(500):
            images, labels = get_inputs(sess, image_lists, jpeg_input, distorted_image)
            loss_value, _ = sess.run([losses, train_op], feed_dict={input_images: images, input_labels: labels})
            print('step {:03d}: loss: {:.6f}'.format(i + 1, loss_value))

if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
