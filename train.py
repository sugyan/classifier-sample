import hashlib
import os
import tensorflow as tf
from tensorflow.python.util import compat

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', './dataset',
                           '''Path to dataset directory''')


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


def main(argv=None):
    # image_lists = create_image_lists(FLAGS.dataset_dir)
    # print(len(image_lists))

    jpeg_data = tf.placeholder(tf.string)
    decoded = tf.image.decode_jpeg(jpeg_data, channels=3)
    image = tf.image.per_image_standardization(decoded)
    image = tf.random_crop(image, [60, 60, 3])
    image = tf.image.random_flip_left_right(image)
    print(image)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
