import os
import tensorflow as tf

GRAPH_PATH = os.path.join(os.path.dirname(__file__), 'model.pb')


class Recognizer:
    def __init__(self):
        self.labels = ['パスタ', 'ラーメン', 'うどん']
        self.load_graph()
        self.sess = tf.Session()
        self.input_jpeg = tf.placeholder(tf.string)
        image = tf.image.decode_jpeg(self.input_jpeg, channels=3)
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize_images(image, (64, 64))
        self.inputs = tf.expand_dims(image, axis=0)
        self.softmax = tf.nn.softmax(self.sess.graph.get_tensor_by_name('inference_1:0'))

    def load_graph(self):
        with tf.gfile.FastGFile(GRAPH_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    def run(self, image_path):
        jpeg_data = tf.gfile.FastGFile(image_path, 'rb').read()
        inputs = self.sess.run(self.inputs, feed_dict={self.input_jpeg: jpeg_data})
        predictions = self.sess.run(self.softmax, feed_dict={'input_images:0': inputs})
        top = predictions[0].argsort()[-len(predictions[0]):][::-1]
        result = []
        for idx in top:
            label = self.labels[idx]
            score = round(predictions[0][idx] * 100.0, 2)
            print('%s (score = %.2f)' % (label, score))
            info = {
                'name': label,
                'score': score,
            }
            result.append(info)
        return result
