import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
import logging
from predict_client.abstract_client import AbstractPredictClient
from tfwrapper import twimage

logger = logging.getLogger(__name__)


class MockPredictClient(AbstractPredictClient):

    def __init__(self, host, model_name, model_version, model_path=None):
        super().__init__(host, model_name, model_version)

        if not model_path:
            logging.error('MockPredictClient needs a .pd file to load!')
            exit(1)

        sess = tf.Session()
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)

        print(tf.saved_model.main_op.main_op())
        # with tf.gfile.FastGFile(model_path, 'rb') as f:
        #     print(model_path, f)
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     tf.import_graph_def(graph_def, name='')

        img = twimage.imread('request.png')

        #print(img)

        # x = [n.name for n in tf.get_default_graph().as_graph_def().node]
        # for e in x:
        #     print(e)

        predict_op = sess.graph.get_tensor_by_name(signature_constants.CLASSIFY_OUTPUT_SCORES)
        sess.run(predict_op, feed_dict={signature_constants.CLASSIFY_INPUTS: img})

    def predict(self, request_data, request_timeout=10):

        logger.info('Sending request to in memory model')
        logger.info('Model name: ' + str(self.model_name))
        logger.info('Model version: ' + str(self.model_version))
        logger.info('Host: ' + str(self.host))

        with tf.Session() as sess:
            print(signature_constants.CLASSIFY_OUTPUT_SCORES, signature_constants.CLASSIFY_INPUTS)

            x = [n.name for n in tf.get_default_graph().as_graph_def().node]
            for e in x:
                print(e)
            predict_op = sess.graph.get_tensor_by_name(signature_constants.CLASSIFY_OUTPUT_SCORES + ':0')
            sess.run(predict_op, feed_dict={signature_constants.CLASSIFY_INPUTS + ':0': request_data})
