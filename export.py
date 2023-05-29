import logging
import os
import tensorflow as tf
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")


def serving_input_receiver_fn():
    input_index = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name='feat_index')
    input_value = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None], name='feat_value')

    features = {'feature_indices': input_index, 'feature_values': input_value, 'ids': input_index}
    receiver_tensors = {'feat_index': input_index, 'feat_value': input_value}
    return tf.estimator.export.ServingInputReceiver(features=features, receiver_tensors=receiver_tensors)


def main(cf):
    if not os.path.exists(cf['train.export_dir']):
        os.makedirs(cf['train.export_dir'])

    best_export_dir = os.path.join(os.path.join(cf['train.model_dir'], 'export'), 'best_exporter')
    saved_model_dir = os.path.join(best_export_dir, sorted(os.listdir(best_export_dir))[-1])
    logging.info('saved_model_dir: {}'.format(saved_model_dir))

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.compat.v1.saved_model.loader.load(sess, ["serve"], saved_model_dir)
        input_graph_def = sess.graph.as_graph_def()
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, input_graph_def, ['out'])
        #outgraph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
        #tf.compat.v1.train.write_graph(outgraph, cf['train.export_dir'], cf['train.export_model'], as_text=False)
        tf.compat.v1.train.write_graph(frozen_graph, cf['train.export_dir'], cf['train.export_model'], as_text=False)
    logging.info('export done, export model name: {}'.format(cf['train.export_model']))
