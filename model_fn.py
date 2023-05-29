import tensorflow as tf
from din_deepfm_mmoe import din_deepfm_mmoe


def model_fn(features, labels, mode, params):
    # get network output
    #output = deepfm_din(features, mode, params)
    output,loss_gates,cl_loss = din_deepfm_mmoe(features, mode, params)

    # ============= predict =============
    predictions = {'ids': features['ids'], 'predict_scores': output}
    export_outputs = {'export_scores': tf.estimator.export.PredictOutput(output)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # ============= loss =============
    loss = tf.compat.v1.losses.log_loss(labels=labels, predictions=output)
    print('loss shape')
    print(tf.shape(loss))
    loss += loss_gates
    loss += cl_loss
    loss += tf.compat.v1.losses.get_regularization_loss()

    # ============= optimizer =============
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params['network.learning_rate'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())

    # used for mean update and variance update in batch_normalization layer
    if params['network.batch_norm']:
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, update_ops])

    # ============= evaluate =============
    eval_metric_ops = {'auc': tf.compat.v1.metrics.auc(labels=labels, predictions=output, name='auc')}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
