import os
import tensorflow as tf
from input_fn import input_fn
from model_fn import model_fn
from parse_config import get_config
from export import serving_input_receiver_fn, main as export_main
from predict import main as predict_main
from recall_rank_metrics import main as metrics_main
from update_model import main as update_model_main
tf.compat.v1.disable_eager_execution()

flags = tf.compat.v1.flags
flags.DEFINE_string("province", None, "province")
flags.DEFINE_string("config_file_path", None, "config file path")
flags.DEFINE_string("timestamp", None, "timestamp")
FLAGS = flags.FLAGS

# get parsed config
cf = get_config(FLAGS.province, FLAGS.config_file_path, FLAGS.timestamp)

def main(_):
    # set run config
    run_config = tf.estimator.RunConfig(model_dir=cf['train.model_dir'], save_summary_steps=100,
                                        save_checkpoints_steps=10000, log_step_count_steps=100,
                                        session_config=tf.compat.v1.ConfigProto(log_device_placement=False,
                                                                                gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)))
    '''
    run_config = tf.estimator.RunConfig(model_dir=cf['train.model_dir'], save_summary_steps=100,
                                        save_checkpoints_steps=10, log_step_count_steps=100,
                                        session_config=tf.compat.v1.ConfigProto(log_device_placement=False)) 
    '''
    # constructs an `Estimator` instance
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=cf)

    # constructs an exporter instance
    exporter = tf.estimator.BestExporter(serving_input_receiver_fn=serving_input_receiver_fn,
                                         compare_fn=lambda best_eval_result, current_eval_result:
                                         best_eval_result['auc'] <= current_eval_result['auc'], exports_to_keep=1)

    # train and evaluate the `estimator`
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(cf['data.train'], is_train=True, params=cf))

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(cf['data.eval'], is_train=False, params=cf),
                                      steps=1000, exporters=exporter, start_delay_secs=60, throttle_secs=30)

    # train and evaluate
    #tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    # export and predict and metrics
    #export_main(cf)
    predict_main(cf)
    #metrics_main(cf, FLAGS.province)
    if "data.feedback_sample_path" in cf: 
        predict_feedback_main(cf)
        metrics_feedback_main(cf, FLAGS.province)
    #if cf["train.daily_update"]:
        #update_model_main(cf, FLAGS.province, FLAGS.config_file_path, FLAGS.timestamp)
if __name__ == '__main__':
    if not os.path.exists(cf['train.model_dir']):
        os.makedirs(cf['train.model_dir'])
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.app.run()
