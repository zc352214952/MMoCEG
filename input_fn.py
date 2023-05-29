import tensorflow as tf
import os


def _parse_function(example_proto, params):
    feature_description = {
        "encode": tf.io.FixedLenFeature([params["data.field_size"]], tf.int64),
        "value": tf.io.FixedLenFeature([params["data.field_size"]], tf.float32),
        "labels": tf.io.FixedLenFeature([len(params["data.labels"])], tf.float32),
        "ids": tf.io.FixedLenFeature([len(params["data.ids"])], tf.string)
    }
    parsed_example = tf.io.parse_example(example_proto, feature_description)

    features = {
        'feature_indices': tf.cast(parsed_example["encode"], tf.int32),
        'feature_values': parsed_example["value"],
        'ids': parsed_example["ids"]
    }
    return features, parsed_example["labels"]


def input_fn(file_pattern, is_train, params):
    available_cpu_num = int(os.popen('cat /proc/cpuinfo |grep "process"|wc -l').read())
    num_parallel_readers = int(available_cpu_num * 2.25)
    num_parallel_calls = int(available_cpu_num / 2) + 1

    dataset = tf.data.TFRecordDataset(filenames=tf.io.gfile.glob(file_pattern), num_parallel_reads=num_parallel_readers,
                                      compression_type="GZIP")

    if is_train:
        dataset = dataset.shuffle(params['train.batch_size']*10).repeat(params["train.epoch"])
    dataset = dataset.batch(params['train.batch_size'], drop_remainder=True)
    dataset = dataset.map(map_func=lambda x: _parse_function(x, params), num_parallel_calls=num_parallel_calls, deterministic=False)
    dataset = dataset.prefetch(20)

    return dataset
