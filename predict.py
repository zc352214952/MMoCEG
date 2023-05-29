import logging
import os
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(WORK_DIR)
from src.input_fn import input_fn
tf.compat.v1.disable_eager_execution()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")


def main(cf):
    if not os.path.exists(cf['train.predict_dir']):
        os.makedirs(cf['train.predict_dir'])

    with tf.io.gfile.GFile(cf['train.predict_model'], 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    dataset = input_fn(file_pattern=cf['data.predict'], is_train=False, params=cf)
    feats, _ = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    output = tf.import_graph_def(graph_def=graph_def,
                                 input_map={'feat_index': feats['feature_indices'],
                                            'feat_value': feats['feature_values']},
                                 return_elements=['out:0','experts/expert_out0:0','experts/expert_out1:0','experts/expert_out2:0','experts/expert_out3:0','experts/expert_out4:0','experts/expert_out5:0','experts/expert_out6:0','experts/expert_out7:0','experts/expert_out8:0','experts/expert_out9:0','task_out0:0','task_out1:0','task_out2:0','task_out3:0','task_out4:0','task_out5:0','task_out6:0','task_out7:0','task_out8:0','task_out9:0'])
    file_path = os.getcwd()
    with tf.compat.v1.Session() as sess:
        try:
            rows = 0
            k = 0
            while True:
                k += 1
                feat_data, output_data = sess.run([feats, output])
                print('test difference:')
                feat_indices = feat_data['feature_indices']
                feat_values = feat_data['feature_values']
                #print(feat_indices[:2,:])
                #print(feat_values[:2,:])
                #print()
                #print()
                #print('output_data[0]',output_data[0])
                #print(type(output_data[1])) 
                #print('output_data[1]',output_data[1])
                #print(type(output_data[1][0]))
                for i in range(10):
                    with open(file_path+'/log/expert'+str(i)+'.txt', 'wb') as f:
                        for j in range(output_data[i+1].shape[0]):
                            line = output_data[i+1][j]
                            np.savetxt(f, line, fmt='%.10f')
                    f.close()
                print('gate shape:')
                print(output_data[11].shape)
                print('k: ', k)
                if k == 2:
                    for i in range(10):
                        with open(file_path+'/log/gate'+str(i)+'.txt', 'wb') as f:
                            for j in range(output_data[i+11].shape[0]):
                                line = output_data[i+1][j]
                                np.savetxt(f, line, fmt='%.10f')
                        f.close()
                #df_value2 = pd.DataFrame(output_data[0], columns=['predict_score'], dtype=float)
                #print(df_value2.head())
                #df_value1 = pd.DataFrame(output_data[1], columns=['expert0'], dtype=float)
                #print(df_value1.head())
                df_ids = pd.DataFrame(feat_data['ids'], columns=cf["data.ids"]).applymap(lambda x: x.decode('utf-8'))
                #df_value = pd.DataFrame(output_data, columns=['predict_score','expert_out0','expert_out1','expert_out2','expert_out3','expert_out4','expert_out5','expert_out6','expert_out7','expert_out8','expert_out9','task_out0','task_out1','task_out2','task_out3','task_out4','task_out5'], dtype=float)
                #print(df_value.head())
                df_score = pd.DataFrame(output_data[0], columns=['predict_score'], dtype=float)
                df = pd.concat([df_ids, df_score], axis=1)
                df['base_score'] = df['predict_score']

                # save
                if not os.path.exists(cf['train.predict_result']):
                    df.to_csv(cf['train.predict_result'], index=False, header=True, encoding='utf-8', sep=',')
                else:
                    df.to_csv(cf['train.predict_result'], index=False, header=False, encoding='utf-8', sep=',', mode='a')
                rows += len(df)
                logging.info("model_name = %s, predict_rows = %d" % (cf['train.predict_model'], rows))
        except tf.errors.OutOfRangeError:
            logging.info("predict done")
