import json
import logging
import os
WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")


def get_config(prov, fp, ts):
    # load frozen config
    with open(file=fp.replace(".json", "_{}.json".format(ts)), mode="r", encoding="utf-8") as f:
        config = json.load(f)

    # set train data path, eval data path, predict data path
    for name in ["data.train", "data.eval", "data.predict","data.predict_feedback"]:
        config[name] = "hdfs://{}/{}/{}_*/part*".format(config["data.export_path"], ts, name.split(".")[-1])

    # set dir
    result_dir = os.path.join(WORK_DIR, "result")
    config['train.model_dir'] = os.path.join(os.path.join(result_dir, 'model'), ts)
    config['train.export_dir'] = os.path.join(result_dir, 'export')
    config['train.export_model'] = 'model_%s_%s.pb' % (prov, ts)
    config['train.predict_dir'] = os.path.join(result_dir, 'predict')
    config['train.predict_model'] = os.path.join(config['train.export_dir'], config['train.export_model'])
    config['train.predict_result'] = os.path.join(config['train.predict_dir'], 'predict_%s_%s.csv' % (prov, ts))
    config['train.predict_feedback_result'] = os.path.join(config['train.predict_dir'], 'predict_feedback_%s_%s.csv' % (prov, ts))
    config['data.signals'] = ['all']
    '''
    # set feature indices
    config["index.deepfm"] = []
    tmp_dim = 0
    for fd in config["data.feats"]:
        config["index.deepfm"].append([i for i in range(tmp_dim, tmp_dim + fd["dim"])])
        tmp_dim += fd["dim"]
        for k, v in config["data.indices"].items():
            if int(fd["id"]) == v:
                config[k] = config["index.deepfm"].pop()
                break
    config["index.deepfm"] = [j for i in config["index.deepfm"] for j in i]
    '''
    config["index.deepfm"] = []
    tmp_dim = 0
    for fd in config["data.feats"]:
        #config["index.deepfm"].append([i for i in range(tmp_dim, tmp_dim + fd["dim"])])
        tmp_list = [i for i in range(tmp_dim, tmp_dim + fd["dim"])]
        config["index.deepfm"].append(tmp_list)
        tmp_dim += fd["dim"]
        for k, v in config["data.indices"].items():
            if int(fd["id"]) == v:
                if k in ['index.albumid','index.channelid']:
                    config[k] = tmp_list
                else:
                    config[k] = config["index.deepfm"].pop()
                break
    config["index.deepfm"] = [j for i in config["index.deepfm"] for j in i]
    # set gpu devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["train.cuda_visible_devices"]

    # logging config
    #for k, v in config.items():
        #logging.info('{} = {}'.format(k, v))
    return config
