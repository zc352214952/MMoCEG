##MMoCEG
This code is used to reproduce the main experiment of our paper MMoCEG.

## Requirements

+ Python 3.7.11
+ tensorflow_gpu-2.4.0
+ Tesla V100
+ CUDA Version 11.0

## Code
+ main.py 
+ parse_config.py:config file
+ input_fn.py
+ model_fn.py
+ MMoCEG.py
+ export.py
+ predict.py

## Usage

you can run the file `main.py` to train the model. <br>

For example: `python -u main.py --province ${PROVINCE} --config_file_path ${FILE_PATH} --timestamp ${TIMESTAMP}`

```bash
usage: main.py [--province PROVINCE] [--config_file_path FILE_PATH] [--timestamp TIMESTAMP]
```

You can also change other parameters according to the usage:

```bash
usage: 

optional arguments:
  --batchSize BATCHSIZE
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --epoch EPOCH         the number of epochs to train 
  --lr LR               learning rate
  --embeddingSize EMBEDDINGSIZE
                        input embedding size
  --expertnum  EXPERTNUM
                        imput expert num
```

## Baselines
This folder contains all the baselines we compared in the paper. <br>
For cross_stitch、MMOE、PLE and STAR we implement them by ourselves referring to the original paper and open source implementation.

##有问题反馈
在使用中有任何问题，欢迎反馈给我，可以用以下联系方式跟我交流

* 邮件(zhangcongyjy@chinamobile.com)
