#!/usr/bin/env python
# coding: utf-8

# In[81]:


import os
import sys
import pickle
import math
import pandas as pd
import numpy as np

import datetime
from pyhive import hive
from multiprocessing import Pool
from collections import defaultdict
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_rows', 100)
from pymongo import MongoClient
WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(WORK_DIR)


null_list=[np.nan,None]
class MongoBase(object):
    def __init__(self,ip,db):
        self.client=MongoClient(ip)

    def getDB(self,db):
        self.db=self.client[db]
   
    def getCollect(self,collection):
        self.collection=self.db[collection]
        return self.collection
    def closeDB(self):
        self.client.close()


# # 召回算法

# ## 召回算法评估

# In[ ]:


class RecallAlgoMetrics:
    def __init__(self, cp_list, channel_list, df_hot_album, k):
        self._cp_list = cp_list
        self._channel_list = channel_list
        self._df_hot_album = df_hot_album
        self._k = k
    
    # precision, recall, ndcg
    def pr_ndcg(self,predictions,answers):
        precision_user, recall_user, ndcg_user =[], [], []
        for user, items in answers.items():
            items = set(answers[user])
            pred = predictions.get(user,[])[:self._k]
            if len(items) > 0 and len(pred) > 0:
                ndcg_best = np.sum([1.0 / np.log2(rank + 2.0) for rank in range(0, len(items) if len(items) < self._k else self._k)])
                ndcg_real = 0
                for i in range(0,len(pred)):
                    if pred[i] in items:
                        ndcg_real += 1.0 / np.log2(i + 2.0)
                ndcg_user.append(ndcg_real / ndcg_best)
                hitnum = len(set(pred).intersection(items))
                recall = hitnum / len(items)
                precision = hitnum /min(len(pred),self._k)
                recall_user.append(recall)
                precision_user.append(precision)
        precision = sum(precision_user) / len(precision_user)
        recall = sum(recall_user) / len(recall_user)
        ndcg = sum(ndcg_user) / len(ndcg_user)
        print("有效用户数",len(recall_user))
        print("Precision",precision)
        print("Recall",recall)
        print("NDCG",ndcg)
        return precision, recall, ndcg
    
    # 媒资覆盖度  
    def cover_rate_at_recall(self,df_album,df_pred):
        all_album_cnt = len(df_album)
        album_list=list(df_album['id'])
        df_pred = df_pred.drop_duplicates('itemids')
        cover_list=list(df_pred['itemids'])
        cover_recal = len(set(cover_list).intersection(set(album_list))) / all_album_cnt
        print("cover:", cover_recal)
        return  cover_recal
    
    # 热度
    def hot_rate_at_recall(self,df_pred):
        df_hotalbum = self._df_hot_album
        predList = list(df_pred['itemids'])
        df_recal = df_hotalbum[df_hotalbum['album'].isin(predList)]
        recalhot = np.mean(df_recal['hot'].apply(lambda x:(1 + np.log(x))))                 
        print("hot:%.4f"%(recalhot))
        return recalhot  
    
    # cp覆盖度
    def cp_at_recall(self,df_pred):   
            if self._k <= 0 or len(self._cp_list) == 0:
                return None 
            cp_rate_dict = {}   
            for cpid in self._cp_list:
                df_pred['cp'] = df_pred['itemids'].apply(lambda x:x.split('|')[0])                
                pred_len = len(df_pred['userid'].unique())
                cpid_rate = list(df_pred['cp']).count(cpid) / (pred_len * self._k)
                cp_rate_dict[cpid] = cpid_rate
            return cp_rate_dict
    
    # 频道覆盖度
    def channel_at_recall(self,df_album,df_pred):   
        if self._k <= 0 or len(self._channel_list) == 0:
            return None 
        channel_rate_dict = {}   
        for channel in self._channel_list:
            album_channel= dict(zip(df_album['id'],df_album['channelid']))
            df_pred["channelid"] = df_pred['itemids'].apply(lambda x: album_channel.get(x, []))
            pred_len = len(df_pred['userid'].unique())                       
            channel_rate = len(df_pred[df_pred['channelid'] == channel]) / (pred_len * self._k) 
            channel_rate_dict[channel] = channel_rate
        return channel_rate_dict
    
    # 召回推出率
    def precision_rank_recom(self,predictions,rankRecs):
        recal_rec_user =[]
        for user, items in rankRecs.items():
            items = set(rankRecs[user])
            pred = predictions.get(user,[])[:self._k]
            if len(items) > 0 and len(pred) > 0:
                hitnum = len(set(pred).intersection(items))
                rec_prec = hitnum / min(len(pred),self._k)
                recal_rec_user.append(rec_prec)
        precision = np.mean(recal_rec_user)
        print("召回推出率",precision)
        return precision


# ## 召回评估调用

# In[ ]:


def replace_id(df,df_info,r_id):
    df_size = df.shape[0]
    cols = [i if i != r_id else 'userid' for i in df.columns.tolist()]
    df_info = df_info[['mac', r_id]].drop_duplicates().reset_index(drop=True)
    df = pd.merge(df, df_info, how='inner', on=r_id)
    df = df.rename(columns = {'mac': 'userid'})[cols]
    return df

def read_hive(start_date, end_date, province):
    null_list=['',"",'""',' ','“”',0,'0','NaN',None,np.nan,'Null','暂无','不祥','Nan','Na','空','null','NULL','None','NAN','nan','none','无','na','NaN','NA','未知','NONE','佚名']
    conn = hive.Connection(host = '172.31.226.5', port = 10000, username = '', database = province + 'ftpdata')
    if province == "sc":
        userID ='stbid'
    else:
        userID = 'userid'
    table = province + 'ftpdata'
    print("----sql-read---")
    sql = "select {},collect_set(concat(cpid,'|',albumid)) as itemids from {}.userplaydata where dt>={} and dt<={} group by {}".format(userID, table, start_date, end_date,userID)
    print(sql)
    df_play = pd.read_sql(sql, conn)
    df_play_drop = df_play[~df_play[userID].isin(null_list)]
    df_play_drop = df_play_drop.reset_index(drop=True)
    df_play_drop['itemids'] = df_play_drop['itemids'].apply(lambda x: [i.replace('"','') for i in x[1:-2].split(',')])
    return df_play_drop

def get_dataset_sc(start_date, end_date,test_date, province):
    conn = hive.Connection(host='172.31.226.5', port=10000, username='', database='scftpdata')
    sql_info = "select usercode,stbid,mac from scftpdata.boxinfo where dt={}".format(end_date)
    df_info = pd.read_sql(sql_info, conn)
    df_play_drop = read_hive(start_date, end_date, province)
    df_user = replace_id(df_play_drop, df_info, 'stbid')
    history_hits = dict(zip(df_user['userid'], df_user['itemids']))
    df_test = read_hive(test_date, test_date, province)
    df_user_test = replace_id(df_test, df_info, 'stbid')
    return history_hits, df_user_test

def get_dataset(start_date,end_date,test_date,province):
    df_user = read_hive(start_date, end_date, province)
    history_hits = dict(zip(df_user['userid'], df_user['itemids']))
    df_test = read_hive(test_date, test_date, province)
    return history_hits, df_test

def get_recall_train_test_data(start_date,end_date,test_date, province):
    if province == "sc":
        history_hits,test_data = get_dataset_sc(start_date,end_date,test_date,province)
    else:
        history_hits,test_data = get_dataset(start_date,end_date,test_date,province)  
    return history_hits, test_data

def get_test_pay(province,df_album,testPay_reset, history_hit):
    testPay_reset["hits"] = testPay_reset['userid'].apply(lambda x: history_hit.get(x, []))
    testPay_reset['pay_newclicks'] = testPay_reset.apply(lambda x: list(set(x['itemids']) - set(x['hits'])), axis=1)
    payClick = dict(zip(testPay_reset['userid'], testPay_reset['itemids']))
    pay_newClick = dict(zip(testPay_reset['userid'], testPay_reset['pay_newclicks']))
    return  payClick, pay_newClick
def get_recall_metrics(start_date, end_date,test_date,province,df_album,df_rank_dict, resPath, reslist,channel_filter,topk):
    history_hit,test_reset= get_recall_train_test_data(start_date, end_date,test_date,province)    
    #测试集中增加频道过滤，付费
    test_reset['itemids']=pd.DataFrame(test_reset['itemids']).agg({'itemids':','.join})
    test_reset = test_reset.join(test_reset["itemids"].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('id'))
    test_channel=test_reset.merge(df_album,on='id')
    if len(channel_filter)==0:
        test_filter=test_channel[['userid','id','paytype']]
    else:
        test_filter=test_channel[test_channel['channelid'].isin(channel_filter)].reset_index(drop=True)[['userid','id','paytype']]
    test_filter.rename(columns={'id':'itemids'},inplace=True)
    test_reset = test_filter.groupby('userid')['itemids'].apply(list).reset_index()
    testPay=test_filter[test_filter['paytype']=="1"].reset_index(drop=True)
    testPay_reset= testPay.groupby('userid')['itemids'].apply(list).reset_index()
    test_ans = dict(zip(test_reset['userid'], test_reset['itemids']))
    # 推新度评测
    test_reset["hits"] = test_reset['userid'].apply(lambda x: history_hit.get(x, []))
    test_reset['new_clicks'] = test_reset.apply(lambda x: list(set(x['itemids']) - set(x['hits'])), axis=1)
    newclick = dict(zip(test_reset['userid'], test_reset['new_clicks']))
    # 付费和推新付费
    payClick, pay_newClick = get_test_pay(province,df_album,testPay_reset, history_hit)
    test_ans = dict(zip(test_reset['userid'], test_reset['itemids'])) 
    for i in reslist:
        print("-----------%s----------" % (i))
        file = resPath + i + '.pickle'
        with open(file, 'rb') as f:
            pred_res = pickle.load(f)
        for user,item in pred_res.items():
             pred_res[user]=item[0:topk]
        pdata=[(k,v) for k,l in zip(pred_res.keys(),pred_res.values()) for v in l]
        df_pred=pd.DataFrame(pdata,columns =["userid","itemids"])
        print("原始测试集评测")
        pre, recall, ndcg = recall_algo_metrics.pr_ndcg(pred_res, test_ans)
        print("推新度评测")
        pre, recall, ndcg = recall_algo_metrics.pr_ndcg(pred_res, newclick)
        print("付费媒资评测")
        pre, recall, ndcg = recall_algo_metrics.pr_ndcg(pred_res, payClick)
        print("推新的且付费媒资评测")
        pre, recall, ndcg =recall_algo_metrics.pr_ndcg(pred_res, pay_newClick)
        print("cover--hot--cp_cover---channel_cover")
        cover_recal=recall_algo_metrics.cover_rate_at_recall(df_album,df_pred)
        recalhot=recall_algo_metrics.hot_rate_at_recall(df_pred)
        cp_rate_dict=recall_algo_metrics.cp_at_recall(df_pred)
        for cp, rate in cp_rate_dict.items():
            print("cp_cover--%s---%.4f"%(cp,rate))
        channel_rate_dict=recall_algo_metrics.channel_at_recall(df_album,df_pred)
        for channelid, rate in channel_rate_dict.items():
            print("channel_cover--%s--%.4f---"%(channelid,rate))
        print("召回推出率")
        precision =recall_algo_metrics.precision_rank_recom(pred_res,df_rank_dict)        


# # 排序算法

# ## 排序算法评估

# In[30]:


class RankAlgoMetrics:
    def __init__(self, cp_list, channel_list, album_hot_dict, topk, gap):
        self._cp_list = cp_list
        self._channel_list = channel_list
        self._album_hot_dict = album_hot_dict
        self._k = topk
        self._k2 = topk - gap
    
    # ndcg
    def _calc_dcg(self, df, k):
        sumdcg = 0
        i = 0
        for name, row in df[:k].iterrows():
            act = row['label']
            if act > 0:
                sumdcg += 1.0 / math.log(2 + i)
            i = i + 1
        return sumdcg

    def ndcg_at_topk(self, label, score):
        if self._k <= 0 or len(label) == 0 or len(score) == 0 or 1 not in list(label):
            return 0
        k = min(self._k, len(label))
        df_label = pd.DataFrame(pd.Series(label).rename('label')).reset_index(drop=True)
        df_score = pd.DataFrame(pd.Series(score).rename('score')).reset_index(drop=True)
        df = pd.concat([df_label, df_score], axis=1)
        df.sort_values('score', ascending=False, inplace=True)
        s1 = self._calc_dcg(df, k)
        df.sort_values('label', ascending=False, inplace=True)
        s2 = self._calc_dcg(df, k)
        if s2 <= 1e-9:
            return 0
        else:
            return s1 / s2
    
    # auc
    def auc(self, label, score):
        auc = 0
        try:
            auc = roc_auc_score(label, score)
        except ValueError:
            pass
        return auc
    
    # ctr
    def ctr_at_topk(self, label, score):   
        if self._k2 <= 0 or len(label) == 0 or len(score) == 0 or 1 not in list(label):
            return 0
        k = min(self._k2, len(label))
        df_label = pd.DataFrame(pd.Series(label).rename('label')).reset_index(drop=True)
        df_score = pd.DataFrame(pd.Series(score).rename('score')).reset_index(drop=True)
        df = pd.concat([df_label, df_score], axis=1)
        df.sort_values('score', ascending=False, inplace=True)
        df = df[:k]
        click_cnt = pd.Series(df['label']).sum()
        return click_cnt / k

    def playduration_at_topk(self, label, score):   
        k = min(self._k2, len(label))
        df_label = pd.DataFrame(pd.Series(label).rename('label')).reset_index(drop=True)
        df_score = pd.DataFrame(pd.Series(score).rename('score')).reset_index(drop=True)
        df = pd.concat([df_label, df_score], axis=1)
        df.sort_values('score', ascending=False, inplace=True)
        df = df[:k]
        click_cnt = pd.Series(df['label']).sum()
        return click_cnt / k
    # hot
    def hot_rate_at_topk(self, album, score):
        if self._k2 <= 0 or len(album) <= 0 or len(score) <= 0:
            return 0
        k = min(self._k2, len(album))
        df_album = pd.DataFrame(pd.Series(album).rename('album')).reset_index(drop=True)
        df_score = pd.DataFrame(pd.Series(score).rename('score')).reset_index(drop=True)
        df = pd.concat([df_album, df_score], axis=1)
        df.sort_values('score', ascending=False, inplace=True)
        df = df[:k]
        rate = 0
        for albumid in df['album']:
            if albumid in self._album_hot_dict:
                rate += math.log(1 + self._album_hot_dict[albumid])
        return rate / k
        
    # cover
    def cover_rate_at_topk(self, album, score):
        if self._k2 <= 0 or len(album) <= 0 or len(score) <= 0:
            return None
        k = min(self._k2, len(album))
        df_album = pd.DataFrame(pd.Series(album).rename('album')).reset_index(drop=True)
        df_score = pd.DataFrame(pd.Series(score).rename('score')).reset_index(drop=True)
        df = pd.concat([df_album, df_score], axis=1)
        df.sort_values('score', ascending=False, inplace=True)
        df = df[:k]
        return list(df['album'])

    # cp
    def cp_at_topk(self, cp, score):  
        if self._k2 <= 0 or len(self._cp_list) == 0 or len(cp) == 0 or len(score) == 0:
            return None
        k = min(self._k2, len(cp))
        df_cp = pd.DataFrame(pd.Series(cp).rename('cp')).reset_index(drop=True)
        df_score = pd.DataFrame(pd.Series(score).rename('score')).reset_index(drop=True)
        df = pd.concat([df_cp, df_score], axis=1)
        df.sort_values('score', ascending=False, inplace=True)
        df = df[:k]
        cp_rate_dict = {}       
        for cpid in self._cp_list:
            cpid_cnt = str(df['cp']).count(cpid)
            cp_rate_dict[cpid] = cpid_cnt / k
        return cp_rate_dict
    
    # channel
    def channel_at_topk(self, channel, score):
        if self._k2 <= 0 or len(self._channel_list) == 0 or len(channel) == 0 or len(score) == 0:
            return None
        k = min(self._k2, len(channel))
        df_channel = pd.DataFrame(pd.Series(channel).rename('channel')).reset_index(drop=True)
        df_score = pd.DataFrame(pd.Series(score).rename('score')).reset_index(drop=True)
        df = pd.concat([df_channel, df_score], axis=1)
        df.sort_values('score', ascending=False, inplace=True)
        df = df[:k]
        channel_rate_dict= {}
        for channelid in self._channel_list:
            channel_cnt = str(df['channel']).count(channelid)
            channel_rate_dict[channelid] = channel_cnt / k
        return channel_rate_dict


# ## 排序评估调用

# In[84]:


def get_rank_metric_of_each_group(x):
    key, values = x
    if cal_valid_metrics:
        predict_score, base_score, item_id, cp, channel, label, label_valid, play_duration = values['predict_score'], values['base_score'],values['albumid'], values['cpid'], values['channelid'], values['label'], values['label_valid'],values['play_duration']
        label_valid = label_valid.apply(lambda x : 0 if x == 'NULL' or x == '""' else (1 if int(x) >= 60 else 0))
        if len(list(predict_score)) <= 3:
            if cal_cp:
                return None, None, None, None, None, None, None, None, None, None, None, None,                     None, None, None, None, None, None, None, None
            else:
                return None, None, None, None, None, None, None, None, None, None, None, None,                     None, None, None, None, None, None
    else:
        predict_score, base_score, item_id, cp, channel, label, play_duration = values['predict_score'], values['base_score'],             values['itemid'], values['cpid'], values['channelid'], values['click_label'], values['play_duration']
        if len(list(predict_score)) <= 3:
            if cal_cp:
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None
            else:
                return None, None, None, None, None, None, None, None, None, None, None, None
    
    # auc
    auc_base = rank_algo_metrics.auc(label, base_score)
    auc_predict = rank_algo_metrics.auc(label, predict_score)
    # ndcg
    ndcg_base = rank_algo_metrics.ndcg_at_topk(label, base_score)
    ndcg_predict = rank_algo_metrics.ndcg_at_topk(label, predict_score)
    # ctr
    ctr_base = rank_algo_metrics.ctr_at_topk(label, base_score)
    ctr_predict = rank_algo_metrics.ctr_at_topk(label, predict_score)
    # play_duration
    play_duration_base = rank_algo_metrics.playduration_at_topk(play_duration, base_score)
    play_duration_predict = rank_algo_metrics.playduration_at_topk(play_duration, predict_score)
    if cal_valid_metrics:
        # auc_valid
        auc_valid_base = rank_algo_metrics.auc(label_valid, base_score)
        auc_valid_predict = rank_algo_metrics.auc(label_valid, predict_score)
        # ndcg_valid
        ndcg_valid_base = rank_algo_metrics.ndcg_at_topk(label_valid, base_score)
        ndcg_valid_predict = rank_algo_metrics.ndcg_at_topk(label_valid, predict_score)
        # ctr_valid
        ctr_valid_base = rank_algo_metrics.ctr_at_topk(label_valid, base_score)
        ctr_valid_predict = rank_algo_metrics.ctr_at_topk(label_valid, predict_score)
    # hot
    hot_base = rank_algo_metrics.hot_rate_at_topk(item_id, base_score)
    hot_predict = rank_algo_metrics.hot_rate_at_topk(item_id, predict_score) 
    # cover
    cover_base = rank_algo_metrics.cover_rate_at_topk(item_id, base_score)
    cover_predict = rank_algo_metrics.cover_rate_at_topk(item_id, predict_score)
    # channel
    channel_rate_base_dict = rank_algo_metrics.channel_at_topk(channel, base_score)
    channel_rate_predict_dict = rank_algo_metrics.channel_at_topk(channel, predict_score)
    if cal_valid_metrics:
        if cal_cp:
            # cp
            cp_rate_base_dict = rank_algo_metrics.cp_at_topk(cp, base_score)
            cp_rate_predict_dict = rank_algo_metrics.cp_at_topk(cp, predict_score)
            return auc_base, auc_predict, ndcg_base, ndcg_predict, ctr_base, ctr_predict,                     auc_valid_base, auc_valid_predict, ndcg_valid_base, ndcg_valid_predict, ctr_valid_base, ctr_valid_predict,                     hot_base, hot_predict, cover_base, cover_predict, channel_rate_base_dict, channel_rate_predict_dict, cp_rate_base_dict, cp_rate_predict_dict,play_duration_base,play_duration_predict 
        else:
            return auc_base, auc_predict, ndcg_base, ndcg_predict, ctr_base, ctr_predict,                     auc_valid_base, auc_valid_predict, ndcg_valid_base, ndcg_valid_predict, ctr_valid_base, ctr_valid_predict,                     hot_base, hot_predict, cover_base, cover_predict, channel_rate_base_dict, channel_rate_predict_dict, play_duration_base,play_duration_predict
    else:
        if cal_cp:
            # cp
            cp_rate_base_dict = rank_algo_metrics.cp_at_topk(cp, base_score)
            cp_rate_predict_dict = rank_algo_metrics.cp_at_topk(cp, predict_score)
            return auc_base, auc_predict, ndcg_base, ndcg_predict, ctr_base, ctr_predict,                     hot_base, hot_predict, cover_base, cover_predict, channel_rate_base_dict, channel_rate_predict_dict, cp_rate_base_dict, cp_rate_predict_dict,play_duration_base, play_duration_predict
        else:
            return auc_base, auc_predict, ndcg_base, ndcg_predict, ctr_base, ctr_predict,                     hot_base, hot_predict, cover_base, cover_predict, channel_rate_base_dict, channel_rate_predict_dict,play_duration_base, play_duration_predict
    
def get_rank_metrics(file_path, album_online_list, group_key):
    if cal_valid_metrics:
        testdata = pd.read_csv(file_path, sep=',',converters={'cpid':str}).rename(columns={'playduration':'label_valid'})
        print('testdata')
        testdata['play_duration'] = testdata['label_valid']
        label_valid = testdata['label_valid'].apply(lambda x : 0 if x == 'NULL' or x == '""' else (1 if int(x) >= 60 else 0))
        testdata['play_duration'] = testdata['play_duration'].apply(lambda x: 0 if x == 'NULL' or x == '""' else int(x))
    else:
        testdata = pd.read_csv(file_path, sep=',',converters={'cpid':str})

    group_data = [(x[0], x[1]) for x in testdata.groupby(group_key)]   
    pool = Pool(processes = 32)
    group_list_raw = pool.map(get_rank_metric_of_each_group, group_data)
    pool.close(); 
    pool.join()
    
    group_metric_df = pd.DataFrame(group_list_raw)
    print("the input number of group:%s" % (len(group_metric_df)))
    group_metric_df.dropna(inplace=True)
    print("the actural number of group:%s" % (len(group_metric_df)))
    
    cover_base_list, cover_predict_list = [], []
    channel_rate_base_dict = defaultdict(lambda :0)
    channel_rate_predict_dict = defaultdict(lambda :0)
    if cal_cp:
        cp_rate_base_dict = defaultdict(lambda :0)
        cp_rate_predict_dict = defaultdict(lambda :0) 
    
    auc_base_avg = group_metric_df[0].mean()
    auc_predict_avg = group_metric_df[1].mean()
    ndcg_base_avg = group_metric_df[2].mean()
    ndcg_predict_avg = group_metric_df[3].mean()
    ctr_base_avg = group_metric_df[4].mean()
    ctr_predict_avg = group_metric_df[5].mean()

    if cal_valid_metrics:
        auc_valid_base_avg = group_metric_df[6].mean()
        auc_valid_predict_avg = group_metric_df[7].mean()
        ndcg_valid_base_avg = group_metric_df[8].mean()
        ndcg_valid_predict_avg = group_metric_df[9].mean()
        ctr_valid_base_avg = group_metric_df[10].mean()
        ctr_valid_predict_avg = group_metric_df[11].mean()

        hot_base_avg = group_metric_df[12].mean()
        hot_predict_avg = group_metric_df[13].mean()
        cnt = len(group_metric_df[14])
        for i in list(group_metric_df[14].keys()):
            cover_base_list += group_metric_df[14][i]
            cover_predict_list += group_metric_df[15][i]
            for key, val in group_metric_df[16][i].items():
                channel_rate_base_dict[key] += val    
            for key, val in group_metric_df[17][i].items():
                channel_rate_predict_dict[key] += val 
            if cal_cp:
                for key, val in group_metric_df[18][i].items():
                    cp_rate_base_dict[key] += val
                for key, val in group_metric_df[19][i].items():
                    cp_rate_predict_dict[key] += val
                play_duration_base_avg = group_metric_df[20].mean()
                play_duration_predict_avg = group_metric_df[21].mean()
            else:
                play_duration_base_avg = group_metric_df[18].mean()
                play_duration_predict_avg = group_metric_df[19].mean()
               
    else:
        hot_base_avg = group_metric_df[6].mean()
        hot_predict_avg = group_metric_df[7].mean()
        cnt = len(group_metric_df[8])
        for i in list(group_metric_df[8].keys()):
            cover_base_list += group_metric_df[8][i]
            cover_predict_list += group_metric_df[9][i]
            for key, val in group_metric_df[10][i].items():
                channel_rate_base_dict[key] += val    
            for key, val in group_metric_df[11][i].items():
                channel_rate_predict_dict[key] += val 
            if cal_cp:
                for key, val in group_metric_df[12][i].items():
                    cp_rate_base_dict[key] += val
                for key, val in group_metric_df[13][i].items():
                    cp_rate_predict_dict[key] += val
                play_duration_base_avg = group_metric_df[14].mean()
                play_duration_predict_avg = group_metric_df[15].mean()
            else:
                play_duration_base_avg = group_metric_df[12].mean()
                play_duration_predict_avg = group_metric_df[13].mean()
                
            
    cover_base_rate = len(set(cover_base_list).intersection(set(album_online_list))) / len(set(album_online_list))
    cover_predict_rate = len(set(cover_predict_list).intersection(set(album_online_list))) / len(set(album_online_list))
    for channelid, rate in channel_rate_base_dict.items():
        channel_rate_base_dict[channelid] = rate / cnt
        channel_rate_predict_dict[channelid] = rate / cnt
    if cal_cp:
        for cpid, rate in cp_rate_base_dict.items():
            cp_rate_base_dict[cpid] = rate / cnt
            cp_rate_predict_dict[cpid] = rate / cnt
        
    total_auc_base = rank_algo_metrics.auc(testdata['label'], testdata['base_score'])
    total_auc_predict = rank_algo_metrics.auc(testdata['label'], testdata['predict_score'])
    if cal_valid_metrics:
        total_valid_auc_base = rank_algo_metrics.auc(label_valid, testdata['base_score'])
        total_valid_auc_predict = rank_algo_metrics.auc(label_valid, testdata['predict_score'])
    print("************************************")
    print("-----------total auc-----------")
    print("total_auc_base:%s; total_auc_test:%s" % (total_auc_base, total_auc_predict))
    print("-----------group auc-----------")
    print("auc_base:%s; auc_test:%s" % (auc_base_avg, auc_predict_avg))
    print("-----------group top ndcg-----------")
    print("ndcg_base_avg:%s; ndcg_test_avg:%s" % (ndcg_base_avg, ndcg_predict_avg))
    print("-----------group top ctr-----------")
    print("ctr_base_avg:%s; ctr_test_avg:%s" % (ctr_base_avg, ctr_predict_avg))
    print("-----------group top play duration-----------")
    print("play_duration_base_avg:%s; play_duration_test_avg:%s" % (play_duration_base_avg, play_duration_predict_avg))
    if cal_valid_metrics:
        print("************************************")
        print("-----------total valid auc-----------")
        print("total_valid_auc_base:%s; total_valid_auc_test:%s" % (total_valid_auc_base, total_valid_auc_predict))
        print("-----------group valid auc-----------")
        print("auc_valid_base:%s; auc_valid_test:%s" % (auc_valid_base_avg, auc_valid_predict_avg))
        print("-----------group top valid ndcg-----------")
        print("ndcg_valid_base_avg:%s; ndcg_valid_test_avg:%s" % (ndcg_valid_base_avg, ndcg_valid_predict_avg))
        print("-----------group top valid ctr-----------")
        print("ctr_valid_base_avg:%s; ctr_valid_test_avg:%s" % (ctr_valid_base_avg, ctr_valid_predict_avg))
        print("************************************")
    print("-----------group top hot-----------")
    print("hot_base_avg:%s; hot_test_avg:%s" % (hot_base_avg, hot_predict_avg))
    print("-----------group top cover-----------")
    print("cover_base_rate:%s; cover_predict_rate:%s" % (cover_base_rate, cover_predict_rate))
    print("************************************")
    print("-----------channel-----------")
    print("channelid, base, test:")
    for key, val in channel_rate_base_dict.items():
        print("%s, %s, %s" % (key, val, channel_rate_predict_dict[key]))
    if cal_cp:
        print("-----------cp-----------")
        print("cpid, base, test:")
        for key, val in cp_rate_base_dict.items():
            print("%s, %s, %s" % (key, val, cp_rate_predict_dict[key]))
        


# # 评估模块

# In[38]:


def get_album_online_list(province):
    if province == "js":
        mongo_database_online = 'video'
    else:
        mongo_database_online = province + 'video'
    mongo_ip_online ='mongodb://rwadmin:dmp123321@172.22.245.136:27017,172.22.245.137:27017,172.22.245.138:27017'
    album_collect_online = 'videonew'
    mongoClient_online = MongoBase(mongo_ip_online, mongo_database_online)
    db_online = mongoClient_online.getDB(mongo_database_online)
    mongo_online = mongoClient_online.getCollect(album_collect_online)
    print(mongo_online)
    cursor_online = mongo_online.find({"Updatetype": "1"})
    df_on = pd.DataFrame(list(cursor_online))
    df_album = df_on[['id','channelid','paytype']]
    album_list = list(df_album['id'].drop_duplicates().reset_index(drop=True))
    return df_album, album_list

def get_album_hot_dict(start_dt, end_dt, province):
    conn = hive.connect(host='172.21.190.44',port=10000,auth='LDAP',username='hive', password='Hive321',database=province + 'ftpdata')
    db = province + 'ftpdata'
    print("----sql-read---")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sql = "select concat(cpid,'|',albumid) as album, count(*) as hot from {}.userplaydata where dt>={} and dt<={}         group by cpid,albumid".format(db, start_dt, end_dt)
    print(sql)
    df = pd.read_sql(sql, conn).reset_index()
    df_hot_album=df[~df['album'].isin(null_list)].reset_index(drop=True)
    album_hot_dict = dict(zip(df['album'],df['hot']))
    return df_hot_album, album_hot_dict

def get_reclist(dt,province):
    conn = hive.connect(host='172.21.190.44',port=10000,auth='LDAP',username='hive', password='Hive321',database=province + 'ftpdata')
    db = province + 'ftpdata'
    print("----sql-read---")
    sql = "select userid,itemids from {}.newresulttraceid where dt={} and signal='01' ".format(db,dt)
    print(sql)
    df_rank = pd.read_sql(sql, conn).reset_index(drop=True)
    df_rank['itemids']=df_rank['itemids'].apply(lambda x:x.split(','))
    df_rankRec=df_rank.groupby('userid')['itemids'].sum().reset_index()
    df_rank_dict=dict(zip(df_rankRec['userid'],df_rankRec['itemids']))
    return df_rank_dict


# In[5]:

def main(cf, province):
    global cal_cp, cal_valid_metrics, recall_algo_metrics, rank_algo_metrics
    print("################### start metrics ###################")
    type = 'rank' # recall or rank
    # province = prov # js, jx, gd, sc
    end_dt = cf["data.train_date"]
    start_dt = (datetime.datetime.strptime(end_dt, "%Y%m%d") + datetime.timedelta(1 - cf['data.train_days'])).strftime("%Y%m%d")
    file_path = cf['train.predict_result'] # recall:directory of test files; rank: predict_score,base_score,userid,trace_id,itemid,cpid,channelid,label(,play_duration)
    #file_path = '/data1/model_deepfm/zjrec/metrics/predictuser_channel_valid2d_emb20_win8_rank_1206_emb_zj_filternull.csv'
    cp_list = ['59','38','04','03','01','02','05','06','64','47','24','27','61','33','23','29','18','70',\
                  '69','21','28','20','65','66','26','10','56','36','46','48','77']
    channel_list = ['1000000','1000001','1000002','1000003','1000004','1000005','1000006','1000007','1000008',                '1000009','1000010','1000011','1000012','1000013','1000014','1000015','1000019','1000029',                '1000030','1000031','1000032','1000033','1000034','1000035']
    cal_cp = False
    cal_valid_metrics = True
    print('province: {}'.format(province))
    print('start_dt: {}'.format(start_dt))
    print('end_dt: {}'.format(end_dt))
    print('file_path: {}'.format(file_path))

    df_album, album_online_list = get_album_online_list(province)
    df_hot_album,album_hot_dict = get_album_hot_dict(start_dt, end_dt, province)


    if type == 'recall':
        topk = 30
        res_list = ['bipart','itemcf','ngram','item2vec','ALS_rec_comic']
        recall_algo_metrics = RecallAlgoMetrics(cp_list, channel_list, df_hot_album, topk)
        get_recall_metrics(start_dt, end_dt, province, df_album, file_path, res_list, topk)
    else:
        topk = 6
        gap = 1 # when group_key is trace_id, used in ctr/hot/cover/cp/channel
        group_key = 'userid' # trace_id, user_id
        rank_algo_metrics = RankAlgoMetrics(cp_list, channel_list, album_hot_dict, topk, gap)
        print('RankAlgoMetrics is finished')
        get_rank_metrics(file_path, album_online_list, group_key)
        
