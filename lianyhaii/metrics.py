# coding:utf-8

import random
import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error

# def metric_zs(y_true,y_pred):

def mape_zx(y_true,y_pred):
    assert len(y_true) == len(y_pred),'lenght must be the same'
    ## 排除ytrue中的缺失情况
    ytruena = np.isnan(y_true)
    y_pred = y_pred[~ytruena]
    y_true = y_true[~ytruena]
    ## demo
    # y_pred = np.array([1, np.nan, 3,4,5,6,9])
    # y_true = np.array([2, 3, 5,4.8,5.8,6.1,np.nan])
    w = np.abs(y_true - y_pred) / (y_true)
    na_ids = np.isnan(y_pred)
    max_ids = (w>=0.2)&(~na_ids)
    min_ids = (w<0.2)&(~na_ids)
    w[min_ids] = 100
    w[max_ids] = 0
    w[na_ids] = 40

    return w.sum()/w.shape[0]

def mape_zx_lgb(y_true,y_pred):
    assert len(y_true) == len(y_pred),'lenght must be the same'
    w = np.abs(y_true - y_pred) / (y_true)
    max_ids = w>=0.2
    min_ids = w<0.2
    w[min_ids] = 100
    w[max_ids] = 0

    return w.sum()/w.shape[0]

def mape_obj(y_pred,data):
    y_true = data.get_label()
    w = y_true
    # w[y_true<0.001] = w[y_true<0.001]**0.0001
    grad = np.sign(y_pred - y_true)/(w)

    hess = 1/(w)
    # hess = np.zeros_like(y_true)
    # grad[y_true < 0.01] *= 0.5
    # hess[y_true < 0.01] *= 0.5

    grad[(y_true==0)] = 0
    hess[(y_true==0)] = 0

    return grad,hess

def mape_zs_task1(y_pred,data):
    data['pred'] = y_pred
    ## 删除岗位B的非工作日记录
    tmp = data[~((data['post_id']=='B')&(~data['WKD_TYP_CD'].isin(['WN','WS'])))]
    # tmp = data[~((data['post_id']=='B')&(~data['WKD_TYP_CD']==0))]

    score = np.abs(tmp['pred'] - tmp['amount'])/(tmp['amount']+1)
    return score.sum()/score.shape[0]

def mape_zs_task2(y_pred,data):
    data['pred'] = y_pred
    ## 删除岗位B的非工作日记录
    tmp = data[~((data['post_id']=='B')&(~data['WKD_TYP_CD'].isin(['WN','WS'])))]
    ## 删除非工作时间的数据
    tmp['hour'] = tmp['date'].dt.strftime('%H%M')
    tmp = tmp[tmp['hour'] > '0830']
    tmp = tmp[tmp['hour'] < '1830']
    tmp.reset_index(drop=True,inplace=True)

    score = np.abs(tmp['pred'] - tmp['amount'])/(tmp['amount']+1)
    return score.sum()/score.shape[0]


def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()

    d['prob'] = list(y_predict)

    d['y'] = list(y_true)

    d = d.sort_values(['prob'], ascending=[0])

    y = d.y

    PosAll = pd.Series(y).value_counts()[1]

    NegAll = pd.Series(y).value_counts()[0]

    pCumsum = d['y'].cumsum()

    nCumsum = np.arange(len(y)) - pCumsum + 1

    pCumsumPer = pCumsum / PosAll

    nCumsumPer = nCumsum / NegAll

    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]

    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]

    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    # print(f'0.1% tpr is {TR1}, 0.5% tpr is {TR2} ,1% tpr is {TR3}')
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3
# def mape(y_true, y_pred):
#     return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


class pymetric():
    def __init__(self,y_true,y_pred,):
        # self.metric_name = metric_name
        self.yt = y_true
        self.yp = y_pred
        self.tt = {}
        self.new_metrics = []
    def base_metric(self,metric_est,threshold=None):
        """

        :param metric_est: a estimtor to score
        # :param direct:
        :param threshold:
        :return: int of score!
        """
        if not threshold:
            return metric_est(self.yt,self.yp)
        else:
            ytmp = [int(x>threshold) for x in self.yp]
            return metric_est(self.yt,ytmp)
    def add_metric(self,metric_name,metric_fuction):

        self.new_metrics.append((metric_name,metric_fuction))

    @staticmethod
    def recall_score(y_true,y_pred):
        yt = np.array(y_true)
        yp = np.array(y_pred)
        p_sum = yt.sum()
        tp = yp[yt == 1].sum()
        return tp / (p_sum + 1e-10)

    @staticmethod
    def acc_score(y_true,y_pred):
        yt = np.array(y_true)
        yp = np.array(y_pred)
        p_sum = yp.sum()
        tp = yp[yt == 1].sum()
        return tp / (p_sum + 1e-10)

    def gen_metric_dict(self,metric_names,th=0.5):
        for name in metric_names:
            if name == 'auc':
                self.tt['auc'] = self.base_metric(roc_auc_score)
            if name == 'f1':
                self.tt['f1'] = self.base_metric(f1_score,threshold=th)
            if name == 'recall':
                self.tt['recall'] = self.base_metric(self.recall_score,threshold=th)
            if name == 'acc':
                self.tt['acc'] = self.base_metric(self.acc_score,threshold=th)
            if name == 'tpr':
                self.tt['tpr'] = self.base_metric(tpr_weight_funtion)

            if name == 'mse':
                self.tt['mse'] = self.base_metric(mean_squared_error)
            if name == 'rmse':
                self.tt['rmse'] = np.sqrt(self.base_metric(mean_squared_error))
            if name == 'mae':
                self.tt['mae'] = self.base_metric(mean_absolute_error)
            if name == 'mape':
                self.tt[name] = self.base_metric(mean_absolute_percentage_error)
            if name == 'smape':
                self.tt[name] = self.base_metric(smape)
            if name == 'mape_zx':
                self.tt[name] = self.base_metric(mape_zx)
            for n,f in self.new_metrics:
                if name == n:
                    self.tt[n] = self.base_metric(f)
            # if name == 'mape_zs_task1':
            #     self.tt[name] = self.base_metric(mape_zx)

        return self.tt




