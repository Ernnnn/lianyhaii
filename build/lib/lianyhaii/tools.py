# coding:utf-8


import functools
import random
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson,boxcox
from sklearn.metrics import f1_score


# from autograd import elementwise_grad as egrad
# import autograd.numpy as npe

from lianyhaii.feature_tools import fillna_tools

import logging
from logging import handlers


# def f1_macro_(y_pred, y_true, n_labels=2):
#     total_f1 = 0.
#     for i in range(n_labels):
#         yt = y_true == i
#         yp = y_pred == i

#         tp = np.sum(yt & yp)

#         tpfp = npe.sum(yp)
#         tpfn = npe.sum(yt)
#         if tpfp == 0:
#             # print('[WARNING] F-score is ill-defined and being set to 0.0 in labels with no predicted samples.')
#             precision = 0.
#         else:
#             precision = tp / tpfp
#         if tpfn == 0:
#             # print(f'[ERROR] label not found in y_true...')
#             recall = 0.
#         else:
#             recall = tp / tpfn

#         if precision == 0. or recall == 0.:
#             f1 = 0.
#         else:
#             f1 = 2 * precision * recall / (precision + recall)
#         total_f1 += f1
#     return total_f1 / n_labels


# def f1_macro_loss(preds,data):
#     # print(type(preds),type(data))
#     y_label = npe.array(data.get_label())
#     preds = npe.array([int(x>0.5) for x in preds])

#     grad_f1 = egrad(f1_macro_,1)
#     hessian_f1 = egrad(egrad(f1_macro_,1))
#     grad_f1_score = grad_f1(preds,y_label)
#     hessian_f1_score = hessian_f1(preds,y_label)

#     return grad_f1_score,hessian_f1_score

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_reg_params(seed):
    lgb_params = {
        'objective': 'rmse',
        'boosting_type': 'gbdt',
        # 'boosting_type': 'goss',
        'metric': 'rmse',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 2**5-1,
        # 'max_depth': 5,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 1000,
        # 'max_bin': 100,
        'verbose': -1,
        # 'min_child_samples': 30,
        'seed': seed,
        # 'early_stopping_rounds': 100,
    }

    xgb_params = {
        'objective': 'reg:squarederror',
        # 'objective':'binary:logitraw',
        # 'objective':'binary:hinge',
        #     'booster': 'GPU_hist',
        'eval_metric': 'mae',
        #     'eval_metric': 'auc',

        'n_estimators':10000,
        'tree_method': 'hist',
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.01,
        'seed': seed,
        'nthread': -1,
        # 'silent': True,
    }

    cat_params = {
        'learning_rate': 0.1,
        'bagging_temperature': 0.1,
        'l2_leaf_reg': 30,
        'depth': 12,
        # 'max_leaves': 48,
        'max_bin': 255,
        'iterations': 10000,
        'task_type': 'GPU',
        'loss_function': "MAE",
        # 'objective': 'MAE',
        'eval_metric': "MAE",
        'bootstrap_type': 'Bayesian',
        'random_seed': seed,
        'early_stopping_rounds': 100,
        'use_best_model': True
    }

    lr_params = {
        'random_state': seed,
        'C': 1,
        'max_iter': 1000,
        'n_jobs': -1,

    }

    model_params = {
        'lgb': lgb_params,
        'xgb': xgb_params,
        'cat':cat_params,
        'lr':lr_params,
    }

    return model_params

def time_it(fun):
    @functools.wraps(fun)
    def wrapper(*args,**kwargs):
        """this is wrapper function"""
        func_name = fun.__name__
        start_time = time.time()
        res = fun(*args,**kwargs)
        end_time = time.time()
        print(f'{func_name} costing time : {(end_time-start_time):.2f}' )
        # print(f'costing time :{(end_time-start_time)}' )
        return res
    return wrapper

def add_base_features(test_feats=True, save_feats=False):
    global train, test
    global feature_path,ID
    new_feats = []
    file_name = sys._getframe().f_code.co_name[4:]
    if (not test_feats) and (not save_feats):
        ## 读取数据
        tr_feats = pd.read_feather(feature_path + f'train_feats_{file_name}.feather')
        tt_feats = pd.read_feather(feature_path + f'test_feats_{file_name}.feather')
        feats = [x for x in tr_feats if x not in [ID]]
        ## 合并数据
        train = train.merge(tr_feats, on=ID, how='left')
        test = test.merge(tt_feats, on=ID, how='left')
        return feats

    tt_new_features = [x for item in new_feats for x in item]
    if (not test_feats) and save_feats:
        train[[ID] + tt_new_features].to_feather(feature_path + f'train_feats_{file_name}.feather')
        test[[ID] + tt_new_features].to_feather(feature_path + f'test_feats_{file_name}.feather')
        print('saved feats', file_name)
    if test_feats:
        new_feats.append(tt_new_features)
        return new_feats
    else:
        return tt_new_features

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def transform_label(train,label,transform_type='ln',params=None):

    fitted_lambda = None
    if transform_type == 'ln':
        f_label = f"ln_{label}"
        train[f_label] = np.log(train[label]-train[label].min()+1)
        fitted_lambda = train[label].min()
        # return fitted_lambda
    elif transform_type == 'boxcox':
        f_label = f"bc_{label}"
        train[f_label] ,fitted_lambda= boxcox(train[label]-train[label].min()+1)
        fitted_lambda ={
            'lambda':fitted_lambda,
            'min_values':train[label].min(),
        }

    elif transform_type == 'yeojohnson':
        f_label = f"yj_{label}"
        train[f_label],fitted_lambda = yeojohnson(train[label],)
    elif transform_type == 'my':
        smoothing = 1e4
        f_label = f'my_{label}'
        train[f_label] = train[label] / smoothing
        fitted_lambda = smoothing

    elif transform_type == 'minmax':
        smoothing = 15
        f_label = f'my_{label}'
        train[f_label] = train[label].clip(0,smoothing) / smoothing
        fitted_lambda = smoothing

    else:
        f_label = label
        pass
    label = f_label
    return label,fitted_lambda


def clean_null(test_feats=True,label=None):
    global train,test
    new_feats = []
    tt_cols =  ['X38']
    ft = fillna_tools(train,test,tt_cols,label)
    ff = ft.fillna_min()
    new_feats.append(ff)
    ff = ft.fillna_mean()
    new_feats.append(ff)
    ff = ft.fillna_median()
    new_feats.append(ff)
    ff = ft.fillna_mode()
    new_feats.append(ff)
    new_feats.append(tt_cols)

    tt_new_features = [x for item in new_feats for x in item]
    if test_feats:
        new_feats.append(tt_new_features)
        return new_feats
    else:
        return tt_new_features

class resample_tools:

    def __init__(self,train:pd.DataFrame,test:pd.DataFrame,group:str,label:str):
        self.train = train
        self.test = test
        self.group = group

    def StratifiedSample(self):
        """ 使得他们的分层比例一样"""
        tr_group = self.train[self.group].unique()
        tt_group = self.test[self.group].unique()

        tr_size = self.train.shape[0]
        tt_size = self.test.shape[0]

        keep_tr = []

        for n in tt_group:
            ## 如果n不在train中则跳过
            if n not in tr_group:
                continue
            ## 如果tt中比例高于tr也跳过,但是保留所有索引
            tt_ratio = self.test[self.test[self.group] == n].shape[0] / tt_size
            tr_ratio = self.train[self.train[self.group] == n].shape[0] / tr_size
            if tt_ratio >= tr_ratio:
                keep_tr += self.train[self.train[self.group] == n].index.tolist()
                continue

            ## 应抽样量
            tr_sample = int(tr_size * tt_ratio)

            ## 不重复抽样
            keep_tr += list(np.random.choice(self.train[self.train[self.group] == n].index,tr_sample,replace=False))
        ## 返回索引
        keep_tr = sorted(keep_tr)

        return self.train.loc[keep_tr,:]


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]

    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
#     log = Logger('all.log',level='debug')
#     log.logger.debug('debug')
#     log.logger.info('info')
#     log.logger.warning('警告')
#     log.logger.error('报错')
#     log.logger.critical('严重')
#     Logger('error.log', level='error').logger.error('error')



