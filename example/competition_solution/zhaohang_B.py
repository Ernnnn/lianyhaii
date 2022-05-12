from itertools import combinations
from matplotlib.pyplot import plot
import pandas as pd 
import numpy as np 

import lianyhaii
from sklearn import feature_selection

from lianyhaii.model import *
from lianyhaii.fe_selector import *
from lianyhaii.eda_tool import *
from lianyhaii.tools import * 


lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'learning_rate':0.01,
    'colsample_bytree':0.7,
    'subsample':0.7,
    'num_leaves': 2**7-1,
    'max_depth': 7,
    'tree_learner': 'serial',
    'max_bin': 155,
    'min_child_samples': 30,
    'early_stopping_rounds':100,
    'num_boost_round':5000,
    'reg_lambda':0.5,
    'reg_alpha':0.5,
    'seed':42,
    'n_jobs': -1,
    'verbose':-1,
}

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test_B榜.csv')
label = "LABEL"
test[label] = -1
remove_features = ['CUST_UID',label]
total_features = [x for x in train if x not in remove_features]

data = pd.concat([train,test],axis=0,ignore_index=True)

abcd_feats = ['NB_CTC_HLD_IDV_AIO_CARD_SITU',
'WTHR_OPN_ONL_ICO','LGP_HLD_CARD_LVL']
aio_map ={
    'A':1,    'B':2,    'C':3,    'D':4,    'E':5,    'F':6,    '?':-999,
}
yes_map = {    'Y':1,    '?':-999,}
yes_feats = ['MON_12_CUST_CNT_PTY_ID']
for f in abcd_feats:
    data[f] =data[f].map(aio_map)
data['MON_12_CUST_CNT_PTY_ID'] = data['MON_12_CUST_CNT_PTY_ID'].map(yes_map)
remove_features = ['CUST_UID',label]
for f in data.columns:
    if f in remove_features:continue
    data[f] = pd.to_numeric(data[f],errors='coerce',) - 2
train = data[data[label]!=-1].reset_index(drop=True)
test = data[data[label]==-1].reset_index(drop=True)
from lianyhaii.feature_tools import *
tnf = trans_num_feats(train,test,num_feats=[x for x in total_features if x not in abcd_feats+yes_feats],label=label)
tnf.log_num_feats(inplace=True)
total_features =  [x for x in train if x not in remove_features]
fs = FeatureSelected(train,test,feats=total_features,label=label)
# # fs.std_selector(plot_it=True)
# fs.adv_test_selector(plot_it=False,threshold=0.05)
# # fs.ks_test_selector(plot_it=True)
fs.remove_by_all()
fs.tt_drop_feats += ['AGN_CUR_YEAR_WAG_AMT', 'CUR_YEAR_PUB_TO_PRV_TRX_PTY_CNT','CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT','CUR_YEAR_COUNTER_ENCASH_CNT', 'OPN_TM', 'REG_DT', 'ICO_CUR_MON_ACM_TRX_TM', 'CUR_MON_COR_DPS_MON_DAY_AVG_BAL','CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT','NB_RCT_3_MON_LGN_TMS_AGV','AGN_AGR_LATEST_AGN_AMT',"REG_CPT",'COUNTER_CUR_YEAR_CNT_AMT', 'COR_KEY_PROD_HLD_NBR', 'CUR_YEAR_MON_AGV_TRX_CNT', 'CUR_YEAR_MID_BUS_INC','CUR_YEAR_COR_DPS_YEAR_DAY_AVG_INCR','AI_STAR_SCO','EMP_NBR', 'PUB_TO_PRV_TRX_AMT_CUR_YEAR','CUR_YEAR_COR_DMND_DPS_DAY_AVG_BAL', 'AGN_CUR_YEAR_AMT', 'ICO_CUR_MON_ACM_TRX_AMT', 'WTHR_OPN_ONL_ICO',]
lgb_params['boosting_type'] = 'rf'
lgb_params['bagging_freq'] = 1
lgb_params['bagging_fraction'] = 0.8
mt = make_test(train,test,base_features=[x for x in total_features if x not in fs.tt_drop_feats],new_features=[],
            m_score=[[0.,0.]],label=label,metrices=['auc','f1'],log_tool=None)
mt.init_CV(seed=42,CV_type='skFold',n_split=5)
oof,pred = mt.lgb_test(lgb_params=lgb_params)
sub = test[['CUST_UID']].copy()
sub['pred'] = mt.predictions
sub.to_csv('submission.txt',index=False,header=False,sep='\t')