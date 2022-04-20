import pandas as pd 
import numpy  as np 
import warnings 
warnings.filterwarnings('ignore')
import sys 
sys.path.append('../')
import lianyhaii

train = pd.DataFrame({
    'y':(np.random.random(size=500)>0.5).astype(int)
})
test = pd.DataFrame({
    'y':(np.random.random(size=500)>0.5).astype(int)
})
train['day'] = range(500)
test['day'] = range(500,1000)
for i in range(10):
    train[f'x{i}'] = np.random.random(size=500)
    test[f'x{i}'] = np.random.random(size=500)
label = 'y'

print(train.head())
lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_jobs': 4,
    'verbose':-1,
}
from lianyhaii.ts_model import *
### todo
# mt = ts_test(train,test,base_features=[f'x{i}' for i in range(10)],new_features=[],
#             m_score=[[0.90913,0.5]],label=label,metrices=['auc',],log_tool=None)
# mt.init_CV(seed=412,CV_type='skFold',n_split=5)
# oof,pred = mt.lgb_test(lgb_params=lgb_params)