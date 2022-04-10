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
from lianyhaii.model import *

from sklearn.metrics import precision_score
def top255_precision(true,pred,n=255):
    tmp = pd.DataFrame({
        'pred':pred,
        'true':true,
    })
    print(tmp.mean())
    tmp = tmp.sort_values('pred',ascending=False).head(n)
    tmp['pred'] = 1
    return precision_score(y_true=tmp['true'],y_pred=tmp['pred'])

mt = make_test(train,test,
base_features=[f'x{i}' for i in range(10)],new_features=[],
            m_score=[[0.90913,0.5]],label=label,metrices=['auc',('tp255',top255_precision)],log_tool=None)
mt.init_CV(seed=412,CV_type='skFold',n_split=5)
oof,pred = mt.lgb_test(lgb_params=lgb_params)