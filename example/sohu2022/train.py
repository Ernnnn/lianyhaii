

import json
import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
import os ,random,gc

from lianyhaii import *
from lianyhaii.model import *
from lianyhaii.feature_tools import *
from lianyhaii.tools import *
from tqdm import tqdm 
tqdm.pandas()


cat_params = {
    'learning_rate': 0.05,
    'bagging_temperature': 0.1,
    'l2_leaf_reg': 30,
    'depth': 12,
    # 'max_leaves': 48,
    'max_bin': 255,
    'iterations': 10000,
    # 'subsample':0.8,
    'task_type': 'GPU',
    'loss_function': "CrossEntropy",
    # 'loss_function': "LogLoss",

    # 'objective': 'MAE',
    'eval_metric': "AUC",
    'bootstrap_type': 'Bayesian',
    # 'bootstrap_type': 'Bernoulli',
    # 'boosting_type': 'Plain',
    'random_seed': 42,
    'early_stopping_rounds': 200,
    'use_best_model': True
}

class encode_din_features(object):
    """
    you should fill na using value "-1"
    hist_feature: [1,2,3,4,5]
    target_feture:1
    """
    def __init__(self,train,test,hist_feature:str,target_feature:str,weight_feature:str) -> None:

        self.train = train 
        self.test = test 
        self.hist_feature = hist_feature
        self.target_feature = target_feature
        self.weight_feature = weight_feature        
    
    @staticmethod
    def _calc_corr(x,y):
        return np.corrcoef(x,y)[0][1]

    def __make(self,size,window,max_len,method):
        sentence = []
        tmp_df = self.train[self.hist_feature].append(self.test[self.hist_feature], )
        target_df = self.train[self.target_feature].append(self.test[self.target_feature], )
        sentence = list(tmp_df)
        print('sample setence')
        print(sentence[0])
        print('start word2vec ...')
        model = Word2Vec(sentence, vector_size=size, window=window, min_count=1,
                         workers=-1, epochs=10, seed=1)

        values = tmp_df.tolist()
        targets = target_df.tolist()
        w2v = []
        for i in tqdm(range(len(values))):
            # if values[i] == '-1':
            #     w2v.append(0)
            #     continue
            cur_hist = [model.wv[x] for x in values[i][:max_len]]
            try :
                cur_target = model.wv[targets[i]]
            except:
                cur_target = [0]*size
            res = list(map(lambda x:self._calc_corr(cur_target,x),cur_hist))
            amean = np.mean(res)
            asum = np.sum(res)
            amax = np.max(res)
            amin = np.min(res)
            astd = np.std(res)
            w2v.append([amean,asum,amax,amin,astd])
        w2v = np.array(w2v)
        np.save('din_encoding.npy',w2v)
        
        ff = []
        for i,f in enumerate(['mean','sum','max','min','std']):
            f_name = f'{self.hist_feature}_{self.target_feature}_din_{f}'
            
            self.train[f_name] = w2v[:len(self.train),i]
            self.test[f_name] = w2v[-len(self.test):,i]
            ff.append(f_name)
        print('finish target encoding')
        return ff
    
    def encode(self,size:int=10,window:int=2,method:str='mean',max_len:int=5):
        return self.__make(size,window,max_len,method) 

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    np.random.seed(seed)
def clac_gauc(valid):
    auc_score = roc_auc_score(y_true=valid['label'],y_score=valid['pred'])
    print('valid auc',auc_score)
    tmp_cnt = valid.groupby('pvId')['label'].nunique()
    tmp_cnt = tmp_cnt[tmp_cnt==2].index.tolist()
    tmp_valid = valid[valid['pvId'].isin(tmp_cnt)].reset_index(drop=True)
    valid_score = tmp_valid.groupby('pvId').apply(lambda x:roc_auc_score(y_true=x['label'],y_score=x['pred']))
    print('valid gauc',valid_score.mean())


def features_extractor():

    train = pd.read_csv('Sohu2022_data/rec_data/train-dataset.csv')
    test = pd.read_csv('Sohu2022_data/rec_data/test-dataset.csv')
    print(train.shape,test.shape)
    print(train.head())
    test['label'] = -2
    trainb_path= '复赛训练数据/推荐赛道-复赛-训练数据/rec-train-dataset.csv'
    testb_path = '复赛测试数据/rec-test-dataset.csv'
    trainb = pd.read_csv(trainb_path)
    testb = pd.read_csv(testb_path)
    testb['label']  = -1

    test.rename(columns={'testSampleId':'sampleId'},inplace=True)
    testb.rename(columns={'testSampleId':'sampleId'},inplace=True)

    test = pd.concat([test,testb],axis=0,ignore_index=True)
    train = pd.concat([train,trainb],axis=0,ignore_index=True)



    ecf = encode_cat_feats(train,test,base_features+['pvId'],label)
    ecf.encode_lb(inplace=True,method='skl')

    ecf = encode_cat_feats(train,test,['logTs'],label)
    new_time = ecf.encode_lb(inplace=False,method='skl')[0]

    def strings2list(q):
        q = list(map(int,map(lambda x:x.split(':')[0],str(q).split(';'))))
        return q
    from tqdm import tqdm 
    train['userSeq'].fillna('-1',inplace=True)
    test['userSeq'].fillna('-1',inplace=True)
    train['userSeq'] = train['userSeq'].map(strings2list)
    test['userSeq'] = test['userSeq'].map(strings2list)
    ecf = encode_cat_feats(train,test,['userSeq'],label)
    w2v_feats = ecf.encode_w2v()


    data = pd.concat([train,test],axis=0,ignore_index=True)

    data['logTs'] = pd.to_datetime(data['logTs'],unit='ms')
    data['day'] = data['logTs'].dt.day

    ## 当天曝光
    data['day_item_cnt'] = data.groupby(['itemId','day'])['logTs'].transform('count')
    data['day_suv_cnt'] = data.groupby(['suv','day'])['logTs'].transform('count')
    data['day_suv_itemId_cnt'] = data.groupby(['suv','day','itemId'])['logTs'].transform('count')
    data['day_item_nunique'] = data.groupby(['itemId','day'])['logTs'].transform('nunique')
    data['day_suv_nunique'] = data.groupby(['suv','day'])['logTs'].transform('nunique')
    data['day_suv_itemId_nunique'] = data.groupby(['suv','day','itemId'])['logTs'].transform('nunique')


    data['pvId_rank'] = data.groupby('pvId')['logTs'].rank()
    data['suv_rank'] = data.groupby(['pvId', 'suv'])['logTs'].rank()
    data['itemId_rank'] = data.groupby(['pvId', 'itemId'])['logTs'].rank()
    data['rank_nunique'] = data['pvId'].map(data.groupby('pvId')['logTs'].nunique())
    data['suv_nunique'] = data['suv'].map(data.groupby('suv')['logTs'].nunique())
    data['itemId_nunique'] = data['itemId'].map(data.groupby('itemId')['logTs'].nunique())

    cnt_features = []
    for f in tqdm(['pvId','logTs','itemId','suv', 'operator',
                        'browserType',
                        'deviceType',
                        'osType',
                        'province',
                        'city']):
        f_name = f'{f}_cnt'
        data[f_name] = data[f].map(data[f].value_counts())
        cnt_features.append(f_name)

    ## 时间窗口大小
    data['pvId_maxmin'] = data.groupby(['pvId'])['logTs'].transform(lambda x:x.max()-x.min())
    data['pvId_demean'] = data.groupby(['pvId'])['logTs'].apply(lambda x:x-x.mean())
    data['pvId_demean_norm'] = data.groupby(['pvId'])[new_time].apply(lambda x:(x-x.mean())/x.std())

    data.to_feather('Sohu2022_data/rec_data.feather')




def features_extractor2():
    data = pd.read_feather('Sohu2022_data/rec_data.feather')

    data['id_rank'] = data.groupby('pvId')['itemId'].rank()
    data['id_rank_rank'] = data['pvId_rank'] - data['id_rank']

    data['suv_item_cnt'] = data.groupby(['suv','itemId'])['logTs'].transform('count')
    data['suv_item_nunique'] = data.groupby(['suv','itemId'])['logTs'].transform('nunique')
    data['logts_diff'] = data.groupby('pvId')['logTs'].diff().fillna(pd.to_timedelta('0s'))
    data['itemId_suv_rank'] = data.groupby(['suv', 'itemId'])['logTs'].rank()
    data['seq_len'] = data['userSeq'].map(len)
    nuni_feats = ['itemId','pvId',
                'operator',
                'browserType',
                'deviceType',
                'osType',
                'province',
                'city']
    
    for f in nuni_feats:
        data[f'suv_{f}_nuni'] = data.groupby('suv')[f].transform('nunique')
        data[f'suv_{f}_nuni_day'] = data.groupby(['suv','day'])[f].transform('nunique')

    data.to_feather('Sohu2022_data/rec_data.feather')


def features_extractor3():
    data = pd.read_feather('Sohu2022_data/rec_data.feather')

    ## 加入历史序列情感
    res_emo = pd.read_csv('res(1).csv')
    emo_arr = np.array(list(map(list,res_emo['pred'].map(lambda x:np.fromstring(x[1:-1],sep=' ')).values)))
    for i in range(5):
        res_emo[f'{i}_emo'] = emo_arr[:,i]
    emo_f = [f'{i}_emo' for i in range(5)]
    res_emo_stats = res_emo.groupby('id')[emo_f].agg(['mean','std','max','min'])
    emo_feats = [f"agg_{f1}_{f2}" for f1,f2 in res_emo_stats ]
    
    res_emo_stats = res_emo_stats.to_dict()
    emo_hists = []
    for k,v in res_emo_stats.items():
        data[f'emo_hist_{k[0]}_{k[1]}'] = data['userSeq'].progress_map(lambda x:combine_hist(x,v))
        emo_hists.append(f'emo_hist_{k[0]}_{k[1]}')
        # break
    data[emo_hists].to_feather('emo_hists.feather')
    print('emo hist :',emo_hists)


    train,test = data[data['label']!=-1].reset_index(drop=True),data[data['label']==-1].reset_index(drop=True)
    del data  

    edf = encode_din_features(train,test,hist_feature='userSeq',target_feature='itemId',weight_feature='')
    edf.encode(size=5,window=2,max_len=5)
    
    w2v = np.load('din_encoding.npy')
    din_feats= []
    for f in [2,3,4]:
        train[f'din_{f}'] = w2v[:len(train),f]
        test[f'din_{f}'] = w2v[-len(test):,f]
        din_feats.append(f'din_{f}')
    ecf = agg_num_feats(train,test,cat_feats=['pvId'],num_feats=din_feats)
    anf_feats= ecf.agg_feats(agg_type=['max','min','mean','sum'])
    print('din features ',din_feats)
    print('din stat feature ',anf_feats)

    
    res_emo = pd.read_csv('res(1).csv')
    emo_arr = np.array(list(map(list,res_emo['pred'].map(lambda x:np.fromstring(x[1:-1],sep=' ')).values)))
    res_emo['pred'] = res_emo['pred'].map(lambda x:np.fromstring(x[1:-1],sep=' '))
    for i in range(5):
        res_emo[f'{i}_emo'] = emo_arr[:,i]
    emo_f = [f'{i}_emo' for i in range(5)]
    res_emo_stats = res_emo.groupby('id')[emo_f].agg(['mean','std','max','min','sum'])
    emo_feats = [f"agg_{f1}_{f2}" for f1,f2 in res_emo_stats ]
    res_emo_stats.columns = emo_feats
    res_emo_stats.reset_index(drop=False,inplace=True)
    res_emo_stats.rename(columns={'id':'itemId'},inplace=True)
    
    train = train.merge(res_emo_stats,on='itemId',how='left')
    test = test.merge(res_emo_stats,on='itemId',how='left')
    ft = fillna_tools(train,test,emo_feats,label)
    ft.fillna_mean(inplace=True)
    tnf = trans_num_feats(train,test,num_feats=emo_feats,label=label)
    bin_emo_feats = tnf.cut_groups(bins=[20],strategy='kmeans')
    print('bins emo features',bin_emo_feats)

    data = pd.concat([train,test],axis=0,ignore_index=True)
    del train 
    del test 
    
    data.to_feather('Sohu2022_data/rec_data.feather')


def features_extractor4():
    data = pd.read_feather('Sohu2022_data/rec_data.feather')


    data.sort_values('logTs', inplace=True)
    data['logTs_lb_enc'] = data.logTs.map(lambda x:pd.Timestamp(x).timestamp())
    data['log_pv_diff'] = data.groupby(['pvId'])['logTs_lb_enc'].diff()
    df_temp = data.groupby(['pvId'])['log_pv_diff'].agg([
        ('day_range_max', 'max'),
        ('day_range_min', 'min'),
        ('day_range_mean', 'mean'),
        ('day_range_std', 'std'),
        ('day_range_skew', lambda x: x.skew()),
    ])
    diff_features = ['day_range_max','day_range_min','day_range_mean','day_range_std','day_range_skew']
    data = pd.merge(data, df_temp, on='pvId', how='left')

    data['log_suv_diff'] = data.groupby(['suv'])['logTs_lb_enc'].diff()
    df_temp = data.groupby(['suv'])['log_suv_diff'].agg([
        ('suv_range_max', 'max'),
        ('suv_range_min', 'min'),
        ('suv_range_mean', 'mean'),
        ('suv_range_std', 'std'),
        ('suv_range_skew', lambda x: x.skew()),
    ])
    diff_features += ['suv_range_max','suv_range_min','suv_range_mean','suv_range_std','suv_range_skew']
    data = pd.merge(data, df_temp, on='suv', how='left')

    data['log_itemId_diff'] = data.groupby(['itemId'])['logTs_lb_enc'].diff()
    df_temp = data.groupby(['itemId'])['log_itemId_diff'].agg([
        ('itemId_range_max', 'max'),
        ('itemId_range_min', 'min'),
        ('itemId_range_mean', 'mean'),
        ('itemId_range_std', 'std'),
        ('itemId_range_skew', lambda x: x.skew()),
    ])
    diff_features += ['itemId_range_max','itemId_range_min','itemId_range_mean','itemId_range_std','itemId_range_skew']
    data = pd.merge(data, df_temp, on='itemId', how='left')
    

    
    data.to_feather('Sohu2022_data/rec_data.feather')


def features_extractor5():
    
    data = []
    f = open('Sohu2022_data/rec_data/recommend_content_entity_0317.txt',encoding='utf-8',mode='r')
    for line in tqdm(f):
        line = json.loads(line)
        id = line['id']
        entity = ",".join(line['entity'])
        data.append([id,entity])


    f = open('复赛测试数据/recommend_content_entity.txt',encoding='utf-8',mode='r')
    for line in tqdm(f):
        line = json.loads(line)
        id = line['id']
        entity = ",".join(line['entity'])
        data.append([id,entity])


    f = open('复赛训练数据/recommend_content_entity.txt',encoding='utf-8',mode='r')
    for line in tqdm(f):
        line = json.loads(line)
        id = line['id']
        entity = ",".join(line['entity'])
        data.append([id,entity])
    data = pd.DataFrame(data,columns=['id','entity'])

    from sklearn.feature_extraction.text import TfidfVectorizer

    tfv = TfidfVectorizer(max_features=5000,)
    res = tfv.fit_transform(data['entity'])
    from sklearn.decomposition import TruncatedSVD

    tsvd = TruncatedSVD(n_components=30,random_state=1003)
    res_svd = tsvd.fit_transform(res)
    for f in range(30):
        data[f'tfidf_svd_{f}'] = res_svd[:,f]

    data['id'] = data['id'].astype(int)
    data.to_feather('entity_svd_data.feather')

def combine_hist(x,dict):
    xs = []
    for f in x :
        if f in dict:
            xs.append(dict[f])
        else:
            xs.append(0)
    return sum(xs)

if __name__ == '__main__':
    label = 'label'

    base_features = [
                'suv',
                # 'itemId',
                'operator',
                'browserType',
                'deviceType',
                'osType',
                'province',
                'city']
    features_extractor()
    features_extractor2()
    features_extractor3()
    features_extractor4()
    features_extractor5()
    data = pd.read_feather('Sohu2022_data/rec_data.feather')
    remove_features= ['sampleId','label','logTs','pvId','userSeq']


    new_features = [
    'day_item_cnt','day_suv_cnt','pvId_rank','suv_rank','itemId_rank','rank_nunique',
    'suv_nunique','itemId_nunique','day_suv_itemId_cnt',
        'pvId_cnt', 'logTs_cnt', 'itemId_cnt', 'suv_cnt', 'operator_cnt', 'browserType_cnt', 'deviceType_cnt', 'osType_cnt', 'province_cnt', 'city_cnt',
        'userSeq_W2V_0', 'userSeq_W2V_1', 'userSeq_W2V_2', 'userSeq_W2V_3', 'userSeq_W2V_4', 'userSeq_W2V_5', 'userSeq_W2V_6', 'userSeq_W2V_7', 'userSeq_W2V_8', 'userSeq_W2V_9',

    ]+['pvId_demean','pvId_maxmin','pvId_demean_norm']\
        +['pvId','itemId']+['id_rank','id_rank_rank',]\

    new_features += ['suv_item_cnt','suv_item_nunique','logts_diff']

    ## 对用户的刻画
    new_features += ['itemId_suv_rank','seq_len']
    nuni_feats = ['itemId','pvId',
                'operator',
                'browserType',
                'deviceType',
                'osType',
                'province',
                'city']

    cnt_features = []
    for f in nuni_feats:
        cnt_features.extend([f'suv_{f}_nuni',f'suv_{f}_nuni_day'])
    new_features += cnt_features

    diff_features = ['day_range_max', 'day_range_min', 'day_range_mean', 'day_range_std', 'day_range_skew', 'suv_range_max', 'suv_range_min', 'suv_range_mean', 'suv_range_std', 'suv_range_skew', 'itemId_range_max', 'itemId_range_min', 'itemId_range_mean', 'itemId_range_std', 'itemId_range_skew',]
    new_features += diff_features


    stats_din_featurs = ['pvId_din_2_agg_max', 'pvId_din_2_agg_min', 'pvId_din_2_agg_mean', 'pvId_din_2_agg_sum', 'pvId_din_3_agg_max', 'pvId_din_3_agg_min', 'pvId_din_3_agg_mean', 'pvId_din_3_agg_sum', 'pvId_din_4_agg_max', 'pvId_din_4_agg_min', 'pvId_din_4_agg_mean', 'pvId_din_4_agg_sum']
    din_feats = ['din_2', 'din_3', 'din_4']
    new_features += stats_din_featurs+din_feats

    emo_hists = ['emo_hist_0_emo_mean', 'emo_hist_0_emo_std', 'emo_hist_0_emo_max', 'emo_hist_0_emo_min', 'emo_hist_1_emo_mean', 'emo_hist_1_emo_std', 'emo_hist_1_emo_max', 'emo_hist_1_emo_min', 'emo_hist_2_emo_mean', 'emo_hist_2_emo_std', 'emo_hist_2_emo_max', 'emo_hist_2_emo_min', 'emo_hist_3_emo_mean', 'emo_hist_3_emo_std', 'emo_hist_3_emo_max', 'emo_hist_3_emo_min', 'emo_hist_4_emo_mean', 'emo_hist_4_emo_std', 'emo_hist_4_emo_max', 'emo_hist_4_emo_min']
    bin_emo_feats = ['agg_0_emo_mean_20_bin_kmeans', 'agg_0_emo_std_20_bin_kmeans', 'agg_0_emo_max_20_bin_kmeans', 'agg_0_emo_min_20_bin_kmeans', 'agg_0_emo_sum_20_bin_kmeans', 'agg_1_emo_mean_20_bin_kmeans', 'agg_1_emo_std_20_bin_kmeans', 'agg_1_emo_max_20_bin_kmeans', 'agg_1_emo_min_20_bin_kmeans', 'agg_1_emo_sum_20_bin_kmeans', 'agg_2_emo_mean_20_bin_kmeans', 'agg_2_emo_std_20_bin_kmeans', 'agg_2_emo_max_20_bin_kmeans', 'agg_2_emo_min_20_bin_kmeans', 'agg_2_emo_sum_20_bin_kmeans', 'agg_3_emo_mean_20_bin_kmeans', 'agg_3_emo_std_20_bin_kmeans', 'agg_3_emo_max_20_bin_kmeans', 'agg_3_emo_min_20_bin_kmeans', 'agg_3_emo_sum_20_bin_kmeans', 'agg_4_emo_mean_20_bin_kmeans', 'agg_4_emo_std_20_bin_kmeans', 'agg_4_emo_max_20_bin_kmeans', 'agg_4_emo_min_20_bin_kmeans', 'agg_4_emo_sum_20_bin_kmeans']
    new_features += emo_hists+bin_emo_feats
    emo_hists_df = pd.read_feather('emo_hists.feather')
    data = pd.concat([data,emo_hists_df],axis=1)
    del emo_hists_df 
    gc.collect()

    w2v_feats = ['userSeq_W2V_0', 'userSeq_W2V_1', 'userSeq_W2V_2', 'userSeq_W2V_3', 'userSeq_W2V_4', 'userSeq_W2V_5', 'userSeq_W2V_6', 'userSeq_W2V_7', 'userSeq_W2V_8', 'userSeq_W2V_9',]
    dense_features = diff_features + ['itemId_suv_rank','seq_len',] + cnt_features+emo_hists+stats_din_featurs+din_feats
    data[dense_features] = data[dense_features].fillna(-1)

    entity_df = pd.read_feather('entity_svd_data.feather')
    entity_df.rename(columns={'id':'itemId'},inplace=True)
    del entity_df['entity']
    gc.collect()

    entity_df = entity_df.groupby('itemId').max().reset_index(drop=False)
    entity_feats = entity_df.drop(['itemId',],axis=1).columns.tolist()
    
    data = data.merge(entity_df,on='itemId',how='left')
    data[entity_feats] = data[entity_feats].fillna(-1)
    new_features += entity_feats
    del entity_df
    gc.collect()

    data = data[data['label']!=-2].reset_index(drop=True)
    train,test = data[data['label']!=-1].reset_index(drop=True),data[data['label']==-1].reset_index(drop=True)
    del data 
    gc.collect()
    
    time_gap = '20220104160000'
    trn_mask = train['logTs']<=time_gap
    val_mask = train['logTs']>time_gap 

    train.loc[trn_mask,'fold'] = 0 
    train.loc[~trn_mask,'fold'] = 1 


    cat_params['learning_rate'] = 0.01
    cat_params['iterations'] = int(2600)
    cat_params['early_stopping_rounds'] = 10000
    mt = make_test(train,test,
    base_features=base_features,new_features=new_features,
            m_score=[[0.5,0.5]],label=label,metrices=['auc','f1'],log_tool=None)
    mt.init_CV(seed=42,CV_type='gFold',n_split=5,group_col='pvId')
    oof,pred = mt.cat_test(cat_params=cat_params,save_path='.')
    # train.loc[val_mask,'pred'] = oof 
    # clac_gauc(train[val_mask].reset_index(drop=True))

    version = 'submission'
    from pathlib import Path 
    Path(version).mkdir(exist_ok=True,parents=True)
    sub = test[['sampleId']]
    sub['Id'] = sub['sampleId'].astype(int)
    sub['result'] = pred.reshape(-1)
    sub[['Id','result']].to_csv(f'{version}/section2.txt',sep='\t',index=False)