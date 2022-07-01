# coding:utf-8
import os
import pickle
import time
from multiprocessing import Pool

import optuna
import plotly
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier,LGBMRegressor
import lightgbm as lgb
from optuna.visualization import plot_param_importances, plot_optimization_history
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_recall_curve, auc,precision_score
from sklearn.model_selection import StratifiedKFold, KFold,GroupKFold
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,precision_score,recall_score
from sklearn.linear_model import LinearRegression
# import pmdarima as pm
# from statsmodels.gam.tests.test_gam import sigmoid
from tqdm import tqdm
import xgboost as xgb

from lianyhaii.metrics import pymetric
from scipy.misc import derivative

def pr_auc(y_true, probas_pred):
    p, r, _ = precision_recall_curve(y_true, probas_pred)
    return auc(r, p)

def f2_score(y_true,y_pred):
    # print(y_true.shape)
    # print(y_pred.shape)
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    # print(y_true.shape,y_pred.shape)
    # TP = np.sum((y_true@y_pred))
    # FP = np.sum(((~y_true)@y_pred))
    # FN = np.sum(((y_true)@(~y_pred)))
    # precision = TP/(TP+FP)
    # recall = TP/(TP+FN)
    recall = recall_score(y_true=y_true,y_pred=y_pred)
    precision = precision_score(y_true=y_true,y_pred=y_pred)
    F2 = 5*recall*precision/(4*precision+recall)
    return F2 



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


def tpr_weight_cunstom(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'tpr_weight',tpr_weight_funtion(y_predict=y_hat,y_true=y_true),True

# def init_score(y_true):
#     # 样本初始值寻找过程
#     res = optimize.minimize_scalar(
#         lambda p: (y_true, p).sum(),
#         bounds=(0, 1),
#         method='bounded'
#     )
#     p = res.x
#     log_odds = np.log(p / (1 - p))
#     return log_odds

def focal_loss_lgb(y_pred, dtrain, alpha, gamma, num_class):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    # N observations x num_class arrays
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1,num_class, order='F')
    # alpha and gamma multiplicative factors with BCEWithLogitsLoss
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    # flatten in column-major (Fortran-style) order
    return grad.flatten('F'), hess.flatten('F')

def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma, num_class):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class, order='F')
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    # a variant can be np.sum(loss)/num_class
    return 'focal_loss', np.mean(loss), False


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    is_higher_better = True
    return 'f1', f1_score(y_true=y_true, y_pred=y_hat,average='macro'), is_higher_better


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def f1_score_custom(y_true, y_pred):
    y_pred = y_pred.round()
    return 'f1', f1_score(y_true, y_pred), True





class make_test():
    def __init__(self,tr_df,tt_df,base_features,new_features,m_score,label,metrices=None,log_tool=None):
        print(tr_df.shape,tt_df.shape)
        print(base_features+new_features)
        self.train = tr_df
        self.test = tt_df
        self.base_features = base_features
        self.new_features = new_features
        self.m_score = m_score
        self.label = label
        self.features = base_features + new_features
        self.predictions = None
        self.model = []
        self.features_imp = []
        self.metrices = metrices
        self.run = log_tool


    def init_CV(self,seed,n_split=5,shuffle=True,CV_type='skFold',group_col=None):
        self.cv_conf = {}
        if  self.run is not None:
            self.run['cv_type'] = CV_type
        if CV_type == 'skFold':
            cv = StratifiedKFold(n_splits=n_split,shuffle=shuffle,random_state=seed)
            self.cv_conf['iter'] = cv.split(self.train[self.features],self.train[self.label])
            self.cv_conf['n'] = n_split

        elif CV_type == 'kFold':
            cv = KFold(n_splits=n_split, shuffle=shuffle, random_state=seed)
            self.cv_conf['iter'] = [x for x in cv.split(X=self.train[self.features], y=self.train[self.label])]

            self.cv_conf['n'] = n_split

        elif CV_type == 'lastFold':
            folds = sorted(self.train[group_col].unique())

            cv = [[self.train[self.train[group_col]<folds[-1]].index,
                 self.train[(self.train[group_col]==folds[-1])].index]]
            self.cv_conf['iter'] = cv
            self.cv_conf['n'] = 1

        elif CV_type == 'gFold':
            cv = GroupKFold(n_splits=n_split,)
            self.cv_conf['iter'] = cv.split(self.train[self.features],y=self.train[self.label],groups=self.train[group_col])
            self.cv_conf['n'] = n_split
        elif CV_type == 'online':
            folds = sorted(self.train[group_col].unique())
            cv = [[self.train.index,self.train[(self.train[group_col]==folds[-1])].index]]

            self.cv_conf['iter'] = cv
            self.cv_conf['n'] = 1

        else:
            raise ValueError('no this type of fold')

    def __deal_cat_features(self,cat_features,m_type='lgb'):
        from sklearn.preprocessing import LabelEncoder
        if m_type == 'lgb':
            for col in cat_features:
                if self.train[col].dtype.name != 'category':
                    self.train[col] = self.train[col].fillna('unseen_before').astype(str)
                    self.test[col] = self.test[col].fillna('unseen_before').astype(str)


                    le = LabelEncoder()
                    le.fit(list(self.train[col])+list(self.test[col]))
                    self.train[col] = le.transform(self.train[col])
                    self.test[col] = le.transform(self.test[col])

                    self.train[col] = self.train[col].astype('category')
                    self.test[col] = self.test[col].astype('category')
    def __check_diff_score(self,oof_predictions,val_idx=None):

        if val_idx is None:
            pm = pymetric(y_true=self.train[self.label],y_pred=oof_predictions)
        else:
            pm = pymetric(y_true=self.train.loc[val_idx,self.label],y_pred=oof_predictions[val_idx])
        ## 加入新函数
        old_metrics = [x for x in self.metrices if type(x) == str]
        new_metrics = [x for x in self.metrices if type(x) != str]
        for name,f in new_metrics:
            pm.add_metric(name,f)
        old_metrics += [x for x,_ in new_metrics]
        result_score = pm.gen_metric_dict(metric_names=old_metrics,th=0.5)
        
        for key,value in result_score.items():
            print(f'global {key} : {value}')
            if  self.run is not None :
                self.run[f'metrics/global_{key}'] = value
        print('='*10+'different with previous version'+'='*10)
        score_list = []
        for n,(key,value) in enumerate(result_score.items()):
            print(f'diff of {key} : {np.round(value-self.m_score[-1][n],5)}')
            if (key == 'auc')&(self.run is not None)*(np.round(value-self.m_score[-1][n],5)>0):
                self.run['sys/tags'].add(['boosted'])
            score_list.append(value)

        self.m_score.append(score_list)

    def lgb_test(self,lgb_params,cv_score=False,weight=None):
        # self.__deal_cat_features(cat_features,m_type='lgb')
        # imp = pd.DataFrame({
        #     'feature':self.features,
        #     'gain':0,
        #     'split':0,
        # })

        feature_imp = np.zeros(len(self.features))
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))

        for n,(trn,val) in enumerate(self.cv_conf['iter']):
            trn_X,trn_y = self.train.loc[trn,self.features],self.train.loc[trn,self.label]
            val_X,val_y = self.train.loc[val,self.features],self.train.loc[val,self.label]
            if weight is not None:
                trn_data = lgb.Dataset(trn_X,label=trn_y,weight=self.train.loc[trn,weight])
            else:
                trn_data = lgb.Dataset(trn_X,label=trn_y)
            # trn_data = lgb.Dataset(trn_X,label=trn_y)
            val_data = lgb.Dataset(val_X,label=val_y)

            estimator = lgb.train(lgb_params,
                                  trn_data,
                                  valid_sets=[trn_data,val_data],
                                  # fobj=f1_macro_loss,
                                  # feval=lgb_f1_score,
                                  # feval=tpr_weight_3_cunstom,
                                  verbose_eval=-1,
                                  )

            oof_predictions[val] = estimator.predict(val_X)
            self.model.append(estimator)
            cv_score_list.append(roc_auc_score(y_true=val_y,y_score=oof_predictions[val]))
            tt_predicts += estimator.predict(self.test[self.features]) / self.cv_conf['n']
            # imp['gain'] = estimator.feature_importance(importance_type='gain') / self.cv_conf['n']
            feature_imp += estimator.feature_importance(importance_type='split') / self.cv_conf['n']
            if self.run is not None:
                self.run['metrics/test_auc'].log(cv_score_list[-1])

        print(f"training CV oof mean : {np.round(np.mean(cv_score_list),5)}")
        if self.cv_conf['n']==1:
            self.__check_diff_score(oof_predictions,val_idx=val)
        else:
            self.__check_diff_score(oof_predictions)
        self.predictions = tt_predicts
        self.features_imp = feature_imp
        if self.run is not None:
            self.run['feature_importance'] = self.features_importance()
        if cv_score:
            return oof_predictions,tt_predicts,cv_score_list
        else:
            if self.cv_conf['n']==1:
                # print(oof_predictions.shape,oof_predictions[val].shape,)
                return oof_predictions[val],tt_predicts,val_y.values

            return oof_predictions,tt_predicts

    def xgb_test(self,xgb_params,cv_score=False):
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))

        for n,(trn,val) in enumerate(self.cv_conf['iter']):
            trn_X,trn_y = self.train.loc[trn,self.features],self.train.loc[trn,self.label]
            val_X,val_y = self.train.loc[val,self.features],self.train.loc[val,self.label]

            tr_data = xgb.DMatrix(trn_X, label=trn_y)
            val_X = xgb.DMatrix(val_X,label=val_y)

            watchlist = [(tr_data, 'train'), (val_X, 'eval')]
            estimator = xgb.train(
                xgb_params,
                tr_data,
                evals=watchlist,
                verbose_eval=100,
                num_boost_round=30000,
                # early_stopping_rounds=200,
                early_stopping_rounds=100,
            )

            oof_predictions[val] = estimator.predict(val_X)

            cv_score_list.append(roc_auc_score(y_true=val_y,y_score=estimator.predict(val_X)))
            tt_predicts += estimator.predict(xgb.DMatrix(self.test[self.features])) / self.cv_conf['n']
            if self.run is not None:
                self.run['metrics/test_auc'].log(cv_score_list[-1])

        self.__check_diff_score(oof_predictions)
        # print(imp.sort_values(['split','gain'],ascending=False,ignore_index=True).loc[imp['feature'].isin(self.new_features),:])
        self.predictions = tt_predicts
        if cv_score:
            if self.cv_conf['n']==1:
                # print(oof_predictions.shape,oof_predictions[val].shape,)
                return oof_predictions[val],tt_predicts,val_y.values

            return oof_predictions,tt_predicts

    def cat_test(self,cat_params,cv_score=False,cat_features=None,save_path='.'):
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))
        feature_imp = np.zeros(len(self.features))
        if cat_features is None:
            cat_features = [x for x in self.train.select_dtypes(include=['category']).columns.tolist() if x in self.features]

        for c in cat_features:
            self.train[c] = self.train[c].astype('str')
            self.test[c] = self.test[c].astype('str')

        for n,(trn,val) in enumerate(self.cv_conf['iter']):
            trn_X,trn_y = self.train.loc[trn,self.features],self.train.loc[trn,self.label]
            val_X,val_y = self.train.loc[val,self.features],self.train.loc[val,self.label]

            estimator = CatBoostClassifier(**cat_params)
            print(cat_params)
            print(cat_params)
            if os.path.exists(save_path+f'/model_cbt_f{n}.cbt'):
                estimator.load_model(save_path+f'/model_cbt_f{n}.cbt')
            else:
                estimator.fit(
                    trn_X,trn_y,
                    cat_features=cat_features,
                    # early_stopping_rounds=100,
                    # eval_set=[(val_X,val_y)],
                    eval_set=[(val_X,val_y)] if cat_params['task_type'] == 'GPU' else [(trn_X,trn_y),(val_X,val_y)],
                    use_best_model=True,
                    metric_period=200,
                    verbose=True,
                )
                estimator.save_model(save_path+f'/model_cbt_f{n}.cbt')
            oof_predictions[val] = estimator.predict_proba(val_X)[:,1]
            self.model.append(estimator)
            cv_score_list.append(roc_auc_score(y_true=val_y,y_score=estimator.predict_proba(val_X)[:,1]))
            tt_predicts += estimator.predict_proba(self.test[self.features])[:,1] / self.cv_conf['n']
            feature_imp += estimator.feature_importances_ / self.cv_conf['n']
            if self.run is not None:
                self.run['metrics/test_auc'].log(cv_score_list[-1])

        print(f"training CV oof mean : {np.round(np.mean(cv_score_list), 5)}")
        self.__check_diff_score(oof_predictions,val_idx=val)
        self.predictions = tt_predicts
        self.features_imp = feature_imp
        if self.run is not None:
            self.run['feature_importance'] = self.features_importance()
        if cv_score:
            return oof_predictions,tt_predicts,cv_score_list
        else:
            if self.cv_conf['n']==1:
                # print(oof_predictions.shape,oof_predictions[val].shape,)
                return oof_predictions[val],tt_predicts,val_y.values

            return oof_predictions,tt_predicts

    def sklearn_test(self, sklean_model, cv_score=False, weight=None):
        # self.__deal_cat_features(cat_features,m_type='lgb')
        # imp = pd.DataFrame({
        #     'feature':self.features,
        #     'gain':0,
        #     'split':0,
        # })

        feature_imp = np.zeros(len(self.features))
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))

        for n, (trn, val) in enumerate(self.cv_conf['iter']):
            trn_X, trn_y = self.train.loc[trn, self.features], self.train.loc[trn, self.label]
            val_X, val_y = self.train.loc[val, self.features], self.train.loc[val, self.label]


            estimator = sklean_model.fit(trn_X,trn_y)

            oof_predictions[val] = estimator.predict_proba(val_X)[:,1]
            self.model.append(estimator)

            cv_score_list.append(roc_auc_score(y_true=val_y, y_score=oof_predictions[val]))
            tt_predicts += estimator.predict_proba(self.test[self.features])[:,1] / self.cv_conf['n']
            
            if self.run is not None:
                self.run['metrics/test_auc'].log(cv_score_list[-1])


        print(f"training CV oof mean : {np.round(np.mean(cv_score_list), 5)}")

        self.__check_diff_score(oof_predictions)
        self.predictions = tt_predicts
        self.features_imp = feature_imp

        if self.run is not None:
            self.run['feature_importance'] = self.features_importance()
        if cv_score:
            return oof_predictions,tt_predicts,cv_score_list
        else:
            if self.cv_conf['n']==1:
                # print(oof_predictions.shape,oof_predictions[val].shape,)
                return oof_predictions[val],tt_predicts,val_y.values

            return oof_predictions,tt_predicts

    def find_threshold(self,oof,ylabel,weight_range,method='offline',predictions=None,online_thd=None):
        if method == 'offline':
            res = pd.DataFrame({
                'weight': weight_range,
                'oof_f1': 0,
                # 'oof_auc': 0,
            })
            # def record_f1(th):
            #     oof_f1 = f1_score(y_true=ylabel, y_pred=[int(x > n) for x in oof])
            #     # oof_auc = roc_auc_score(y_score=oof,y_true=ylabel)
            #
            #     # print('the weighted oof auc : ',oof_auc)
            #     print(f'the weighted {np.round(n, 4)} oof f1 {np.round(oof_f1, 4)} ')
            #     # res.loc[res['weight'] == n,'oof_auc'] = oof_auc
            #     res.loc[res['weight'] == n, 'oof_f1'] = oof_f1
            # p= Pool()
            # p.apply_async(record_f1,args=weight_range,)
            # p.join()
            # p.close()


            for n in weight_range:
                # oof_f1 = f1_score(y_true=ylabel,y_pred=[int(x>n) for x in oof ])
                # print(ylabel.shape,oof.shape)
                oof_f1 = f2_score(y_true=ylabel,y_pred=np.array([int(x>n) for x in oof]))
                # oof_auc = roc_auc_score(y_score=oof,y_true=ylabel)

                # print('the weighted oof auc : ',oof_auc)
                print(f'the weighted {np.round(n,4)} oof f1 {np.round(oof_f1,4)} ')
                # res.loc[res['weight'] == n,'oof_auc'] = oof_auc
                res.loc[res['weight'] == n,'oof_f1'] = oof_f1
            fig = plt.figure(figsize=(8, 6))
            ax2 = fig.add_subplot(111)
            res.index = res['weight']

            res['oof_f1'].plot(ax=ax2, style='y>-.', alpha=0.7, legend=True)
            plt.legend(loc=1)
            plt.show()
            best_threshold = res.loc[res['oof_f1']==res['oof_f1'].max(),'weight'].values
            print(f'the best threshold {best_threshold} and best f1 {res["oof_f1"].max()}')

        elif method == 'ythreshold':

            print(f'train dataset 01 distributions :{sum(ylabel)/len(ylabel)}')
            preds_index = int(len(predictions)*(1-(sum(ylabel)/len(ylabel))))
            sorted_pred = sorted(predictions)
            best_threshold = sorted_pred[preds_index]

            ##
            oof_index = sum(ylabel)
            sorted_oof = sorted(oof)
            best_oof = sorted_oof[oof_index]

            res = f1_score(y_true=ylabel,y_pred=[int(x>=best_oof) for x in oof],average='macro')
            print(f'weight is {best_oof} and test weight {best_threshold} oof was {res}')
        elif method == 'online':
            preds_index = int(len(predictions)*(online_thd/(1-online_thd)))
            sorted_pred = sorted(predictions)
            best_threshold = sorted_pred[preds_index]
            print(f'online wight is {best_threshold}')
            res = 0

        elif method == 'sortedPvalue':
            best_threshold,res = 0,0
            predictions = sorted(list(predictions))

            plt.figure(figsize=(20,10))
            sns.lineplot(predictions,)
            plt.show()
        else:
            raise NotImplementedError

        return best_threshold,res

    def features_importance(self):
        df = pd.DataFrame(self.features_imp, index=self.features,columns=['values'])
        df.sort_values('values',ascending=False,inplace=True,)

        return df

    def analysis_residual(self,oof_predictions):
        self.train['predic_res'] = oof_predictions
        res_train_t = self.train[self.train[self.label] == np.round(oof_predictions)].copy()
        res_train_f = self.train[self.train[self.label] != np.round(oof_predictions)].copy()
        try:
            import os
            os.mkdir('res_analysis')
        except:
            pass
        today = time.strftime("%Y-%m-%d", time.localtime())[5:]
        res_train_f.to_csv(f'./res_analysis/res_train_f_{today}.csv',index=False)
        res_train_t.to_csv(f'./res_analysis/res_train_t_{today}.csv',index=False)

    def submit_seeds_ensembel(self,lgb_params,seed_list):
        """
        the aim of fuction is to find good seed or ensembel seek to submit.

        """
        tt_predictions = np.zeros(len(self.test))
        seed_df = pd.DataFrame({
            'seed':seed_list,
            'oof_score':0,
            'cv_mean':0,
            'cv_std':0,
        })
        final_oof = np.zeros(len(self.train))

        for n,s in enumerate(seed_list):
            # seed_everything(s)
            self.init_CV(seed=s)
            oof_preds,tt_preds,cv_list = self.lgb_test(lgb_params=lgb_params,cv_score=True)
            tt_predictions += tt_preds/len(seed_list)
            final_oof += oof_preds/len(seed_df)

            seed_df.loc[n,'oof_auc'] = roc_auc_score(y_score=oof_preds,y_true=self.train[self.label])
            seed_df.loc[n,'oof_score'] = tpr_weight_funtion(y_predict=oof_preds,y_true=self.train[self.label])
            seed_df.loc[n,'cv_mean'] = np.mean(cv_list)
            seed_df.loc[n,'cv_std'] = np.std(cv_list)
            print(seed_df.loc[n,:],end='\n')

        self.predictions = tt_predictions
        global_auc = roc_auc_score(y_true=self.train[self.label],y_score=final_oof)
        global_tpr = tpr_weight_funtion(y_true=self.train[self.label],y_predict=final_oof)
        self.m_score.append([global_auc,global_tpr])
        seed_df['diff_abs'] = (seed_df['oof_auc'] - seed_df['cv_mean']).abs()
        seed_df.sort_values(['diff_abs','cv_std'],inplace=True)
        return seed_df

    def submit_pseudo_label_model(self,params,seed,model_type):
        # if len(self.predictions) == 0:
        print('training pseudo label .....')
        self.test['label'] = self.predictions.copy()
        test2p = self.test[(self.test['label']<=0.01)|(self.test['label']>=0.95)].copy()
        # relabel
        test2p.loc[test2p['label'] >= 0.5,'label'] = 1
        test2p.loc[test2p['label'] < 0.5,'label'] = 0

        train2p = pd.concat([self.train,test2p],axis=0)
        train2p.reset_index(drop=True,inplace=True)

        tmp_mt = make_test(train2p,self.test,self.features,[],m_score=self.m_score,label=self.label)
        tmp_mt.init_CV(seed)
        if model_type =='lgb':
            tmp_mt.lgb_test(params)
        if model_type == 'xgb':
            tmp_mt.xgb_test(params)

        self.predictions = tmp_mt.predictions

    def submit(self,ID,sub_file=True,threshold=None):
        today = time.strftime("%Y-%m-%d", time.localtime())[5:]
        # self.test[self.label] = [int(x) for x in self.predictions > 0.5]
        if threshold is None:
            self.test[self.label] = self.predictions
        else:
            self.test[self.label] = [int(x >= threshold) for x in self.predictions]
            print('sum of label',np.sum(self.test[self.label]))

        sub_train = self.test[[ID,self.label]].copy()
        if sub_file:
            sub_test = pd.read_csv('./data/submit.csv')
            sub = sub_test[[ID]].merge(sub_train, on=ID, how='left')
        else:
            sub = sub_train.copy()

        # plt.figure(figsize=(20, 10))
        # sns.distplot(sub['label'], bins=100)
        # plt.show()
        print('null in sub','\n',sub.isnull().sum())
        sub.fillna(0, inplace=True)
        score = str(np.round(self.m_score[-1][0],4))+"_"+str(np.round(self.m_score[-1][1],4))
        sub.to_csv(f'./result/sub_{today}_{score}.csv', index=False)
        return sub

    def psm_samples(self,lgb_params):

        ## p值
        tr,tt = self.train.copy(),self.test.copy()
        tr['test_label'] = 0
        tt['test_label'] = 1
        data = pd.concat([tr,tt],ignore_index=True,)
        # train,test = train_test_split(data,test_size=0.1,random_state=0)

        mt = make_test(data,data,self.features,[],m_score=[[0,0,0]],label='test_label')
        mt.init_CV(0)
        oof , test = mt.lgb_test(lgb_params)
        data['oof'] = oof
        ## 可视化
        # tmp_df = self.train[[self.label,'oof']]
        sns.boxplot(x='label',y='oof',data=data)
        plt.show()
        ## 找特征重要性
        imp_df = mt.features_importance()
        imp_df.reset_index(drop=False,inplace=True)
        imp_df.columns = ['features','values']
        sns.barplot(y='features',x='values',data=imp_df)
        plt.show()
        print(imp_df.head(10))
        return data,imp_df

    def submit_easy_ensemble(self,n_group=10,):
        pass

    def tune_model(self,params,model,type='binary',N_TRIALS=100,model_type='lgb'):

        def objective(trial):
            if model_type == 'lgb':
                obj_params = {
                    # 'random_state': params['seed'],
                    'metric': params['metric'],
                    'n_estimators': 5000,
                    'n_jobs': -1,
                    'seed':params['seed'],
                    'early_stopping_rounds': 100,
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
                    'max_depth': trial.suggest_int('max_depth', 6, 16),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 2**8),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                    'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.3, 0.9),
                    'max_bin': trial.suggest_int('max_bin', 128, 1024),
                    'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
                    'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
                    'cat_l2': trial.suggest_int('cat_l2', 1, 20),
                    'verbose': -1,
                      }

            elif model_type == 'xgb':
                obj_params = {
                    'objective': params['objective'],
                    # 'task_type': 'GPU',
                    'nthread': -1,
                    'seed': params['seed'],
                    'tree_method': 'hist',
                    'eval_metric': params['eval_metric'],
                    'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                    'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                    'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
                    'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
                    'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02]),
                    # 'n_estimators': 10000,
                    'max_depth': trial.suggest_int('max_depth', 5,14),
                    # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                }

            else:
                raise NotImplementedError
            oof, _ = model(obj_params, cv_score=False,)
            score = 0
            if type == 'binary':
                score = roc_auc_score(y_true=self.train[self.label],y_score=oof)
            if type == 'f1':
                score = f1_score(y_true=self.train[self.label],y_pred=[int(x>0.5) for x in oof],average='macro')

            return score

        study = optuna.create_study(study_name=f"optimization", direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        ## save log
        print(study.trials_dataframe())
        import os
        if not os.path.exists('user_data'):
            os.mkdir('user_data')

        study.trials_dataframe().to_csv('./user_data/' + f"{type}trial_parameters.csv", index=False)
        with open('./user_data/' + f'{type}_study.pickle', 'wb') as f:
            pickle.dump(study, f)

        fig = plot_param_importances(study,)
        plotly.offline.plot(fig, filename='imp_fig.html')
        fig = plot_optimization_history(study)
        plotly.offline.plot(fig, filename='history_fig.html')

        return study.best_trial.params



class multi_class_test():
    def __init__(self,tr_df,tt_df,num_class,base_features,new_features,m_score,label,metrices=None,log_tool=None):
        print(tr_df.shape,tt_df.shape)
        print(base_features+new_features)
        self.train = tr_df
        self.test = tt_df
        self.base_features = base_features
        self.new_features = new_features
        self.m_score = m_score
        self.label = label
        self.features = base_features + new_features
        self.predictions = None
        self.model = []
        self.features_imp = []
        self.metrices = metrices
        self.run = log_tool
        self.num_class = num_class


    def init_CV(self,seed,n_split=5,shuffle=True,CV_type='skFold',group_col=None):
        self.cv_conf = {}
        if  self.run is not None:
            self.run['cv_type'] = CV_type
        if CV_type == 'skFold':
            cv = StratifiedKFold(n_splits=n_split,shuffle=shuffle,random_state=seed)
            self.cv_conf['iter'] = cv.split(self.train[self.features],self.train[self.label])
            self.cv_conf['n'] = n_split

        elif CV_type == 'kFold':
            cv = KFold(n_splits=n_split, shuffle=shuffle, random_state=seed)
            self.cv_conf['iter'] = [x for x in cv.split(X=self.train[self.features], y=self.train[self.label])]

            self.cv_conf['n'] = n_split

        elif CV_type == 'lastFold':
            folds = sorted(self.train[group_col].unique())

            cv = [[self.train[self.train[group_col]<folds[-1]].index,
                 self.train[(self.train[group_col]==folds[-1])].index]]
            self.cv_conf['iter'] = cv
            self.cv_conf['n'] = 1

        elif CV_type == 'gFold':
            cv = GroupKFold(n_splits=n_split,)
            self.cv_conf['iter'] = cv.split(self.train[self.features],y=self.train[self.label],groups=self.train[group_col])
            self.cv_conf['n'] = n_split
        elif CV_type == 'online':
            folds = sorted(self.train[group_col].unique())
            cv = [[self.train.index,self.train[(self.train[group_col]==folds[-1])].index]]

            self.cv_conf['iter'] = cv
            self.cv_conf['n'] = 1

        else:
            raise ValueError('no this type of fold')

    def __deal_cat_features(self,cat_features,m_type='lgb'):
        from sklearn.preprocessing import LabelEncoder
        if m_type == 'lgb':
            for col in cat_features:
                if self.train[col].dtype.name != 'category':
                    self.train[col] = self.train[col].fillna('unseen_before').astype(str)
                    self.test[col] = self.test[col].fillna('unseen_before').astype(str)


                    le = LabelEncoder()
                    le.fit(list(self.train[col])+list(self.test[col]))
                    self.train[col] = le.transform(self.train[col])
                    self.test[col] = le.transform(self.test[col])

                    self.train[col] = self.train[col].astype('category')
                    self.test[col] = self.test[col].astype('category')
    def __check_diff_score(self,oof_predictions,val_idx=None):

        if val_idx is None:
            pm = pymetric(y_true=self.train[self.label],y_pred=oof_predictions)
        else:
            pm = pymetric(y_true=self.train.loc[val_idx,self.label],y_pred=oof_predictions[val_idx])

        result_score = pm.gen_metric_dict(metric_names=self.metrices,th=0.5)
        for key,value in result_score.items():
            print(f'global {key} : {value}')
            if  self.run is not None :
                self.run[f'metrics/global_{key}'] = value
        print('='*10+'different with previous version'+'='*10)
        score_list = []
        for n,(key,value) in enumerate(result_score.items()):
            print(f'diff of {key} : {np.round(value-self.m_score[-1][n],5)}')
            if (key == 'auc')&(self.run is not None)*(np.round(value-self.m_score[-1][n],5)>0):
                self.run['sys/tags'].add(['boosted'])
            score_list.append(value)

        self.m_score.append(score_list)

    def lgb_test(self,lgb_params,cv_score=False,weight=None,is_save=False):
        # self.__deal_cat_features(cat_features,m_type='lgb')
        # imp = pd.DataFrame({
        #     'feature':self.features,
        #     'gain':0,
        #     'split':0,
        # })

        feature_imp = np.zeros(len(self.features))
        cv_score_list = []
        oof_predictions = np.zeros((len(self.train),self.num_class))
        tt_predicts = np.zeros((len(self.test),self.num_class))

        for n,(trn,val) in enumerate(self.cv_conf['iter']):
            trn_X,trn_y = self.train.loc[trn,self.features],self.train.loc[trn,self.label]
            val_X,val_y = self.train.loc[val,self.features],self.train.loc[val,self.label]
            if weight is not None:
                trn_data = lgb.Dataset(trn_X,label=trn_y,weight=self.train.loc[trn,weight])
            else:
                trn_data = lgb.Dataset(trn_X,label=trn_y)
            # trn_data = lgb.Dataset(trn_X,label=trn_y)
            val_data = lgb.Dataset(val_X,label=val_y)

            estimator = lgb.train(lgb_params,
                                  trn_data,
                                  valid_sets=[trn_data,val_data],
                                #   fobj=lambda x,y: focal_loss_lgb(x, y, 0.25, 2., self.num_class),
                                #   feval=lambda x,y: focal_loss_lgb_eval_error(x, y, 0.25, 2., self.num_class),
                                  # feval=tpr_weight_3_cunstom,
                                  verbose_eval=-1,
                                  )

            oof_predictions[val,:] = estimator.predict(val_X)
            self.model.append(estimator)
            
            # cv_score_list.append(roc_auc_score(y_true=val_y,y_score=oof_predictions[val]))
            cur_oof = np.argmax(oof_predictions[val,:],axis=1)
            print(oof_predictions[val,:].shape)
            # print()
            print(cur_oof.shape)
            acc = np.sum(cur_oof == np.array(val_y))/(len(val_y))
            cv_score_list.append(acc)
            print('cur acc',acc)
            tt_predicts += estimator.predict(self.test[self.features]) / self.cv_conf['n']
            # imp['gain'] = estimator.feature_importance(importance_type='gain') / self.cv_conf['n']
            feature_imp += estimator.feature_importance(importance_type='split') / self.cv_conf['n']
            if self.run is not None:
                self.run['metrics/test_auc'].log(cv_score_list[-1])

        print(f"training CV oof mean : {np.round(np.mean(cv_score_list),5)}")
        cur_oof = np.argmax(oof_predictions,axis=1)
        print(cur_oof.shape)
        y_label = self.train[self.label].values
        acc = np.sum(cur_oof == np.array(y_label))/(len(y_label))
        print('global acc ',acc)
        # self.__check_diff_score(oof_predictions,val_idx=val)
        self.predictions = tt_predicts
        self.features_imp = feature_imp
        if self.run is not None:
            self.run['feature_importance'] = self.features_importance()
        if cv_score:
            return oof_predictions,tt_predicts,cv_score_list
        else:
            if self.cv_conf['n']==1:
                # print(oof_predictions.shape,oof_predictions[val].shape,)
                return oof_predictions[val],tt_predicts,val_y.values

            return oof_predictions,tt_predicts

    def xgb_test(self,xgb_params,cv_score=False):
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))

        for n,(trn,val) in enumerate(self.cv_conf['iter']):
            trn_X,trn_y = self.train.loc[trn,self.features],self.train.loc[trn,self.label]
            val_X,val_y = self.train.loc[val,self.features],self.train.loc[val,self.label]

            tr_data = xgb.DMatrix(trn_X, label=trn_y)
            val_X = xgb.DMatrix(val_X,label=val_y)

            watchlist = [(tr_data, 'train'), (val_X, 'eval')]
            estimator = xgb.train(
                xgb_params,
                tr_data,
                evals=watchlist,
                verbose_eval=100,
                num_boost_round=30000,
                # early_stopping_rounds=200,
                early_stopping_rounds=100,
            )

            oof_predictions[val] = estimator.predict(val_X)

            cv_score_list.append(roc_auc_score(y_true=val_y,y_score=estimator.predict(val_X)))
            tt_predicts += estimator.predict(xgb.DMatrix(self.test[self.features])) / self.cv_conf['n']
            if self.run is not None:
                self.run['metrics/test_auc'].log(cv_score_list[-1])

        print(f"training CV oof mean : {np.round(np.mean(cv_score_list),5)}")
        if self.cv_conf['n']==1:
            self.__check_diff_score(oof_predictions,val_idx=val)
        else:
            self.__check_diff_score(oof_predictions)
        # print(imp.sort_values(['split','gain'],ascending=False,ignore_index=True).loc[imp['feature'].isin(self.new_features),:])
        self.predictions = tt_predicts
        if cv_score:
            return oof_predictions, tt_predicts, cv_score_list
        else:
            # if self.cv_conf['n'] == 1:
            return oof_predictions[val],tt_predicts,val_y
            # else:
                # return oof_predictions, tt_predicts

    def cat_test(self,cat_params,cv_score=False,cat_features=None,save_path:str='.'):
        """
        save_path:"/home/lianyhaii"
        """
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))
        if cat_features is None:
            cat_features = [x for x in self.train.select_dtypes(include=['category']).columns.tolist() if x in self.features]

        for c in cat_features:
            self.train[c] = self.train[c].astype('str')
            self.test[c] = self.test[c].astype('str')

        for n,(trn,val) in enumerate(self.cv_conf['iter']):
            trn_X,trn_y = self.train.loc[trn,self.features],self.train.loc[trn,self.label]
            val_X,val_y = self.train.loc[val,self.features],self.train.loc[val,self.label]

            estimator = CatBoostClassifier(**cat_params)
            print(cat_params)
            if os.path.exists(save_path+f'/model_cbt_f{n}.cbt'):
                estimator.load_model(save_path+f'/model_cbt_f{n}.cbt')
            else:
                estimator.fit(
                    trn_X,trn_y,
                    cat_features=cat_features,
                    # early_stopping_rounds=100,
                    # eval_set=[(val_X,val_y)],
                    eval_set=[(val_X,val_y)] if cat_params['task_type'] == 'GPU' else [(trn_X,trn_y),(val_X,val_y)],
                    use_best_model=True,
                    metric_period=200,
                    verbose=True,
                )
                estimator.save_model(save_path+f'/model_cbt_f{n}.cbt')

            oof_predictions[val] = estimator.predict_proba(val_X)[:,1]
            self.model.append(estimator)
            cv_score_list.append(roc_auc_score(y_true=val_y,y_score=estimator.predict_proba(val_X)[:,1]))
            tt_predicts += estimator.predict_proba(self.test[self.features])[:,1] / self.cv_conf['n']

            if self.run is not None:
                self.run['metrics/test_auc'].log(cv_score_list[-1])

        print(f"training CV oof mean : {np.round(np.mean(cv_score_list), 5)}")
        self.__check_diff_score(oof_predictions,val_idx=val)
        self.predictions = tt_predicts
        # self.features_imp = feature_imp
        if self.run is not None:
            self.run['feature_importance'] = self.features_importance()
        if cv_score:
            return oof_predictions,tt_predicts,cv_score_list
        else:
            if self.cv_conf['n']==1:
                # print(oof_predictions.shape,oof_predictions[val].shape,)
                return oof_predictions[val],tt_predicts,val_y.values

            return oof_predictions,tt_predicts

    def sklearn_test(self, sklean_model, cv_score=False, weight=None):
        # self.__deal_cat_features(cat_features,m_type='lgb')
        # imp = pd.DataFrame({
        #     'feature':self.features,
        #     'gain':0,
        #     'split':0,
        # })

        feature_imp = np.zeros(len(self.features))
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))

        for n, (trn, val) in enumerate(self.cv_conf['iter']):
            trn_X, trn_y = self.train.loc[trn, self.features], self.train.loc[trn, self.label]
            val_X, val_y = self.train.loc[val, self.features], self.train.loc[val, self.label]


            estimator = sklean_model.fit(trn_X,trn_y)

            oof_predictions[val] = estimator.predict_proba(val_X)[:,1]
            self.model.append(estimator)

            cv_score_list.append(roc_auc_score(y_true=val_y, y_score=oof_predictions[val]))
            tt_predicts += estimator.predict_proba(self.test[self.features])[:,1] / self.cv_conf['n']
            
            if self.run is not None:
                self.run['metrics/test_auc'].log(cv_score_list[-1])


        print(f"training CV oof mean : {np.round(np.mean(cv_score_list), 5)}")

        self.__check_diff_score(oof_predictions)
        self.predictions = tt_predicts
        self.features_imp = feature_imp

        if self.run is not None:
            self.run['feature_importance'] = self.features_importance()
        if cv_score:
            return oof_predictions,tt_predicts,cv_score_list
        else:
            if self.cv_conf['n']==1:
                # print(oof_predictions.shape,oof_predictions[val].shape,)
                return oof_predictions[val],tt_predicts,val_y.values

            return oof_predictions,tt_predicts

    def find_threshold(self,oof,ylabel,weight_range,method='offline',
    predictions=None,online_thd=None):
        print(method)
        if method == 'offline':
            res = pd.DataFrame({
                'weight': weight_range,
                'oof_f1': 0,
                # 'oof_auc': 0,

            })
            # def search_weight(raw_prob, init_weight=[1.0]*n_classes, step=0.001):
            weight = weight_range.copy()*self.num_class
            step = 0.001
            f_best = accuracy_score(y_pred =np.argmax(oof,axis=1),y_true=ylabel)
            flag_score = 0
            round_num = 1
            # while(f_best < 1.77 and round_num < 5):
            while round_num < 5:
                print('round: ', round_num)
                round_num += 1
                flag_score = f_best
                for c in tqdm(range(self.num_class)):
                    for n_w in range(400, 1400, 10):
                        num = n_w * step
                        new_weight = weight.copy()
                        new_weight[c] = num

                        prob_df = oof.copy()
                        prob_df = prob_df * np.array(new_weight)

                        f = accuracy_score(y_pred=np.argmax(prob_df,axis=1),y_true=ylabel)
                        if f > f_best:
                            weight = new_weight.copy()
                            f_best = f
                            print(f_best)
                            # print(weight)
                            
            return weight




            for n in weight_range:
                oof_f1 = f1_score(y_true=ylabel,y_pred=[int(x>n) for x in oof ])
                # oof_auc = roc_auc_score(y_score=oof,y_true=ylabel)

                # print('the weighted oof auc : ',oof_auc)
                print(f'the weighted {np.round(n,4)} oof f1 {np.round(oof_f1,4)} ')
                # res.loc[res['weight'] == n,'oof_auc'] = oof_auc
                res.loc[res['weight'] == n,'oof_f1'] = oof_f1
            fig = plt.figure(figsize=(8, 6))
            ax2 = fig.add_subplot(111)
            res.index = res['weight']

            res['oof_f1'].plot(ax=ax2, style='y>-.', alpha=0.7, legend=True)
            plt.legend(loc=1)
            plt.show()
            best_threshold = res.loc[res['oof_f1']==res['oof_f1'].max(),'weight'].values
            print(f'the best threshold {best_threshold} and best f1 {res["oof_f1"].max()}')



        elif method == 'ythreshold':

            print(f'train dataset 01 distributions :{sum(ylabel)/len(ylabel)}')
            preds_index = int(len(predictions)*(1-(sum(ylabel)/len(ylabel))))
            sorted_pred = sorted(predictions)
            best_threshold = sorted_pred[preds_index]

            ##
            oof_index = sum(ylabel)
            sorted_oof = sorted(oof)
            best_oof = sorted_oof[oof_index]

            res = f1_score(y_true=ylabel,y_pred=[int(x>=best_oof) for x in oof],average='macro')
            print(f'weight is {best_oof} and test weight {best_threshold} oof was {res}')
        elif method == 'online':
            preds_index = int(len(predictions)*(online_thd/(1-online_thd)))
            sorted_pred = sorted(predictions)
            best_threshold = sorted_pred[preds_index]
            print(f'online wight is {best_threshold}')
            res = 0

        elif method == 'sortedPvalue':
            best_threshold,res = 0,0
            predictions = sorted(list(predictions))

            plt.figure(figsize=(20,10))
            sns.lineplot(predictions,)
            plt.show()
        else:
            raise NotImplementedError

        return best_threshold,res

    def features_importance(self):
        df = pd.DataFrame(self.features_imp, index=self.features,columns=['values'])
        df.sort_values('values',ascending=False,inplace=True,)

        return df

    def analysis_residual(self,oof_predictions):
        self.train['predic_res'] = oof_predictions
        res_train_t = self.train[self.train[self.label] == np.round(oof_predictions)].copy()
        res_train_f = self.train[self.train[self.label] != np.round(oof_predictions)].copy()
        try:
            import os
            os.mkdir('res_analysis')
        except:
            pass
        today = time.strftime("%Y-%m-%d", time.localtime())[5:]
        res_train_f.to_csv(f'./res_analysis/res_train_f_{today}.csv',index=False)
        res_train_t.to_csv(f'./res_analysis/res_train_t_{today}.csv',index=False)

    def submit_seeds_ensembel(self,lgb_params,seed_list):
        """
        the aim of fuction is to find good seed or ensembel seek to submit.

        """
        tt_predictions = np.zeros(len(self.test))
        seed_df = pd.DataFrame({
            'seed':seed_list,
            'oof_score':0,
            'cv_mean':0,
            'cv_std':0,
        })
        final_oof = np.zeros(len(self.train))

        for n,s in enumerate(seed_list):
            # seed_everything(s)
            self.init_CV(seed=s)
            oof_preds,tt_preds,cv_list = self.lgb_test(lgb_params=lgb_params,cv_score=True)
            tt_predictions += tt_preds/len(seed_list)
            final_oof += oof_preds/len(seed_df)

            seed_df.loc[n,'oof_auc'] = roc_auc_score(y_score=oof_preds,y_true=self.train[self.label])
            seed_df.loc[n,'oof_score'] = tpr_weight_funtion(y_predict=oof_preds,y_true=self.train[self.label])
            seed_df.loc[n,'cv_mean'] = np.mean(cv_list)
            seed_df.loc[n,'cv_std'] = np.std(cv_list)
            print(seed_df.loc[n,:],end='\n')

        self.predictions = tt_predictions
        global_auc = roc_auc_score(y_true=self.train[self.label],y_score=final_oof)
        global_tpr = tpr_weight_funtion(y_true=self.train[self.label],y_predict=final_oof)
        self.m_score.append([global_auc,global_tpr])
        seed_df['diff_abs'] = (seed_df['oof_auc'] - seed_df['cv_mean']).abs()
        seed_df.sort_values(['diff_abs','cv_std'],inplace=True)
        return seed_df

    def submit_pseudo_label_model(self,params,seed,model_type):
        # if len(self.predictions) == 0:
        print('training pseudo label .....')
        self.test['label'] = self.predictions.copy()
        test2p = self.test[(self.test['label']<=0.01)|(self.test['label']>=0.95)].copy()
        # relabel
        test2p.loc[test2p['label'] >= 0.5,'label'] = 1
        test2p.loc[test2p['label'] < 0.5,'label'] = 0

        train2p = pd.concat([self.train,test2p],axis=0)
        train2p.reset_index(drop=True,inplace=True)

        tmp_mt = make_test(train2p,self.test,self.features,[],m_score=self.m_score,label=self.label)
        tmp_mt.init_CV(seed)
        if model_type =='lgb':
            tmp_mt.lgb_test(params)
        if model_type == 'xgb':
            tmp_mt.xgb_test(params)

        self.predictions = tmp_mt.predictions

    def submit(self,ID,sub_file=True,threshold=None):
        today = time.strftime("%Y-%m-%d", time.localtime())[5:]
        # self.test[self.label] = [int(x) for x in self.predictions > 0.5]
        if threshold is None:
            self.test[self.label] = self.predictions
        else:
            self.test[self.label] = [int(x >= threshold) for x in self.predictions]
            print('sum of label',np.sum(self.test[self.label]))

        sub_train = self.test[[ID,self.label]].copy()
        if sub_file:
            sub_test = pd.read_csv('./data/submit.csv')
            sub = sub_test[[ID]].merge(sub_train, on=ID, how='left')
        else:
            sub = sub_train.copy()

        # plt.figure(figsize=(20, 10))
        # sns.distplot(sub['label'], bins=100)
        # plt.show()
        print('null in sub','\n',sub.isnull().sum())
        sub.fillna(0, inplace=True)
        score = str(np.round(self.m_score[-1][0],4))+"_"+str(np.round(self.m_score[-1][1],4))
        sub.to_csv(f'./result/sub_{today}_{score}.csv', index=False)
        return sub

    def psm_samples(self,lgb_params):

        ## p值
        tr,tt = self.train.copy(),self.test.copy()
        tr['test_label'] = 0
        tt['test_label'] = 1
        data = pd.concat([tr,tt],ignore_index=True,)
        # train,test = train_test_split(data,test_size=0.1,random_state=0)

        mt = make_test(data,data,self.features,[],m_score=[[0,0,0]],label='test_label')
        mt.init_CV(0)
        oof , test = mt.lgb_test(lgb_params)
        data['oof'] = oof
        ## 可视化
        # tmp_df = self.train[[self.label,'oof']]
        sns.boxplot(x='label',y='oof',data=data)
        plt.show()
        ## 找特征重要性
        imp_df = mt.features_importance()
        imp_df.reset_index(drop=False,inplace=True)
        imp_df.columns = ['features','values']
        sns.barplot(y='features',x='values',data=imp_df)
        plt.show()
        print(imp_df.head(10))
        return data,imp_df

    def submit_easy_ensemble(self,n_group=10,):
        pass

    def tune_model(self,params,model,type='binary',N_TRIALS=100,model_type='lgb'):

        def objective(trial):
            if model_type == 'lgb':
                obj_params = {
                    # 'random_state': params['seed'],
                    'metric': params['metric'],
                    'n_estimators': 5000,
                    'n_jobs': -1,
                    'seed':params['seed'],
                    'early_stopping_rounds': 100,
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
                    'max_depth': trial.suggest_int('max_depth', 6, 16),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 2**8),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.9),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                    'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.3, 0.9),
                    'max_bin': trial.suggest_int('max_bin', 128, 1024),
                    'min_data_per_group': trial.suggest_int('min_data_per_group', 50, 200),
                    'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),
                    'cat_l2': trial.suggest_int('cat_l2', 1, 20),
                    'verbose': -1,
                      }

            elif model_type == 'xgb':
                obj_params = {
                    'objective': params['objective'],
                    # 'task_type': 'GPU',
                    'nthread': -1,
                    'seed': params['seed'],
                    'tree_method': 'hist',
                    'eval_metric': params['eval_metric'],
                    'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                    'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                    'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
                    'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
                    'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02]),
                    # 'n_estimators': 10000,
                    'max_depth': trial.suggest_int('max_depth', 5,14),
                    # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                }

            else:
                raise NotImplementedError
            oof, _ = model(obj_params, cv_score=False,)
            score = 0
            if type == 'binary':
                score = roc_auc_score(y_true=self.train[self.label],y_score=oof)
            if type == 'f1':
                score = f1_score(y_true=self.train[self.label],y_pred=[int(x>0.5) for x in oof],average='macro')

            return score

        study = optuna.create_study(study_name=f"optimization", direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)
        ## save log
        print(study.trials_dataframe())
        import os
        if not os.path.exists('user_data'):
            os.mkdir('user_data')

        study.trials_dataframe().to_csv('./user_data/' + f"{type}trial_parameters.csv", index=False)
        with open('./user_data/' + f'{type}_study.pickle', 'wb') as f:
            pickle.dump(study, f)

        fig = plot_param_importances(study,)
        plotly.offline.plot(fig, filename='imp_fig.html')
        fig = plot_optimization_history(study)
        plotly.offline.plot(fig, filename='history_fig.html')

        return study.best_trial.params
