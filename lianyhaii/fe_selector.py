# coding:utf-8
# numpy and pandas for data manipulation
import pandas as pd
import numpy as np

# model used for feature importances
import lightgbm as lgb

# utility for early stopping with a validation set
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# memory management
import gc

# utilities
from itertools import chain

from tqdm import tqdm


class FeatureSelected():
    """
    now contain four way to select features
    :corr with base
    :corr with group
    :null importances
    :std lower
    """

    def __init__(self,train,test,feats,label):
        self.train = train
        self.test = test
        self.label = label
        self.features = feats
        self.adv_drop_feats = []
        self.ks_drop_feats = []
        self.corr_with_base_drop_feats = []
        self.corr_with_groups_drop_feats = []
        self.std_drop_feats = []
        self.null_drop_feats = []
        self.pmt_drop_feats = []
        self.tt_drop_feats = []

    ## 分布差异筛选 by KS test
    def ks_test_selector(self,threshold=0.001,plot_it = False):
        from scipy.stats import ks_2samp
        tt_KS_p = []
        for col in self.features:

            p = np.round(ks_2samp(self.train[col], self.test[col])[1], 4)
            tt_KS_p.append(p)
            if p < threshold:
                self.ks_drop_feats.append(col)

        if plot_it:
            KS = pd.DataFrame({
                'value':tt_KS_p,
                'feature':self.features,
            }).sort_values('value',ascending=True)
            sns.barplot(x='value',y='feature',data=KS.head(20),)
            plt.title('KS p_value feature selector')
            plt.show()
        
    ## 分布差异筛选 by abv test
    def adv_test_selector(self,threshold=0.10,plot_it=False):
        X_train = self.train.copy()
        X_test = self.test.copy()

        features = self.features.copy()
        X_train['target'] = 1
        X_test['target'] = 0
        train_test = pd.concat([X_train, X_test], axis=0, ignore_index=True)

        train1, test1 = train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)
        train_y = train1['target'].values
        test_y = test1['target'].values
        del train1['target'], test1['target']

        if 'target' in features:
            features.remove('target')

        adversarial_result = pd.DataFrame(index=features, columns=['roc'])
        for i in tqdm(features):
            clf = lgb.LGBMClassifier(
                random_state=47,
                max_depth=4,
                boosting_type='gbdt',
                metric='auc',
                n_estimators=1000,
                importance_type='gain'
            )
            clf.fit(
                np.array(train1[i]).reshape(-1, 1),
                train_y,
                eval_set=[(np.array(test1[i]).reshape(-1, 1), test_y)],
                early_stopping_rounds=100,
                verbose=0)
            temp_pred = clf.predict_proba(np.array(test1[i]).reshape(-1, 1))[:, 1]
            roc1 = roc_auc_score(test_y, temp_pred)
            adversarial_result.loc[i, 'roc'] = roc1
        ## all columns
        clf = lgb.LGBMClassifier(
            random_state=47,
            max_depth=2,
            metric='auc',
            n_estimators=1000,
            importance_type='gain'
        )
        clf.fit(train1[features],train_y,eval_set=(test1[features],test_y),early_stopping_rounds=200,verbose=0)
        temp_pred = clf.predict_proba(test1[features])[:,1]
        roc1 = roc_auc_score(test_y,temp_pred)
        adversarial_result.loc['all','roc'] = roc1

        adversarial_result = adversarial_result.sort_values('roc', ascending=False).reset_index()
        adversarial_result.columns = ['feature','value']
        if plot_it:
            plt.figure(figsize=(10,20))
            sns.barplot(x='value',y='feature',data=adversarial_result)
            plt.title('This abversarial test about features')
            plt.show()
        print(adversarial_result)
        drop_feats = adversarial_result.loc[(adversarial_result['value'] - 0.50).abs()>threshold,'feature'].tolist()
        # print(drop_feats)
        self.adv_drop_feats += drop_feats
    ## Permutation importance
    def pmt_imp_selector(self,threshold=0.01,plot_it=False):
        """
        the diff of score more high, the more reason to remove it.
        :param threshold: absolutely score of feature.
        :param plot_it: Whether plot it
        :return: None
        """
        def permutation_importance(X, y, model):
            perm = []
            y_true = model.predict_proba(X)[:, 1]
            baseline = roc_auc_score(y, y_true)
            for cols in X.columns:
                value = X[cols].copy()
                X[cols] = X[cols].sample(frac=1).values
                y_true = model.predict_proba(X)[:, 1]
                perm.append(roc_auc_score(y, y_true) - baseline)
                X[cols] = value
            return perm

        model = lgb.LGBMClassifier(num_leaves=2**8,max_depth=8,learning_rate=0.05,n_estimators=1000
                                   ,n_jobs=-1,subsample=0.8,colsample_bytree=0.8)
        tr_df, tt_df = train_test_split(self.train, test_size=0.33, random_state=1004)
        model.fit(tr_df[self.features], tr_df[self.label],)
        perm_df = pd.DataFrame({
            'feature':self.features,
        })

        for _ in tqdm(range(5)):
            perm = permutation_importance(X=tt_df[self.features],y=tt_df[self.label],model=model)
            perm_df['perm_'+str(_)] = perm
        perm_df['mean'] = perm_df.iloc[:,perm_df.columns.str.contains('perm')].mean(1)
        perm_df['std'] = perm_df.iloc[:,perm_df.columns.str.contains('perm')].std(1)
        drop_feats = perm_df.loc[perm_df['mean'] < threshold,'feature'].tolist()
        perm_df.sort_values('mean',ascending=True,inplace=True,ignore_index=True)

        print(perm_df)
        self.pmt_drop_feats += drop_feats

    def corr_with_base_selector(self,base_features,threshold=0.999,plot_it=False):
        """ only for numerical feature"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import LabelEncoder
        lr = LinearRegression(n_jobs=-1,normalize=True)
        tr,vl = train_test_split(self.train.dropna(),test_size=0.3,random_state=1004)
        cat_feats = [x for x in base_features if tr[x].dtype in ['O']]
        for f in cat_feats:
            le = LabelEncoder()
            tr[f] = le.fit(tr[f].fillna(0))
            vl[f] = le.transform(vl[f].fillna(0))

        tt_corr_r2 = []
        for f in self.features:
            lr.fit(tr[base_features],tr[f],)
            y_pred = lr.predict(vl[base_features])
            corr_r2 = r2_score(y_true=vl[f],y_pred=y_pred)
            tt_corr_r2.append(corr_r2)
            if corr_r2 > threshold:
                self.corr_with_base_drop_feats.append(f)

        if plot_it:
            corr = pd.DataFrame({
                'value':tt_corr_r2,
                'feature':self.features,
            }).sort_values('value',ascending=False)
            sns.barplot(x='value',y='feature',data=corr.head(20),)
            plt.title('R2 score with base feature selector')
            plt.show()

    def corr_with_groups_selector(self,threshold=0.99,plot_it=False):
        """
        this fuction is to remove the features with high corr.
        1. get the corr with target and get the score list
        2. sort the list ascending
        3. for f in the list,remove it if f is high corr with the rest list .
        :param threshold: the threshold of r2 score
        :param plot_it: whether to plot it
        :return: None
        """

        score_list = self.train[self.features].corrwith(self.train[self.label]).reset_index()
        score_list.columns = ['feature','corr']
        score_list['abs_corr'] = score_list['corr'].abs()
        score_list.sort_values('abs_corr',ascending=True,inplace=True)
        if plot_it:
            plt.figure(figsize=(8,16))
            sns.barplot(y='feature',x='abs_corr',data=score_list)
            plt.title('feature correlation with target')
            plt.show()
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import LabelEncoder
        lr = LinearRegression(n_jobs=-1, normalize=True)
        tr, vl = train_test_split(self.train.dropna(), test_size=0.3, random_state=1004)
        cat_feats = [x for x in self.features if tr[x].dtype in ['O']]
        for f in cat_feats:
            le = LabelEncoder()
            tr[f] = le.fit(tr[f].fillna(0))
            vl[f] = le.transform(vl[f].fillna(0))
        tt_corr_r2 = []
        corr_features = score_list['feature'].unique().tolist().copy()
        low_corr_features = score_list['feature'].unique().tolist().copy()

        for n,f in enumerate(corr_features):
            low_corr_features.remove(f)
            lr.fit(tr[low_corr_features], tr[f],)
            y_pred = lr.predict(vl[low_corr_features])
            corr_r2 = r2_score(y_true=vl[f], y_pred=y_pred)
            tt_corr_r2.append(corr_r2)
            if corr_r2 > threshold:
                self.corr_with_groups_drop_feats.append(f)
            else:
                low_corr_features.append(f)

        if plot_it:
            corr = pd.DataFrame({
                'value': tt_corr_r2,
                'feature': corr_features,
            }).sort_values('value', ascending=False)
            plt.figure(figsize=(6,18))

            sns.barplot(x='value', y='feature', data=corr, )
            plt.title('R2 score with groups feature selector')
            plt.show()


    def std_selector(self,threshold=0.01,plot_it=False):
        """
        1.改进了方差受均值影响的bug
        2.改进了变异系数在均值较小时取值波动非常大的bug
        3.修复了有负数的情况出现。
        """
        std_df = pd.DataFrame({
            'feature':self.features,
            'mean':self.train[self.features].mean(),
            'std':self.train[self.features].std(),
            'q1':self.train[self.features].quantile(0.25),
            'q3':self.train[self.features].quantile(0.75),
        })

        std_df['coef_var'] = (std_df['std']*100 / (std_df['mean']+0.00001)).abs()
        std_df['coef_q'] = ((std_df['q3'] - std_df['q1'])*100 / (std_df['q1']+std_df['q3'])).abs()

        std_df.sort_values(['coef_var', 'coef_q'], ascending=True, inplace=True)

        low_var_mask = std_df['coef_var'] < threshold
        low_q_mask = std_df['coef_var'] < threshold

        if plot_it:
            fig,axes = plt.subplots(1,2,sharey=True)
            sns.barplot(y='feature',x='coef_var',data=std_df.head(20),ax=axes[0])
            axes[0].set_title('This is coef of var')
            sns.barplot(y='feature',x='coef_q',data=std_df.head(20),ax=axes[1])
            axes[1].set_title('This is Quartile coef')
            plt.show()


        self.std_drop_feats += std_df.loc[low_q_mask&low_var_mask,'feature'].tolist()

    def __get_feature_importances(self, shuffle, seed=None):
        # Gather real features
        data = self.train.copy()
        train_features = self.features.copy()
        # Go over fold and keep track of CV score (train and valid) and feature importances
        cat_feats = [x for x in train_features if data[x].dtype in ['O']]
        for f in cat_feats:
            data[f] = data[f].astype('category')

        # Shuffle target if required
        y = data[self.label].copy()
        if shuffle:
            # Here you could as well use a binomial distribution
            y = data[self.label].copy().sample(frac=1.0)

        # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
        dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
        lgb_params = {
            'objective': 'binary',
            'boosting_type': 'rf',
            'subsample': 0.623,
            'colsample_bytree': 0.7,
            'num_leaves': 127,
            'max_depth': 8,
            'verbose': -1,
            'seed': seed,
            'bagging_freq': 1,
            'n_jobs': 4
        }

        # Fit the model
        clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=cat_feats)

        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(importance_type='split')
        imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))

        return imp_df

    def __get_null_importances(self,shuffle,runs_num=80,seed=None):
        null_imp_df = pd.DataFrame()
        nb_runs = runs_num
        import time
        start = time.time()
        dsp = ''
        for i in range(nb_runs):
            # Get current run importances
            imp_df = self.__get_feature_importances(shuffle=shuffle,seed=seed)
            imp_df['run'] = i + 1
            # Concat the latest importances with the old ones
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
            # Erase previous message
            for l in range(len(dsp)):
                print('\b', end='', flush=True)
            # Display current run and time used
            spent = (time.time() - start) / 60
            dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
            print(dsp, end='', flush=True)
        return null_imp_df

    def null_importance_selector(self,runs_num=80,seed=None,plot_it=False):
        actual_imp_df = self.__get_feature_importances(shuffle=False,seed=seed)
        null_imp_df = self.__get_null_importances(shuffle=True,runs_num=runs_num,seed=seed)
        correlation_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
            gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
            split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))

        corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
        corr_scores_df.sort_values(['split_score','gain_score'],ascending=True,inplace=True)

        corr_scores_df.index = corr_scores_df['feature']
        split_zero_mask = corr_scores_df['split_score'] == 0
        gain_zero_mask = corr_scores_df['gain_score'] == 0

        if plot_it:
            fig,axes = plt.subplots(1,2,sharey=True)
            sns.barplot(y='feature',x='gain_score',data=corr_scores_df.head(20),ax=axes[0])
            axes[0].set_title('GAIN SCORE')
            sns.barplot(y='feature',x='split_score',data=corr_scores_df.head(20),ax=axes[1])
            axes[1].set_title('SPLIT SCORE')
            plt.show()


        self.null_drop_feats += corr_scores_df[split_zero_mask&gain_zero_mask].index.tolist()

    def remove_by_all(self):
        self.tt_drop_feats += list(set(self.corr_with_base_drop_feats +
                                       self.corr_with_groups_drop_feats +
                                       self.std_drop_feats +
                                       self.ks_drop_feats +
                                       self.null_drop_feats +
                                       self.adv_drop_feats))
        print(f'\nwill totally drop {len(self.tt_drop_feats)} num of features')
        print('the features are:',self.tt_drop_feats)

