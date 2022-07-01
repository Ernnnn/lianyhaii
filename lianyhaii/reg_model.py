# coding:utf-8
import time

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from scipy.special import inv_boxcox
from scipy.stats import boxcox

from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_recall_curve, auc, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
# import pmdarima as pm
# from statsmodels.gam.tests.test_gam import sigmoid
from tqdm import tqdm
import xgboost as xgb

from lianyhaii.metrics import pymetric
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args


def inv_yeojohnson(x, lambda_fitted):
    if lambda_fitted == 0:
        res = np.zeros(len(x))
        for index, value in enumerate(x):
            if value >= 0:
                y = np.exp(value) - 1
            else:
                y = -np.exp(-value) + 1
            res[index] = y
        return res
    if lambda_fitted != 0:
        res = np.zeros(len(x))
        for index, value in enumerate(x):
            if value >= 0:
                y = ((value * lambda_fitted) + 1.0) ** (1.0 / lambda_fitted) - 1.0
            else:
                y = -((-value * (2.0 - lambda_fitted) + 1.0) ** (1.0 / (2.0 - lambda_fitted)) - 1.0)
            res[index] = y
        return res


# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class GroupTimeSeriesSplit(_BaseKFold):
    """
    Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    Examples
    --------
    # >>> import numpy as np
    # >>> from sklearn.model_selection import GroupTimeSeriesSplit
    # >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
    #                        'b', 'b', 'b', 'b', 'b',\
    #                        'c', 'c', 'c', 'c',\
    #                        'd', 'd', 'd'])
    # >>> gtss = GroupTimeSeriesSplit(n_splits=3)
    # >>> for train_idx, test_idx in gtss.split(groups, groups=groups):
    # ...     print("TRAIN:", train_idx, "TEST:", test_idx)
    # ...     print("TRAIN GROUP:", groups[train_idx],\
    #               "TEST GROUP:", groups[test_idx])
    # TRAIN: [0, 1, 2, 3, 4, 5] TEST: [6, 7, 8, 9, 10]
    # TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a']\
    # TEST GROUP: ['b' 'b' 'b' 'b' 'b']
    # TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] TEST: [11, 12, 13, 14]
    # TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b']\
    # TEST GROUP: ['c' 'c' 'c' 'c']
    # TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\
    # TEST: [15, 16, 17]
    # TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c']\
    # TEST GROUP: ['d' 'd' 'd']
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None,
                 test_size=None,
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        ## 将id分配到group_dic里面。
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]

        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        if self.test_size is None:
            group_test_size = n_groups // n_folds
        else:
            group_test_size = self.test_size

        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)

        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end -
                                          self.max_train_size:train_end]
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                concated_test = np.concatenate((test_array,test_array_tmp))
                concated_test = np.unique(concated_test,axis=None)
                test_array = np.sort(concated_test, axis=None)
            yield [int(i) for i in train_array], [int(i) for i in test_array]


class reg_test():
    def __init__(self, tr_df, tt_df, base_features, new_features,
                 m_score, label,
                 transform_type=None, fitted_param=None,
                 metrices=None,
                 ):
        self.train = tr_df
        self.test = tt_df
        self.base_features = base_features
        self.new_features = new_features
        self.m_score = m_score
        self.label = label
        self.features = base_features + new_features
        self.predictions = None
        self.model = []
        self.metrices = metrices
        self.feature_imp = None


        self.__transform_type = transform_type
        self.__fitted_param = fitted_param

    def init_CV(self, seed, n_split=5, shuffle=True, CV_type='kFold',group_col=None):
        self.cv_conf = {}
        # print(self.label)
        self.train['_group_skf'] = pd.cut(self.train[self.label], n_split, labels=False)

        if CV_type == 'kFold':
            cv = KFold(n_splits=n_split, shuffle=shuffle, random_state=seed)
            self.cv_conf['iter'] = cv.split(X=self.train,groups=self.train[group_col])
            self.cv_conf['n'] = n_split

        if CV_type == 'bioKFold':
            cv = KFold(n_splits=n_split, shuffle=shuffle, random_state=seed)
            self.cv_conf['iter'] = cv.split(X=self.train,groups=self.train['_group_skf'])
            self.cv_conf['n'] = n_split

        ## 时间序列类！！
        if CV_type == 'lastFold':
            assert not (group_col is None),'lack group col'
            group_block = sorted(self.train[group_col].unique())
            ## train_index,test_index
            cv = [[self.train[self.train[group_col] < group_block[-1]].index,
                 self.train[(self.train[group_col] == group_block[-1])].index]]
            self.cv_conf['iter'] = cv
            self.cv_conf['n'] = 1

        if CV_type == 'groupTFold':
            assert not (group_col is None),'lack group col'
            cv = GroupTimeSeriesSplit(n_splits=n_split,max_train_size=360,test_size=91)
            self.cv_conf['iter'] = cv.split(X=self.train,groups=self.train[group_col])
            self.cv_conf['n'] = cv.n_splits

        if CV_type == 'submitFold':
            assert not (group_col is None),'lack group col'
            group_block = sorted(self.train[group_col].unique())
            ## train_index,test_index
            ## 使用所有训练集进行提交!
            cv = [[self.train[self.train[group_col] <= group_block[-1]].index,
                 self.train[(self.train[group_col] == group_block[-1])]]]
            self.cv_conf['iter'] = cv
            self.cv_conf['n'] = 1


    def __check_diff_score(self,oof_predictions,trn_idx=None):
        if trn_idx is None:
            pm = pymetric(y_true=self.train[self.label],y_pred=oof_predictions)
        else:
            pm = pymetric(y_true=self.train.loc[trn_idx,self.label],y_pred=oof_predictions)


        result_score = pm.gen_metric_dict(metric_names=self.metrices,th=0.5)
        for key,value in result_score.items():
            print(f'global {key} : {value}')

        print('='*10+'different with previous version'+'='*10)
        score_list = []
        for n,(key,value) in enumerate(result_score.items()):
            print(f'diff of {key} : {np.round(value-self.m_score[-1][n],5)}')
            score_list.append(value)

        self.m_score.append(score_list)

    def __re_reg(self, x):

        if self.__transform_type == 'ln':
            return np.exp(x) + self.__fitted_param - 1

        elif self.__transform_type == 'boxcox':
            return inv_boxcox(x, self.__fitted_param['lambda']) + self.__fitted_param['min_values'] - 1
        elif self.__transform_type == 'yeojohnson':

            return inv_yeojohnson(x, lambda_fitted=self.__fitted_param)
        else:
            return x
    def features_importance(self):
        df = pd.DataFrame(self.features_imp, index=self.features,columns=['values'])
        df.sort_values('values',ascending=False,inplace=True,)
        return df
    def lgb_test(self, lgb_params, cv_score=False, two_step=False):

        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))
        feature_imp = np.zeros(len(self.features))

        for n, (trn, val) in enumerate(self.cv_conf['iter']):
            print(f'==== training fold {n+1} ====')
            trn_X, trn_y = self.train.loc[trn, self.features], self.train.loc[trn, self.label]
            val_X, val_y = self.train.loc[val, self.features], self.train.loc[val, self.label]

            trn_data = lgb.Dataset(trn_X, label=trn_y,free_raw_data=False)
            val_data = lgb.Dataset(val_X, label=val_y,free_raw_data=False)

            estimator = lgb.train(lgb_params,
                                  trn_data,
                                  valid_sets=[trn_data, val_data],
                                  # feval=lgb_f1_score,
                                  # feval=tpr_weight_3_cunstom,
                                  verbose_eval=300,
                                  )

            if two_step:
                lgb_params_copy = lgb_params.copy()
                lgb_params_copy['learning_rate'] = 0.001
                lgb_params_copy['n_estimators'] = 40000
                lgb_params_copy['early_stopping_rounds'] = 1000
                estimator = lgb.train(lgb_params_copy,
                                      trn_data,
                                      valid_sets=[trn_data, val_data],
                                      init_model=estimator,
                                      verbose_eval=-1)

            oof_predictions[val] = self.__re_reg(estimator.predict(val_X))
            feature_imp += estimator.feature_importance(importance_type='split') / self.cv_conf['n']

            cv_score_list.append(mean_absolute_error(y_true=self.__re_reg(val_y), y_pred=oof_predictions[val]))
            tt_predicts += self.__re_reg(estimator.predict(self.test[self.features])) / self.cv_conf['n']
            if self.cv_conf['n'] == 1 :
                self.__check_diff_score(oof_predictions[val],trn_idx=val)

        print(f"training val mean score : {np.mean(cv_score_list)}")
        if self.cv_conf['n'] != 1:
            self.__check_diff_score(oof_predictions)

        # print(imp.sort_values(['split','gain'],ascending=False,ignore_index=True).loc[imp['feature'].isin(self.new_features),:])
        self.predictions = tt_predicts
        self.feature_imp = feature_imp
        if cv_score:
            return oof_predictions, tt_predicts, cv_score_list
        else:
            return oof_predictions, tt_predicts

    def xgb_test(self, xgb_params, cv_score=False):
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))

        for n, (trn, val) in enumerate(self.cv_conf['iter']):
            trn_X, trn_y = self.train.loc[trn, self.features], self.train.loc[trn, self.label]
            val_X, val_y = self.train.loc[val, self.features], self.train.loc[val, self.label]

            tr_data = xgb.DMatrix(trn_X, label=trn_y)
            val_X = xgb.DMatrix(val_X, label=val_y)

            watchlist = [(tr_data, 'train'), (val_X, 'eval')]
            estimator = xgb.train(
                xgb_params,
                tr_data,
                evals=watchlist,
                verbose_eval=1000,
                # num_boost_round=30000,
                # early_stopping_rounds=200,
                early_stopping_rounds=100,
            )

            tt_predicts += estimator.predict(xgb.DMatrix(self.test[self.features])) / self.cv_conf['n']
            oof_predictions[val] = self.__re_reg(estimator.predict(val_X))

            cv_score_list.append(mean_absolute_error(y_true=self.__re_reg(val_y), y_pred=oof_predictions[val]))
            tt_predicts += self.__re_reg(estimator.predict(self.test[self.features])) / self.cv_conf['n']
        self.__check_diff_score(oof_predictions)
        # print(imp.sort_values(['split','gain'],ascending=False,ignore_index=True).loc[imp['feature'].isin(self.new_features),:])
        self.predictions = tt_predicts

        if cv_score:
            return oof_predictions, tt_predicts, cv_score_list
        else:
            return oof_predictions, tt_predicts

    def cat_test(self, cat_params, cv_score=False):
        cv_score_list = []
        oof_predictions = np.zeros(len(self.train))
        tt_predicts = np.zeros(len(self.test))
        cat_features = [x for x in self.train.select_dtypes(include=['category']).columns.tolist() if
                        x in self.features]
        print(cat_features)

        for col in cat_features:
            self.train[col] = pd.to_numeric(self.train[col], errors='ignore', downcast='integer').fillna(-99).astype(
                int)
            self.test[col] = pd.to_numeric(self.test[col], errors='ignore', downcast='integer').fillna(-99).astype(int)

        # print(self.train[cat_features].head())
        from catboost import CatBoostRegressor
        for n, (trn, val) in enumerate(self.cv_conf['iter']):
            trn_X, trn_y = self.train.loc[trn, self.features], self.train.loc[trn, self.label]
            val_X, val_y = self.train.loc[val, self.features], self.train.loc[val, self.label]

            estimator = CatBoostRegressor(**cat_params)
            estimator.fit(
                trn_X, trn_y,
                cat_features=cat_features,
                # early_stopping_rounds=100,
                eval_set=[(val_X,val_y)] if cat_params['task_type'] == 'GPU' else [(trn_X,trn_y),(val_X,val_y)],
                use_best_model=True,
                metric_period=500,
                verbose=True,
            )

            tt_predicts += estimator.predict((self.test[self.features])) / self.cv_conf['n']
            oof_predictions[val] = self.__re_reg(estimator.predict(val_X))

            cv_score_list.append(mean_absolute_error(y_true=self.__re_reg(val_y), y_pred=oof_predictions[val]))
            tt_predicts += self.__re_reg(estimator.predict(self.test[self.features])) / self.cv_conf['n']

        self.__check_diff_score(oof_predictions)

        self.predictions = tt_predicts
        if cv_score:
            return oof_predictions, tt_predicts, cv_score_list
        else:
            return oof_predictions, tt_predicts

    def submit_seeds_ensembel(self, lgb_params, seed_list):
        """
        the aim of fuction is to find good seed or ensembel seek to submit.

        :param cat_features:
        :param lgb_params:
        :param seed_list:
        :return:
        """
        tt_predictions = np.zeros(len(self.test))
        seed_df = pd.DataFrame({
            'seed': seed_list,
            'oof_score': 0,
            'cv_mean': 0,
            'cv_std': 0,
        })
        final_oof = np.zeros(len(self.train))

        for n, s in enumerate(seed_list):
            # seed_everything(s)
            self.init_CV(seed=s)
            oof_preds, tt_preds, cv_list = self.lgb_test(lgb_params=lgb_params, cv_score=True)
            tt_predictions += tt_preds / len(seed_list)
            final_oof += oof_preds / len(seed_df)

            seed_df.loc[n, 'oof_auc'] = roc_auc_score(y_score=oof_preds, y_true=self.train[self.label])
            # seed_df.loc[n,'oof_score'] = tpr_weight_funtion(y_predict=oof_preds,y_true=self.train[self.label])
            seed_df.loc[n, 'cv_mean'] = np.mean(cv_list)
            seed_df.loc[n, 'cv_std'] = np.std(cv_list)
            print(seed_df.loc[n, :], end='\n')

        self.predictions = tt_predictions
        global_auc = roc_auc_score(y_true=self.train[self.label], y_score=final_oof)
        # global_tpr = tpr_weight_funtion(y_true=self.train[self.label],y_predict=final_oof)
        # self.m_score.append([global_auc,global_tpr])
        return seed_df

    def submit_retrain(self,model):
        pass

    def submit_pseudo_label_model(self, params, seed, model_type):
        # if len(self.predictions) == 0:
        print('training pseudo label .....')
        self.test['label'] = self.predictions.copy()
        test2p = self.test[(self.test['label'] <= 0.01) | (self.test['label'] >= 0.99)].copy()
        # relabel
        test2p.loc[test2p['label'] >= 0.5, 'label'] = 1
        test2p.loc[test2p['label'] < 0.5, 'label'] = 0

        train2p = pd.concat([self.train, test2p], axis=0)
        train2p.reset_index(drop=True, inplace=True)

        tmp_mt = reg_test(train2p, self.test, self.features, [], m_score=self.m_score, label=self.label)
        tmp_mt.init_CV(seed)
        if model_type == 'lgb':
            tmp_mt.lgb_test(params)
        if model_type == 'xgb':
            tmp_mt.xgb_test(params)

        self.predictions = tmp_mt.predictions

    def submit(self, ID, sub_label):
        today = time.strftime("%Y-%m-%d", time.localtime())[5:]
        self.test[sub_label] = self.predictions
        sub = self.test[[ID, sub_label]].copy()
        plt.figure(figsize=(20, 10))
        sns.distplot(sub[sub_label], bins=100)
        plt.show()
        score = str(np.round(self.m_score[-1][0], 4)) + "_" + str(np.round(self.m_score[-1][1], 4))
        sub.to_csv(f'./result/sub_{today}_{score}.csv', index=False)
