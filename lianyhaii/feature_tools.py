# coding:utf-8
import gc
import itertools
import multiprocessing

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from pandas.core.indexes.api import all_indexes_same
# from gokinjo import knn_kfold_extract
from scipy.stats import entropy, chi2
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold
from itertools import product

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm





class encode_cat_feats:
    """helper to encode cat features"""

    def __init__(self, train, test, cols, label):
        """
        decide how to encode cat features!
        * frequency encoding
        * one hot encoding
        * label encoding
        * category encoding
        * target encoding
        * knn mean encoding
        * word2vec encoding


        :param train:
        :param test:
        :param encode_type:
        """
        self.train = train
        self.test = test
        # self.encode_type = encode_type
        self.cat_features = cols
        self.new_features = []
        self.label = label

    def __freq_encode(self, df1, df2, cols):
        add_features = []
        for col in cols:
            df = pd.concat([df1[col], df2[col]])
            vc = df.value_counts(dropna=False, normalize=False).to_dict()
            # if np.isnan(vc[-1]):
            #     vc[-1] = -1

            nm = col + '_FrqEnc'
            df1[nm] = df1[col].map(vc)
            df1[nm] = df1[nm].astype('float32')
            df2[nm] = df2[col].map(vc)
            df2[nm] = df2[nm].astype('float32')
            print(nm, ', ', end='\n')
            add_features.append(nm)
        return add_features
    def encode_freq(self):
        return self.__freq_encode(self.train, self.test, self.cat_features)

    def __one_hot_encoding(self, tr_df, tt_df, cols,method='inner'):
        keep_cols_tr = tr_df[cols].copy()
        keep_cols_tt = tt_df[cols].copy()
        tr_df = pd.get_dummies(tr_df, prefix_sep='_ohe_enc_', columns=cols)
        tt_df = pd.get_dummies(tt_df, prefix_sep='_ohe_enc_', columns=cols)
        y_train = tr_df[self.label]

        tr_df, tt_df = tr_df.align(tt_df, join=method, axis=1,fill_value=0)
        tr_df[self.label] = y_train
        tr_df[cols] = keep_cols_tr
        tt_df[cols] = keep_cols_tt
        ohe_features = [x for x in tr_df if '_ohe_enc' in x]
        # ohe_features = [x for x in ohe_features for j in cols if j in x]
        return tr_df, tt_df, ohe_features

    def encode_ohe(self,method='inner'):
        # print(self.train.shape)
        self.train, self.test, ohe_feats = self.__one_hot_encoding(self.train, self.test, self.cat_features,method)
        # print(self.train.shape)
        return ohe_feats

    def __label_encoding(self, train, test, cols,inplace=False,keep_na=False,method='dict'):
        """

        :param train:
        :param test:
        :param cols:
        :param inplace:
        :param keep_na:
        :param method: dict OR pandas
        :return:
        """

        lb_feats = []
        for col in cols:
            if not inplace:
                f_name = f'{col}_lb_enc'
            else:
                f_name = col
            tmp_df = pd.concat([train[col],test[col]],ignore_index=True,)

            if method == 'dict':
                unique_list = tmp_df.unique()
                x_dict =  dict(zip(unique_list,range(len(unique_list))))

                for x in x_dict:
                    if pd.isnull(x):
                        if keep_na:
                            x_dict[x] = np.nan
                        else:
                            x_dict[x] = -1

                train[f_name] = train[col].map(x_dict)
                test[f_name] = test[col].map(x_dict)
            elif method == 'skl':
                le = LabelEncoder()
                le.fit(tmp_df.unique())
                train[f_name] = le.transform(train[col])
                test[f_name] = le.transform(test[col])
            elif method == 'pandas':
                tmp_df,_ = pd.factorize(tmp_df,)
                train[f_name] = tmp_df[:train.shape[0]]
                test[f_name] = tmp_df[train.shape[0]:]

            else:
                raise ValueError('no this method',method)
            lb_feats.append(f_name)

        return lb_feats
    def encode_lb(self,inplace=False,keep_na=False,method='dict'):
        return self.__label_encoding(self.train, self.test, self.cat_features,inplace,keep_na,method)

    def __category_encoding(self, train, test, cols,inplace=False):
        category_feats = []
        for col in cols:
            col_cat = pd.Categorical(train[col].append(test[col]))
            # col_cat = col_cat.dtype 
            if not inplace:
                f_name = f"{col}_cate_enc"
            else:
                f_name = col

            train[f_name] = train[col].astype(col_cat.dtype)
            test[f_name] = test[col].astype(col_cat.dtype)
            category_feats.append(f_name)
        return category_feats
    def encode_ctg(self,inplace=False):
        return self.__category_encoding(self.train, self.test, self.cat_features,inplace)

    def __knn_encoding(self, train, test, cols, nn_n,nfold=5):

        new_feats = []

        skf = StratifiedKFold(n_splits=nfold,shuffle=True, random_state=0)
        # for nn in nn_n:
        oof_knn = np.zeros(train.shape[0])
        tt_knn = np.zeros(test.shape[0])

        for n,(trn_id,val_id) in enumerate(skf.split(train[cols],train[self.label])):
            print(f'==== kfold knn features making fold {n} ====')
            trn_x,trn_y = train.loc[trn_id,cols],train.loc[trn_id,self.label]
            val_x,val_y = train.loc[val_id,cols],train.loc[val_id,self.label]

            knc = KNeighborsClassifier(n_neighbors=nn_n,)
            knc.fit(trn_x,trn_y)

            oof_knn[val_id] = knc.predict_proba(val_x)[:,1]
            tt_knn += knc.predict_proba(test[cols])[:,1] / skf.n_splits
        print(f'auc of this feature is {roc_auc_score(y_true=train[self.label],y_score=oof_knn)}')
        f_name = '_'.join(cols)+ f"{nn_n}_knn_encoding"
        train[f_name] = oof_knn
        test[f_name] = tt_knn
        new_feats.append(f_name)
        return new_feats
    def encode_knn(self, nn_n,nfold=5):
        print('encoding knn with',nn_n)
        return self.__knn_encoding(self.train, self.test, self.cat_features, nn_n=nn_n,nfold=nfold)


    def __word2vec_encoding(self,size=10,method='single'):
        sentence = []
        tmp_df = self.train[self.cat_features].append(self.test[self.cat_features], )
        if method!='single':
            for line in list(tmp_df.values):
                sentence.append([str(float(l)) for idx, l in enumerate(line)])
        else:
            sentence = list(tmp_df[self.cat_features[0]])
        print('sample setence')
        print(sentence[0])
        print('start word2vec ...')
        model = Word2Vec(sentence, vector_size=size, window=2, min_count=1,
                         workers=1, epochs=10, seed=1)

        w2v_feats = []
        for fea in self.cat_features:
            if method != 'single':
                values = tmp_df[fea].unique()
                w2v = []
                for i in values:
                    a = [i]
                    a.extend(model.wv[i])
                    w2v.append(a)
                out_df = pd.DataFrame(w2v)

                out_df.columns = [fea] + [fea + '_W2V_' + str(i) for i in range(size)]
                out_df.index = out_df[fea]

                for n in range(size):
                    f_name = f"{fea}_W2V_{n}"
                    self.train[f_name] = self.train[fea].map(out_df[f_name])
                    self.test[f_name] = self.test[fea].map(out_df[f_name])
                    w2v_feats.append(f_name)
            
            else:
                values = tmp_df[fea].tolist()
                w2v = []
                for i in values:
                    # print(i)
                    a = np.array([model.wv[str(x)] for x in i])
                    a = np.mean(a,axis=0)
                    # print(a.shape)
                    w2v.append(a)
                
                out_df = pd.DataFrame(w2v)
                out_df.columns = [fea + '_W2V_' + str(i) for i in range(size)]
                w2v_feats = [fea + '_W2V_' + str(i) for i in range(size)]

                for f in out_df:
                    cur = out_df[f].tolist()
                    self.train[f] = cur[:self.train.shape[0]]
                    self.test[f] = cur[self.train.shape[0]:]
                    
        return w2v_feats


    def encode_w2v(self,size=10,method='single'):
        return self.__word2vec_encoding(size,method)

    def __kfold_tareget_encoding(self,train,test,cols,label,nfold=5,seed=0,na_method='mean'):
        """a easy way to target encoding"""
        # kf = KFold(n_splits=nfold, shuffle=False, random_state=seed)
        kf = KFold(n_splits=nfold, shuffle=False)
        mean_label = train[label].mean()
        tgt_feats = []

        for i, (trn_id, val_id) in enumerate(kf.split(train[cols], train[label])):
            print('=' * 10 + f'target encoding in fold {i+1}' + '=' * 10)
            for f in cols:
                # trn_f, trn_y = train.loc[trn_id, f], train.loc[trn_id, label]
                val_f, val_y = train.loc[val_id, f], train.loc[val_id, label]

                f_name = f'{f}_tgt_n_enc'
                trn_dict = train.loc[trn_id, [f, label]].groupby(f)[label].mean()

                val_tf = val_f.map(trn_dict)
                print(f'{f}_col_miss ratio {val_tf.isnull().sum() / val_tf.shape[0]}')
                if na_method == 'mean':
                    val_tf = val_tf.fillna(mean_label)
                elif na_method == 'zero':
                    val_tf = val_tf.fillna(0)
                elif na_method == None:
                    pass
                else:
                    raise NotImplementedError

                train.loc[val_id, f_name] = val_tf

        for f in cols:
            f_name = f'{f}_tgt_n_enc'
            train[f_name].fillna(mean_label, inplace=True)
            val_dict = train.groupby(f)[f_name].mean()
            test[f_name] = test[f].map(val_dict)
            print(f'{f}_col_miss ratio {test[f_name].isnull().sum()/test.shape[0]}')

            if na_method == 'mean':
                test[f_name] = test[f_name].fillna(mean_label)
            elif na_method == 'zero':
                test[f_name] = test[f_name].fillna(0)
            elif na_method == None:
                pass
            else:
                raise NotImplementedError
            # test[f_name] = test[f_name].fillna(mean_label)
            tgt_feats.append(f_name)
        return tgt_feats
    def encode_kfold_target(self,nfold=5,seed=0,na_method='mean'):
        return self.__kfold_tareget_encoding(self.train,self.test,self.cat_features,self.label
                                             ,nfold,seed,na_method)

    def __woe_encoding(self,cols):
        tt_cols = []
        def WOE(data, feat, label):
            bin_values = data[feat].unique()
            good_total_num = len(data[data[label] == 1])
            bad_total_num = len(data[data[label] == 0])

            woe_dic = {}
            for i, val in enumerate(bin_values):
                good_num = len(data[(data[feat] == val) & (data[label] == 1)])
                bad_num = len(data[(data[feat] == val) & (data[label] == 0)])

                woe_dic[val] = np.log((good_num / good_total_num) / ((bad_num / bad_total_num + 0.0001)))

            return woe_dic

        for col in cols:
            f_name = f'{col}_WOE'
            woe_dict = WOE(self.train,col,self.label)
            self.train[f_name] = self.train[col].map(woe_dict)
            self.test[f_name] = self.test[col].map(woe_dict)
            tt_cols.append(f_name)
        return tt_cols
    def encode_woe(self):
        return self.__woe_encoding(self.cat_features)

class cross_2cat_feats:
    """
    cat cross cat features
    =======
    cat1_feats is a list containing high cat feat
    cat2_feats is a list containing high or binary feat
    =======
    * combine two feats
    * nunique cats by one
    * entropy cats by one
    * prop cats by one
    * encoding binary cat feats by high cat

    =======
    TO DO LIST:
    * neighbors_target_mean_n : more detail at here :https://www.kaggle.com/c/home-credit-default-risk/discussion/64821


    """

    def __init__(self, train, test, cat1_feats, cat2_feats, label):
        self.train = train
        self.test = test
        self.cat1_feats = cat1_feats
        self.label = label
        self.cat2_feats = cat2_feats

    def __combine_feats(self, df1, df2, cols,is_categroy=False,combin_one=False):
        """
        """
        cb_feats = []
        cols = list(set(cols))
        if not combin_one:
            for col1, col2 in list(itertools.combinations(cols, 2)):
                f_name = f'{col1}_{col2}_comb'
                df1[f_name] = df1[col1].astype(str) + '_' + df1[col2].astype(str)
                df2[f_name] = df2[col1].astype(str) + '_' + df2[col2].astype(str)
                if is_categroy:
                    df1[f_name] = df1[f_name].astype('category')
                    df2[f_name] = df2[f_name].astype('category')

                cb_feats.append(f_name)
        else:
            f_name = ''.join(cols)
            df1[f_name] = ''
            df2[f_name] = ''
            for col in cols:
                df1[f_name] += df1[col].astype(str)
                df2[f_name] += df2[col].astype(str)

            if is_categroy:
                df1[f_name] = df1[f_name].astype('category')
                df2[f_name] = df2[f_name].astype('category')
            cb_feats.append(f_name)

        return cb_feats

    def combine_2cat_feats(self,is_categroy=False,combin_one=False):
        return self.__combine_feats(self.train, self.test, self.cat2_feats + self.cat1_feats,is_categroy,combin_one)

    def __nunique_feats(self, df1, df2, cat1_feats, cat2_feats):
        nunique_feats = []
        for col1 in cat1_feats:
            for col2 in cat2_feats:
                if col1 != col2:
                    tmp_df = pd.concat([df1[[col1, col2]], df2[[col1, col2]]], ignore_index=True)
                    tmp_df_col1 = tmp_df.groupby(col1,sort=False)[col2].nunique().to_dict()

                    f_name1 = f"{col1}_{col2}_unique"

                    df1[f_name1] = df1[col1].map(tmp_df_col1)
                    df2[f_name1] = df2[col1].map(tmp_df_col1)

                    nunique_feats.append(f_name1)

        return nunique_feats

    def nunique_2cat_feats(self):
        return self.__nunique_feats(self.train, self.test, self.cat1_feats, self.cat2_feats)

    def __entropy_feats(self, df1, df2, cat1_feats, cat2_feats):
        entropy_feats = []
        for col1 in cat1_feats:
            for col2 in cat2_feats:
                if col1 != col2:
                    f_name1 = f"{col1}_{col2}_entropy"

                    tmp_df = pd.concat([df1[[col1, col2]], df2[[col1, col2]]], ignore_index=True)
                    tmp_df_col1 = tmp_df.groupby(col1,sort=False)[col2].agg(
                        tmp_name=lambda x: entropy(x.value_counts() / x.shape[0])
                        ).rename(columns={'tmp_name': col1}).to_dict()

                    df1[f_name1] = df1[col1].map(tmp_df_col1[col1])
                    df2[f_name1] = df2[col1].map(tmp_df_col1[col1])

                    entropy_feats.append(f_name1)
        return entropy_feats

    def entropy_2cat_feats(self):
        return self.__entropy_feats(self.train, self.test, self.cat1_feats, self.cat2_feats)

    def __prop_feats(self, df1, df2, cat1_feats, cat2_feats):
        prop_feats = []
        for col1 in cat1_feats:
            for col2 in tqdm(cat2_feats):
                if col1 != col2:
                    tmp_df = pd.concat([df1[[col1, col2]], df2[[col1, col2]]], ignore_index=True)

                    f_name1 = f'{col1}_{col2}_prop'
                    tmp_name = f"{col1}_{col2}_tmp"

                    tmp_df[tmp_name] = tmp_df[col1].astype(str) + '_' + tmp_df[col2].astype(str)

                    for df in [df1, df2]:
                        df[tmp_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
                        vc1 = tmp_df[tmp_name].value_counts()
                        vc2 = tmp_df[col1].value_counts()
                        df[f_name1] = df[tmp_name].map(vc1) / df[col1].map(vc2)

                        df.drop([tmp_name], axis=1, inplace=True)


                    prop_feats.append(f_name1)

        return prop_feats

    def prop_2cat_feats(self):
        return self.__prop_feats(self.train, self.test, self.cat1_feats, self.cat2_feats)

    def __encode_binary_cat_feats(self, df1, df2, cat1_feats, cat2_feats,
                                  aggregations, fillna=True, ):
        agg_feats = []
        for main_column in cat1_feats:
            for col in cat2_feats:
                for agg_type in aggregations:
                    new_col_name = main_column + '_' + col + '_' + agg_type
                    temp_df = pd.concat([df1[[col, main_column]], df2[[col, main_column]]])
                    # if usena: temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan
                    temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                        columns={agg_type: new_col_name})

                    temp_df.index = list(temp_df[col])
                    temp_df = temp_df[new_col_name].to_dict()

                    df1[new_col_name] = df1[col].map(temp_df).astype('float32')
                    df2[new_col_name] = df2[col].map(temp_df).astype('float32')
                    agg_feats.append(new_col_name)

                    if fillna:
                        df1[new_col_name].fillna(-1, inplace=True)
                        df2[new_col_name].fillna(-1, inplace=True)
                    print(new_col_name, ', ', end='\n')
        return agg_feats

    def agg_binary_cat_feats(self, aggregations, fillna=True):
        return self.__encode_binary_cat_feats(self.train, self.test, self.cat1_feats, self.cat2_feats,
                                              aggregations, fillna=fillna)


class agg_num_feats:
    """
    to deal with num and cat
    """

    def __init__(self, train, test, cat_feats, num_feats):
        self.train = train
        self.test = test
        self.cat_feats = cat_feats
        self.num_feats = num_feats

    def __agg_feats(self, df1, df2, cat_feats, num_feats, agg_type, ):
        agg_feats = []
        for f1 in cat_feats:
            tmp_df = pd.concat([df1[[f1] + num_feats], df2[[f1] + num_feats]], ignore_index=True)
            gp = tmp_df.groupby(f1)
            for f2 in num_feats:
                map_df = gp.agg({f2: agg_type})
                map_df.columns = [f"{f1}_{x}_agg_{y}" for x, y in map_df.columns]

                for col in map_df.columns:
                    df1[col] = df1[f1].map(map_df[col])
                    df2[col] = df2[f1].map(map_df[col])
                agg_feats += list(map_df.columns)
        return agg_feats

    def agg_feats(self, agg_type):
        return self.__agg_feats(self.train, self.test, self.cat_feats, self.num_feats, agg_type)

    def __agg_norm_feats(self, df1, df2, cat_feats, num_feats):
        norm_feats = []
        for f1 in cat_feats:
            tmp_df = pd.concat([df1[[f1] + num_feats], df2[[f1] + num_feats]], axis=0, ignore_index=True)
            gp = tmp_df.groupby(f1)
            for f2 in num_feats:
                tmp_df_mean_dict = gp[f2].mean()
                tmp_df_std_dict = gp[f2].std()

                f_name = f'{f2}_norm_by_{f1}'
                df1[f_name] = (df1[f2] - df1[f1].map(tmp_df_mean_dict)) / (
                    df1[f1].map(tmp_df_std_dict))

                df2[f_name] = (df2[f2] - df2[f1].map(tmp_df_mean_dict)) / (
                    df2[f1].map(tmp_df_std_dict))

                norm_feats.append(f_name)
        return norm_feats

    def agg_norm_feats(self):
        return self.__agg_norm_feats(self.train, self.test, self.cat_feats, self.num_feats)

    def __agg_one_zero_feats(self, df1, df2, cat_feats, num_feats):
        norm_feats = []
        for f1 in cat_feats:
            tmp_df = pd.concat([df1[[f1] + num_feats], df2[[f1] + num_feats]], axis=0, ignore_index=True)
            gp = tmp_df.groupby(f1)
            for f2 in num_feats:
                tmp_df_max_dict = gp[f2].max()
                tmp_df_min_dict = gp[f2].min()

                f_name = f'{f2}_one_zero_by_{f1}'
                df1[f_name] = (df1[f2] - df1[f1].map(tmp_df_min_dict)) / (
                        df1[f1].map(tmp_df_max_dict) - df1[f1].map(tmp_df_min_dict))

                df2[f_name] = (df2[f2] - df2[f1].map(tmp_df_min_dict)) / (
                        df2[f1].map(tmp_df_max_dict) - df2[f1].map(tmp_df_min_dict))

                norm_feats.append(f_name)
        return norm_feats

    def agg_one_zero_feats(self):
        return self.__agg_one_zero_feats(self.train, self.test, self.cat_feats, self.num_feats)

    def __agg_dist_feats(self, df1, df2, cat_feats, num_feats, dist_method='norm'):
        """dist method can be chosen from norm median mode max min cumprob"""
        dist_feats = []
        for f1 in cat_feats:
            tmp_df = pd.concat([df1[[f1] + num_feats], df2[[f1] + num_feats]], axis=0, ignore_index=True)
            gp = tmp_df.groupby(f1)
            for f2 in num_feats:
                if dist_method == 'norm':
                    tmp_df_mean_dict = gp[f2].mean()
                    tmp_df_std_dict = gp[f2].std()

                    f_name = f'{f2}_norm_by_{f1}'
                    df1[f_name] = (df1[f2] - df1[f1].map(tmp_df_mean_dict)) / (
                        df1[f1].map(tmp_df_std_dict))

                    df2[f_name] = (df2[f2] - df2[f1].map(tmp_df_mean_dict)) / (
                        df2[f1].map(tmp_df_std_dict))

                    dist_feats.append(f_name)
                elif dist_method == 'max_min':
                    tmp_df_max_dict = gp[f2].max()
                    tmp_df_min_dict = gp[f2].min()

                    f_name = f'{f2}_one_zero_by_{f1}'
                    df1[f_name] = (df1[f2] - df1[f1].map(tmp_df_min_dict)) / (
                            df1[f1].map(tmp_df_max_dict) - df1[f1].map(tmp_df_min_dict))

                    df2[f_name] = (df2[f2] - df2[f1].map(tmp_df_min_dict)) / (
                            df2[f1].map(tmp_df_max_dict) - df2[f1].map(tmp_df_min_dict))
                    dist_feats.append(f_name)
                elif dist_method == 'mode':
                    tmp_df_mode_dict = gp[f2].mode()
                    # tmp_df_min_dict = gp[f2].min()

                    f_name = f'{f2}_mode_dist_by_{f1}'
                    df1[f_name] = (df1[f2] - df1[f1].map(tmp_df_mode_dict))

                    df2[f_name] = (df2[f2] - df2[f1].map(tmp_df_mode_dict))

                    dist_feats.append(f_name)
                elif dist_method == 'mean':
                    tmp_df_mean_dict = gp[f2].mean()
                    # tmp_df_min_dict = gp[f2].min()

                    f_name = f'{f2}_mean_dist_by_{f1}'
                    df1[f_name] = (df1[f2] - df1[f1].map(tmp_df_mean_dict))

                    df2[f_name] = (df2[f2] - df2[f1].map(tmp_df_mean_dict))

                    dist_feats.append(f_name)
                elif dist_method == 'median':
                    tmp_df_dict = gp[f2].median()

                    f_name = f'{f2}_median_dist_by_{f1}'

                    df1[f_name] = (df1[f2] - df1[f1].map(tmp_df_dict))
                    df2[f_name] = (df2[f2] - df2[f1].map(tmp_df_dict))

                    dist_feats.append(f_name)

        return dist_feats


class trans_num_feats:
    """dist of num"""

    def __init__(self, train, test, num_feats, label='label'):
        self.train = train
        self.test = test
        self.num_feats = num_feats
        self.label = label


    def __dist_num(self, sigma_fac=0.001, sigma_base=4, eps=1e-08):
        import scipy.ndimage
        dist_feats = []
        for var in self.num_feats:
            tmp_df = pd.concat([self.train[[var]], self.test[[var]]], axis=0, ignore_index=True)

            X_all_var_int = (tmp_df[var].values * 100).round().astype(int)
            sigmas = []
            num_min = X_all_var_int.min()
            X_all_var_int = X_all_var_int - num_min
            hi = X_all_var_int.max() + 1
            ## frequecny encoding
            counts_all = np.bincount(X_all_var_int, minlength=hi).astype(float)

            sigma_scaled = counts_all.shape[0] * sigma_fac
            sigma = np.power(sigma_base * sigma_base * sigma_scaled, 1 / 3)
            sigmas.append(sigma)

            counts_all_smooth = scipy.ndimage.filters.gaussian_filter1d(counts_all, sigma)
            deviation = counts_all / (counts_all_smooth + eps)
            indices = X_all_var_int

            f_name1 = f'{var}_count'
            f_name2 = f'{var}_density'
            f_name3 = f'{var}_deviation'

            tmp_df[f_name1] = counts_all[indices]
            tmp_df[f_name2] = counts_all_smooth[indices]
            tmp_df[f_name3] = deviation[indices]
            tmp_df.drop_duplicates(subset=[var], inplace=True, ignore_index=True)
            tmp_df.index = tmp_df[var]

            self.train[f_name1] = self.train[var].map(tmp_df[f_name1])
            self.test[f_name1] = self.test[var].map(tmp_df[f_name1])

            self.train[f_name2] = self.train[var].map(tmp_df[f_name2])
            self.test[f_name2] = self.test[var].map(tmp_df[f_name2])

            self.train[f_name3] = self.train[var].map(tmp_df[f_name3])
            self.test[f_name3] = self.test[var].map(tmp_df[f_name1])

            dist_feats.append(f_name1)
            dist_feats.append(f_name2)
            dist_feats.append(f_name3)
        return dist_feats

    def dist_num_feats(self, sigma_fac=0.001, sigma_base=4, eps=1e-08):
        return self.__dist_num(sigma_fac, sigma_base, eps)

    def __log_num(self):
        # log_feats = [x+'_log_trans' for x in self.num_feats]
        log_feats = []
        for f in self.num_feats:
            f_name = f"{f}_log_trans"
            f_min = self.train[f].min()
            self.train[f] -= f_min
            self.train[f_name] = np.log(self.train[f] + 1)
            log_feats.append(f_name)
        return log_feats

    def log_num_feats(self):
        return self.__log_num()

    def _dtree_boundary(self,x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
        '''
            利用决策树获得最优分箱的边界值列表
        '''
        boundary = []  # 待return的分箱边界值列表

        x = x.fillna(nan).values  # 填充缺失值
        y = y.values

        clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                     max_leaf_nodes=6,  # 最大叶子节点数
                                     min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

        clf.fit(x.reshape(-1, 1), y)  # 训练决策树

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
                boundary.append(threshold[i])

        boundary.sort()

        min_x = np.min(x)
        max_x = np.max(x) + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本

        boundary = [min_x] + boundary + [max_x]

        return boundary

    def _best_ks_box(self,data, var_name, target_col, box_num):
        data = data[[var_name, target_col]]
        """
        KS值函数
        """

        def ks_bin(data_, limit):
            g = data_.iloc[:, 1].value_counts()[0]
            b = data_.iloc[:, 1].value_counts()[1]
            data_cro = pd.crosstab(data_.iloc[:, 0], data_.iloc[:, 1])
            data_cro[0] = data_cro[0] / g
            data_cro[1] = data_cro[1] / b
            data_cro_cum = data_cro.cumsum()
            ks_list = abs(data_cro_cum[1] - data_cro_cum[0])
            ks_list_index = ks_list.nlargest(len(ks_list)).index.tolist()
            bst_i = 0
            for i in ks_list_index:
                bst_i = i
                data_1 = data_[data_.iloc[:, 0] <= i]
                data_2 = data_[data_.iloc[:, 0] > i]
                if len(data_1) >= limit and len(data_2) >= limit:
                    break
            return bst_i

        """
        区间选取函数
        """

        def ks_zone(data_, list_):
            list_zone = list()
            list_.sort()
            n = 0
            for val in list_:
                m = sum(data_.iloc[:, 0] <= val) - n
                n = sum(data_.iloc[:, 0] <= val)
                #             print(val,' , m:',m,' n:',n)
                list_zone.append(m)
            # list_zone[i]存放的是list_[i]-list[i-1]之间的数据量的大小
            list_zone.append(50000 - sum(list_zone))
            #         print('sum ',sum(list_zone[:-1]))
            #         print('list zone ',list_zone)
            # 选取最大数据量的区间
            max_index = list_zone.index(max(list_zone))
            if max_index == 0:
                rst = [data_.iloc[:, 0].unique().min(), list_[0]]
            elif max_index == len(list_):
                rst = [list_[-1], data_.iloc[:, 0].unique().max()]
            else:
                rst = [list_[max_index - 1], list_[max_index]]
            return rst

        data_ = data.copy()
        limit_ = data.shape[0] / 20  # 总体的5%
        """"
        循环体
        """
        zone = list()
        for i in range(box_num - 1):
            # 找出ks值最大的点作为切点，进行分箱
            ks_ = ks_bin(data_, limit_)
            zone.append(ks_)
            new_zone = ks_zone(data, zone)
            data_ = data[(data.iloc[:, 0] > new_zone[0]) & (data.iloc[:, 0] <= new_zone[1])]

        zone.append(data.iloc[:, 0].unique().max()+1)
        zone.append(data.iloc[:, 0].unique().min()-1)
        zone.sort()
        return zone
    def _get_chimerge_cutoff(self,ser, tag, max_groups=None, threshold=None):
        # 计算2*2列联表的卡方值
        def get_chi2_value(arr):
            rowsum = arr.sum(axis=1)  # 对行求和
            colsum = arr.sum(axis=0)  # 对列求和
            n = arr.sum()
            emat = np.array([i * j / n for i in rowsum for j in colsum])
            arr_flat = arr.reshape(-1)
            arr_flat = arr_flat[emat != 0]  # 剔除了期望为0的值,不参与求和计算，不然没法做除法！
            emat = emat[emat != 0]  # 剔除了期望为0的值,不参与求和计算，不然没法做除法！
            E = (arr_flat - emat) ** 2 / emat
            return E.sum()

        # 自由度以及分位点对应的卡方临界值
        def get_chi2_threshold(percents, nfree):
            return chi2.isf(percents, df=nfree)
        freq_tab = pd.crosstab(ser, tag)
        cutoffs = freq_tab.index.values  # 保存每个分箱的下标
        freq = freq_tab.values  # [M,N_class]大小的矩阵，M是初始箱体的个数，N_class是目标变量类别的个数
        while True:
            min_value = None  # 存放所有对相邻区间中卡方值最小的区间的卡方值
            min_idx = None  # 存放最小卡方值的一对区间中第一个区间的下标
            for i in range(len(freq) - 1):
                chi_value = get_chi2_value(freq[i:(i + 2)])  # 计算第i个区间和第i+1个区间的卡方值
                if min_value == None or min_value > chi_value:
                    min_value = chi_value
                    min_idx = i
            if (max_groups is not None and max_groups < len(freq)) or (
                    threshold is not None and min_value < get_chi2_threshold(threshold, len(cutoffs) - 1)):
                tmp = freq[min_idx] + freq[min_idx + 1]  # 合并卡方值最小的那一对区间
                freq[min_idx] = tmp
                freq = np.delete(freq, min_idx + 1, 0)  # 删除被合并的区间
                cutoffs = np.delete(cutoffs, min_idx + 1, 0)
            else:
                break
        ## 最小值最大值扩大
        cutoffs[0] = cutoffs[0] - 1
        # cutoffs[-1] = cutoffs[-1] + 1
        cutoffs = np.append(cutoffs,[np.max(ser)+1])
        return cutoffs
    def _cut_with_boundary(self,f_name,col,boundary):
        ## 去重
        boundary = sorted(list(set(boundary)))
        self.train[f_name] = pd.cut(self.train[col],bins=boundary,labels=range(len(boundary)-1),duplicates='drop',include_lowest=True).astype('int')
        self.test[f_name] = pd.cut(self.test[col],bins=boundary,labels=range(len(boundary)-1),duplicates='drop',include_lowest=True).astype('int')

    def __cut_bins(self,cols,bins,strategy=None):

        tt_feat = []
        for f in cols:
            for b in bins:
                f_name = f'{f}_{b}_bin_{strategy}'
                if strategy == 'uniform':

                    kbin = KBinsDiscretizer(n_bins=b, encode='ordinal', strategy='uniform')
                    self.train[f_name] = kbin.fit_transform(np.array(self.train[f]).reshape(-1, 1))
                    self.test[f_name] = kbin.transform(np.array(self.test[f]).reshape(-1, 1))
                elif strategy == 'quantile':

                    kbin = KBinsDiscretizer(n_bins=b, encode='ordinal', strategy='quantile')
                    self.train[f_name] = kbin.fit_transform(np.array(self.train[f]).reshape(-1, 1))
                    self.test[f_name] = kbin.transform(np.array(self.test[f]).reshape(-1, 1))
                elif strategy == 'kmeans':

                    kbin = KBinsDiscretizer(n_bins=b, encode='ordinal', strategy='kmeans')
                    self.train[f_name] = kbin.fit_transform(np.array(self.train[f]).reshape(-1, 1))
                    self.test[f_name] = kbin.transform(np.array(self.test[f]).reshape(-1, 1))
                elif strategy == 'dtree':

                    bin_bounds = self._dtree_boundary(self.train[f],self.train[self.label])
                    self._cut_with_boundary(f_name,f,bin_bounds)
                elif strategy == 'bestKs':
                    bin_bounds = self._best_ks_box(self.train,f,self.label,box_num=b)
                    self._cut_with_boundary(f_name, f, bin_bounds)

                elif strategy == 'chi2_one':
                    bin_bounds = self._get_chimerge_cutoff(self.train[f],self.train[self.label],max_groups=b)
                    self._cut_with_boundary(f_name,f,bin_bounds)

                tt_feat.append(f_name)

        return tt_feat
    def cut_groups(self,bins,strategy):
        print(f'cuting features by {strategy}....')
        return self.__cut_bins(self.num_feats,bins,strategy)


class fillna_tools:

    def __init__(self,train,test,cols,label):
        self.train = train
        self.test = test
        self.test = test
        self.cols = cols
        self.label = label

    def fillna_mean(self):
        tt_cols = []
        for col in self.cols:
            f_name = f"{col}_na_mean"
            tmp_df = pd.concat([self.train[col],self.test[col]],ignore_index=True)
            na_mean = tmp_df.mean()
            self.train[f_name] = self.train[col].fillna(na_mean)
            self.test[f_name] = self.test[col].fillna(na_mean)
            tt_cols.append(f_name)
        return tt_cols
    def fillna_median(self,inplace=False):
        tt_cols = []
        for col in self.cols:
            if inplace:
                f_name = col
            else:
                f_name = f"{col}_na_median"
            tmp_df = pd.concat([self.train[col],self.test[col]],ignore_index=True)
            na_median = tmp_df.median()
            self.train[f_name] = self.train[col].fillna(na_median)
            self.test[f_name] = self.test[col].fillna(na_median)
            tt_cols.append(f_name)
        return tt_cols

    def fillna_mode(self,inplace=False):
        tt_cols = []
        for col in self.cols:
            if inplace:
                f_name = col
            else:
                f_name = f"{col}_na_mode"
            tmp_df = pd.concat([self.train[col],self.test[col]],ignore_index=True)
            na_mode = tmp_df.mode().values[0]
            self.train[f_name] = self.train[col].fillna(na_mode)
            self.test[f_name] = self.test[col].fillna(na_mode)
            tt_cols.append(f_name)
        return tt_cols
    def fillna_neart(self,is_binary=True):
        tt_cols = []
        ## 在所有计数大于自身的类别中，计算和自己最接近的类别进行填充。
        for col in self.cols:
            f_name = f"{col}_na_neart"
            tmp_df  = pd.concat([self.train[col],self.test[col]],ignore_index=True)

            ## 计算次数并找到比自己大的类别。
            cnt = tmp_df.fillna('noseen').value_counts(dropna=False).reset_index()
            # print(cnt)
            cnt.columns = [col,'values']
            nan_index = cnt[cnt[col]=='noseen'].index[0]
            cnt = cnt[:nan_index+1]

            ## 计算target rate
            target_rate = self.train[[col,self.label]].fillna('noseen').groupby(col)[self.label].mean()
            cnt = cnt.merge(target_rate,on=col,how='left')
            nan_mean = cnt.loc[cnt[col]=='noseen',self.label]

            ## 找到和自己最接近的
            cnt[self.label] = cnt[self.label] - nan_mean
            cnt.sort_values(self.label,ascending=True,inplace=True)
            na_values = cnt.loc[1,col]

            self.train[f_name] = self.train[col].fillna(na_values)
            self.test[f_name] = self.test[col].fillna(na_values)
            tt_cols.append(f_name)

        return tt_cols

    def fillna_min(self,inplace=False):
        tt_cols = []
        for col in self.cols:
            if not inplace:
                f_name = f"{col}_na_min"
            else:
                f_name = col
            na_min = self.train[col].append(self.test[col]).min()

            self.train[f_name] = self.train[col].fillna(na_min-1)
            self.test[f_name] = self.test[col].fillna(na_min-1)
            tt_cols.append(f_name)

        return tt_cols






def cut_tail(train_df, test_df, cols, cnt_num=10, interaction=False):
    for f in tqdm(cols):
        valid_card = train_df[f].value_counts()
        valid_card = valid_card[valid_card >= cnt_num]
        valid_card = list(valid_card.index)

        train_df[f] = np.where(train_df[f].isin(valid_card), train_df[f], np.nan)
        test_df[f] = np.where(test_df[f].isin(valid_card), test_df[f], np.nan)

        ##interaction
        if interaction:
            train_df[f] = np.where(train_df[f].isin(test_df[f]), train_df[f], np.nan)
            test_df[f] = np.where(test_df[f].isin(train_df[f]), test_df[f], np.nan)
    if interaction:
        print('finish the cut group task============')
    print('finish the cut group task============')


def freq_encode(df1, df2, cols):
    add_features = []
    for col in cols:
        df = pd.concat([df1[col], df2[col]])
        vc = df.value_counts(dropna=False, normalize=True).to_dict()
        vc[-1] = -1
        nm = col + '_FrqEnc'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm, ', ', end='\n')
        add_features.append(nm)
    return add_features


def deal_outier(train, test, cols):
    for col in cols:
        train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
        test[col] = np.where(test[col].isin(train[col]), test[col], np.nan)


def findRank(x, num):
    x_cnt = pd.value_counts(x).index.tolist()
    x_len = len(x_cnt)
    if num < x_len:
        return x_cnt[num]
    else:
        return np.nan


def cut_bins(df1, df2, cols, bins, abs=True):
    """abs 为等距分组，否则为等频分组"""
    tt_feat = []
    for f in cols:
        for b in bins:
            f_name = f'{f}_{b}_bin_{int(abs)}'
            if abs:
                kbin = KBinsDiscretizer(n_bins=b, encode='ordinal', strategy='uniform')
                df1[f_name] = kbin.fit_transform(np.array(df1[f]).reshape(-1, 1))
                df2[f_name] = kbin.transform(np.array(df2[f]).reshape(-1, 1))
            else:
                kbin = KBinsDiscretizer(n_bins=b, encode='ordinal', strategy='quantile')
                df1[f_name] = kbin.fit_transform(np.array(df1[f]).reshape(-1, 1))
                df2[f_name] = kbin.transform(np.array(df2[f]).reshape(-1, 1))

            tt_feat.append(f_name)

    return tt_feat




def encode_AG(main_columns, uids, aggregations, train_df, test_df,
              fillna=True, usena=False, dist=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    agg_feats = []
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column + '_' + col + '_' + agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
                if usena: temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name] = test_df[col].map(temp_df).astype('float32')
                if dist:
                    f_name = main_column + '_' + col + '_' + agg_type + '_dist'
                    train_df[f_name] = train_df[main_column] - train_df[new_col_name]
                    test_df[f_name] = test_df[main_column] - test_df[new_col_name]
                    agg_feats.append(f_name)
                agg_feats.append(new_col_name)
                if fillna:
                    train_df[new_col_name].fillna(-1, inplace=True)
                    test_df[new_col_name].fillna(-1, inplace=True)
                print("'" + new_col_name + "'", ', ', end='\n')

    return agg_feats


class MeanEncoder:
    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode

        :param n_splits: the number of splits used in mean encoding

        :param target_type: str, 'regression' or 'classification'

        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """

        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}

        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None

        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()

        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()

        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg(means='mean', beta='size')
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['means']
        col_avg_y.drop(['beta', 'means'], axis=1, inplace=True)

        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

        return nf_train, nf_test, prior, col_avg_y

    def fit_transform(self, X, y):

        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)

        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None,
                        self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new
    # pd.cut(retbins=True)
    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()

        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits

        return X_new


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """

    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def kfold_target_encode(train, test, cols, seed, label, nfold=5, method='smoothing'):
    """methond contain two way:somthing OR navie"""
    kf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    mean_label = train[label].mean()
    tgt_feats = []
    for i, (trn_id, val_id) in enumerate(kf.split(train[cols], train[label])):
        print('=' * 10 + f'target encoding in fold {i}' + '=' * 10)
        for f in cols:

            trn_f, trn_y = train.loc[trn_id, f], train.loc[trn_id, label]
            val_f, val_y = train.loc[val_id, f], train.loc[val_id, label]
            if method == 'smoothing':
                f_name = f'{f}_tgt_s_enc'
                trn_tf, val_tf = target_encode(
                    trn_series=trn_f,
                    tst_series=val_f,
                    target=trn_y,
                    min_samples_leaf=100,
                    smoothing=20,
                    noise_level=0.01,
                )

            else:
                f_name = f'{f}_tgt_n_enc'
                trn_dict = train.loc[trn_id, [f, label]].groupby(f)[label].mean()

                val_tf = val_f.map(trn_dict)
                val_tf = val_tf.fillna(mean_label)
            train.loc[val_id, f_name] = val_tf

    for f in cols:
        f_name = f'{f}_tgt_s_enc' if method == 'smoothing' else f'{f}_tgt_n_enc'

        train[f_name].fillna(mean_label, inplace=True)
        val_dict = train.groupby(f)[f_name].mean()
        test[f_name] = test[f].map(val_dict)
        test[f_name] = test[f_name].fillna(mean_label)
        tgt_feats.append(f_name)

    return tgt_feats
