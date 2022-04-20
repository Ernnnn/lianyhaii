import time
# from multiprocessing import Pool
from pathos.multiprocessing import Pool
import pandas as pd
import numpy as np

from lianyhaii.tools import time_it


def gen_date_feat(df_):
    df = df_.copy()
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['holiday'] = df['weekday'].map(lambda x: 1 if x in [6,7] else 0)
    date_feat = ['hour','day','weekday','month','holiday']
    return df, date_feat

def add_lag_feat_byid(df_,col,d_shift=3):
    df = df_.copy()
    for diff in [1,3,12,24]:
        shift = d_shift + diff
        df[col+f"_shift_t{shift}"] = df.groupby(["MN_"])[col].transform(
            lambda x: x.shift(shift)
        )
    # for window in [3,12, 24]:
    #     df[col+f"_rolling_std_t{window}"] = df.groupby(["MN_"])[col].transform(
    #         lambda x: x.shift(d_shift).rolling(window).std()
    #     )

    for window in [3,6,12,24]:
        df[col+f"_rolling_mean_t{window}"] = df.groupby(["MN_"])[col].transform(
            lambda x: x.shift(d_shift).rolling(window).mean()
        )

    # for window in [3,6,12, 24]:
    #     df[col+f"_rolling_min_t{window}"] = df.groupby(["MN_"])[col].transform(
    #         lambda x: x.shift(d_shift).rolling(window).min()
    #     )
    # for window in [3,6,12,24]:
    #     df[col+f"_rolling_max_t{window}"] = df.groupby(["MN_"])[col].transform(
    #         lambda x: x.shift(d_shift).rolling(window).max()
    #     )

    # df[col+"_rolling_skew_t30"] = df.groupby(["MN_"])[col].transform(
    #     lambda x: x.shift(d_shift).rolling(30).skew()
    # )
    # df[col+"_rolling_kurt_t30"] = df.groupby(["MN_"])[col].transform(
    #     lambda x: x.shift(d_shift).rolling(30).kurt()
    # )

    ##���һ���ڵ�����վ���ͳ�����
    # for window in [1,3,6,24]:
    #     df[col+f"_rolling_mean_max_all_t{window}"]  = df.groupby['date'][col].mean().shift(d_shift).rolling(window).max()
    #     df[col+f"_rolling_max_max_all_t{window}"]  = df.groupby['date'][col].max().shift(d_shift).rolling(window).max()
    #
    # for window in [1,3,6,24]:
    #     df[col+f"_rolling_min_all_t{window}"]  = df[col].shift(d_shift).rolling(window).min()
    # for window in [1,3,6,24]:
    #     df[col+f"_rolling_mean_all_t{window}"]  = df[col].shift(d_shift).rolling(window).mean()
    #
    # for window in [1,3,6,24]:
    #     df[col+f"_rolling_std_all_t{window}"]  = df[col].shift(d_shift).rolling(window).std()
    # for diff in [1,3,6,24]:
    #     df[col+f"_shift_all_t{diff}"] = df.groupby('date')[col].mean().shift(d_shift+diff)

    ##����������

    for window in [1,3,12,24]:
        df[col+f"_rolling_ratio_t{window}"] = df.groupby('MN_')[col].transform(
            lambda x:x.pct_change(periods=window)
        )

    lag_feat = [x for x in df.columns if ('rolling' in x ) or ('shift' in x)]
    return df, lag_feat


def add_lag_feat(df_,col,d_shift=3):
    df = df_.copy()
    win_time = [3,5,7]
    more_time = [1] +win_time

    for diff in more_time:
        shift = d_shift + diff
        df[col+f"_shift_t{shift}"] = df[col].transform(
            lambda x: x.shift(shift)
        )
    # for window in win_time:
    #     df[col+f"_rolling_std_t{window}"] = df[col].transform(
    #         lambda x: x.shift(d_shift).rolling(window).std()
    #     )

    for window in win_time:
        df[col+f"_rolling_mean_t{window}"] = df[col].transform(
            lambda x: x.shift(d_shift).rolling(window).mean()
        )

    # for window in win_time:
    #     df[col+f"_rolling_min_t{window}"] = df[col].transform(
    #         lambda x: x.shift(d_shift).rolling(window).min()
    #     )
    # �����˷�λ��
    q_time = [5]
    # for window in q_time:
    #     df[col+f"_rolling_q70_t{window}"] = df[col].transform(
    #         lambda x: x.shift(d_shift).rolling(window).quantile(0.7)
    #     )
    # for window in q_time:
    #     df[col+f"_rolling_q30_t{window}"] = df[col].transform(
    #         lambda x: x.shift(d_shift).rolling(window).quantile(0.3)
    #     )


    ##����������
    for window in more_time:
        df[col+f"_rolling_ratio_t{window}"] = df[col].transform(
            lambda x:x.pct_change(periods=window)
        )

    lag_feat = [x for x in df.columns if ('rolling' in x ) or ('shift' in x)]

    return df, lag_feat


class ts_features():
    def __init__(self,train,test,ids,label,date_col,base_lag):
        self.train = train
        self.train['flat'] = 1
        self.test = test
        self.test['flat'] = 0
        self.ids = [ids] if type(ids) == str else ids
        self.label = label
        self.data = pd.concat([train,test],ignore_index=True,)
        self.date_col = date_col
        self.base_lag = base_lag
        # print(self.ids,[date_col])
        self.data.sort_values(
            by=self.ids+[date_col],
            ascending=True,
            inplace=True,
            ignore_index=True
        )

    @time_it
    def add_lag_feats(self,lags,isParallel=False):
        tt_feats = []
        ### ���л�����
        if isParallel:
            def lag_feats(df,l):
                return df.groupby(self.ids)[self.label].transform(lambda x: x.shift(l))
            res = []
            p = Pool()
            for lag in lags:
                lag += self.base_lag
                tmp = p.apply_async(lag_feats,args=(self.data[self.ids+[self.label]].copy(),lag),)
                res.append(tmp)
            p.close()
            p.join()
            for l,s in zip(lags,res):
                f_name = f'{self.label}_lag_{l}'
                # print(l,s)
                self.data[f_name] = s.get()
                tt_feats.append(f_name)

        if not isParallel:
            for l in lags:
                l += self.base_lag
                f_name = f'{self.label}_lag_{l}'
                self.data[f_name] = self.data.groupby(self.ids)[self.label].transform(lambda x:x.shift(l))
                # print(self.data)
                tt_feats.append(f_name)
        return tt_feats

    @time_it
    def add_rolling_feats(self,windows,method='mean'):
        tt_feats = []

        for window in windows:
            f_name = f'{self.label}_rolling_{method}_w{window}'
            self.data[f_name] = self.data[self.label].transform(
                lambda x: x.shift(self.base_lag).rolling(window).agg([method])
            )
            tt_feats.append(f_name)

        return tt_feats

    def return_dataset(self):
        self.train = self.data[self.data['flat']==1]
        self.train.reset_index(drop=True,inplace=True)
        self.test = self.data[self.data['flat']==0]
        self.test.reset_index(drop=True,inplace=True)
        return self.train,self.test









