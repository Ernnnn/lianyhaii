# coding:utf-8
# import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


warnings.filterwarnings("ignore")
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 1000)

# def plot_null(df):
#     """plot the null of data in three ways"""
#     print('*'*10+'_null_'+'*'*10)
#     """缺失分布图"""
#     msno.matrix(df,labels=True)
#     plt.show()
#     """缺失比例图"""
#     msno.bar(df)
#     plt.show()
#     """缺失相关图"""
#     msno.heatmap(df,)
#     plt.show()

def clean_null(df,threhold=0.97):
    """delete the columns which ratio above the threhold"""
    df_ = df.copy()
    null_ratio = (df_.isnull().sum() / df_.shape[0]).sort_values(ascending=False)
    print(null_ratio.head())
    print('*******deletting null columns*************')
    df_.drop(null_ratio[(null_ratio > threhold) == True].index.tolist(), axis=1, inplace=True)
    print((df_.isnull().sum() / df_.shape[0]).sort_values(ascending=False).head())
    return df_

def plot_numDist_by(train,test,feats,return_p = False,label='label'):
    df1 = train.copy()
    df2 = test.copy()
    df1['test_label'] = 0
    df2['test_label'] = 1
    df = pd.concat([df1,df2],axis=0,ignore_index=True)
    p_list = []
    for feat in feats:
        fig,axes = plt.subplots(1,2,sharex=True)

        sns.distplot(df.loc[df['test_label'] == 1, feat], bins=100, color='r',ax=axes[0])
        sns.distplot(df.loc[df['test_label'] == 0, feat], bins=100, color='b',ax=axes[0])
        axes[0].legend(['test', 'train'])
        p = np.round(ks_2samp(train[feat], test[feat])[1],4)
        p_list.append(p)
        axes[0].set_title(f'The KS test p value is :{p}')

        sns.distplot(df.loc[df[label] == 1, feat], bins=100, color='r',ax=axes[1])
        sns.distplot(df.loc[df[label] == 0, feat], bins=100, color='b',ax=axes[1])
        axes[1].legend(['label: 1', 'label: 0'])
        plt.show()
    return p_list if return_p else None

def plot_cat_by(train,test,cols,label='label',type='binary'):
    df1 = train.copy()
    df2 = test.copy()
    df1['test_label'] = 0
    df2['test_label'] = 1
    df = pd.concat([df1,df2],axis=0,ignore_index=True)
    for col in cols:
        if type == 'binary':
            for by in ['test_label',label]:
                fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,figsize=(15,7))

                (df.groupby(col)[by].sum() / df.groupby(col)[by].count()).plot(kind='bar',ax=axes[0],color='r')

                axes[1] = sns.countplot(x=col,hue=by,data=df)
                plt.xticks(rotation=90)
                plt.show()

                if by == 'test_label':
                    tt = set(df.loc[df[by]==1,col].unique())
                    tr = set(df.loc[df[by]==0,col].unique())
                    print(f'new ratio in test dataset:  { (len(tt)- len(tr))/len(tt)}')
                    print(f'new number in test dataset:  {len(tt - tr)}')

        if type == 'reg':

            fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(15, 7))
            by = 'test_label'
            (df.groupby(col)[by].sum() / df.groupby(col)[by].count()).plot(kind='bar', ax=axes[0], color='r')

            axes[1] = sns.countplot(x=col, hue=by, data=df)
            plt.xticks(rotation=90)
            plt.show()

            print(
                f'new ratio in test dataset:  {len(set(df.loc[df[by] == 1, col]) - set(df.loc[df[by] == 0, col])) / len(set(df.loc[df[by] == 1, col]))}')
            print(
                f'new number in test dataset:  {len(set(df.loc[df[by] == 1, col]) - set(df.loc[df[by] == 0, col]))}')

            sns.boxplot(x=col,y=label,data=df,)
            plt.show()

def quick_look_data(df):
    print(df.shape)
    print(df.info())
    print(df.describe())

