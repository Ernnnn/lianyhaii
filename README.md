# lianyhaii

# usage
## 第一步：安装与测试
`pip install lianyhaii`  
`import lianyhaii`  
`print(lianyhaii.__version__)`  
## 第二步：准备数据、特征、参数
```python
import pandas as pd 
import numpy  as np 
import lianyhaii
import warnings 
import sys 
warnings.filterwarnings('ignore')

## 定义数据集、label、训练参数、特征名
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
lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'early_stopping_rounds': 50,
    'verbose':-1,
}
base_features = [f'x{i}' for i in range(10)]
```
以上是送进模型的主要准备工作，接下来会轻松不少

## 第三步：快速模型测试
```python
mt = make_test(train,test,base_features=base_features,new_features=[],
            m_score=[[0.0,]],label=label,metrices=['auc'],log_tool=None)
mt.init_CV(seed=412,CV_type='skFold',n_split=5)
oof,pred = mt.lgb_test(lgb_params=lgb_params)
## 得到oof和pred方便后续调整或者提交
```

# 获奖经历
> 2021 山东赛 公积金贷款逾期预测 A榜 rank2/xxx B榜 rank6/xxx  
> 2021 梧桐杯 5G潜客识别 B榜 rank7/xxx  
> 2022 招商银行Fintech  rank23/xxx   -> [开源](example/competition_solution/zhaohang_B.md)  
> 2022 搜狐校园 情感分析 × 推荐排序 算法大赛 rank5/xxx  -> [开源](example/sohu2022/sohu2022.md)
