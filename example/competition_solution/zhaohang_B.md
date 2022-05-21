## 招商银行2022fintech比赛开源
A榜：9557 rank 1x(?忘记多少了)<br>
B榜：8665 rank 24<br>


### B榜
思路和群里讨论的比较接近。
1. 首先利用adv剔除了高于0.55的特征，线下0.733 线上0.733，比较一致。（同时这里也对数据进行了对数化处理，减少模型的极端值，主要是方便画图）
2. 通过分析发现last 12的相关特征虽然分布不一致，但是对模型来说非常重要。加上之后线下0.864，线上0.857
3. 通过上两步发现，掉分的基本都是最近的特征，因此把REG_DT、RCT相关的特征都删了，也能进一步上到860
4. 最后观察train和val的分数差异发现，过拟合非常严重，因此把lgb中的basetree换成rf，减轻过拟合，860上到866(赌博成功)

> 这里用到了lianyhaii中的eda_tool和一系列方便的工具，更轻松的进行比赛。



### A榜
思路和B榜相差甚远，有空再更。