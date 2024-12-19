import warnings

import label
from cv2.ml import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 评价指标
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
train = pd.read_csv('train.csv')
test = pd.read_csv('testA.csv')
train.head()
#  样本个数和特征维度
train.shape
# (800000, 47)
test.shape
# (200000, 46)
train.columns

train.info()

train.describe()
# train.describe().T
# 数值类型
numerical_feature = list(train.select_dtypes(exclude=['object']).columns)
numerical_feature
['id', 'loanAmnt', 'term', 'interestRate', 'installment', 'employmentTitle',
 'homeOwnership', 'annualIncome', 'verificationStatus', 'isDefault', 'purpose',
 'postCode', 'regionCode', 'dti', 'delinquency_2years', 'ficoRangeLow',
 'ficoRangeHigh', 'openAcc', 'pubRec', 'pubRecBankruptcies', 'revolBal',
 'revolUtil', 'totalAcc', 'initialListStatus', 'applicationType',
 'title', 'policyCode', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8',
 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']
len(numerical_feature)  ## 42
# 连续型变量
serial_feature = []
# 离散型变量
discrete_feature = []
# 单值变量
unique_feature = []

for fea in numerical_feature:
    temp = train[fea].nunique()# 返回的是唯一值的个数
    if temp == 1:
        unique_feature.append(fea)
     # 自定义变量的值的取值个数小于10就为离散型变量
    elif temp <= 10:
        discrete_feature.append(fea)
    else:
        serial_feature.append(fea)
serial_feature
'''
['id', 'loanAmnt', 'interestRate', 'installment', 'employmentTitle',
 'annualIncome', 'purpose', 'postCode', 'regionCode', 'dti',
 'delinquency_2years', 'ficoRangeLow', 'ficoRangeHigh', 'openAcc',
 'pubRec', 'pubRecBankruptcies', 'revolBal', 'revolUtil', 'totalAcc',
 'title', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8',
 'n9', 'n10', 'n13', 'n14']
'''
#每个数字特征得分布可视化
f = pd.melt(train, value_vars=serial_feature)
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
plt.figure(1 , figsize = (8 , 5))
sns.distplot(train.loanAmnt,bins=40)
plt.xlabel('loanAmnt')
sns.kdeplot(train.loanAmnt[label[label==1].index], label='1', shade=True)#违约
sns.kdeplot(train.loanAmnt[label[label==0].index], label='0', shade=True)#没有违约
plt.xlabel('loanAmnt')
plt.ylabel('Density')
plt.figure(1 , figsize = (8 , 5))
sns.distplot(train['annualIncome'])
plt.xlabel('annualIncome')
discrete_feature
'''
['term', 'homeOwnership', 'verificationStatus', 'isDefault',
 'initialListStatus', 'applicationType', 'n11', 'n12']
'''
for f in discrete_feature:
    print(f, '类型数：', train[f].nunique())
'''
term 类型数： 2
homeOwnership 类型数： 6
verificationStatus 类型数： 3
isDefault 类型数： 2
initialListStatus 类型数： 2
applicationType 类型数： 2
n11 类型数： 5
n12 类型数： 5
'''
df_ = train[discrete_feature]

sns.set_style("whitegrid") # 使用whitegrid主题
fig,axes=plt.subplots(nrows=4,ncols=2,figsize=(8,10))
for i, item in enumerate(df_):
    plt.subplot(4,2,(i+1))
    #ax=df[item].value_counts().plot(kind = 'bar')
    ax=sns.countplot(item,data = df_,palette="Pastel1")
    plt.xlabel(str(item),fontsize=14)
    plt.ylabel('Count',fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.title("Churn by "+ str(item))
    i=i+1
    plt.tight_layout()
plt.show()
unique_feature
'''
['policyCode']
'''
# 分类型特征
category_feature = list(filter(lambda x: x not in numerical_feature,list(train.columns)))
category_feature

['grade', 'subGrade', 'employmentLength', 'issueDate', 'earliesCreditLine']
train[category_feature]
'''
       grade subGrade employmentLength   issueDate earliesCreditLine
0          E       E2          2 years  2014-07-01          Aug-2001
1          D       D2          5 years  2012-08-01          May-2002
2          D       D3          8 years  2015-10-01          May-2006
3          A       A4        10+ years  2015-08-01          May-1999
4          C       C2              NaN  2016-03-01          Aug-1977
     ...      ...              ...         ...               ...
799995     C       C4          7 years  2016-07-01          Aug-2011
799996     A       A4        10+ years  2013-04-01          May-1989
799997     C       C3        10+ years  2015-10-01          Jul-2002
799998     A       A4        10+ years  2015-02-01          Jan-1994
799999     B       B3          5 years  2018-08-01          Feb-2002

[800000 rows x 5 columns]
'''
df_category = train[['grade', 'subGrade']]

sns.set_style("whitegrid") # 使用whitegrid主题
color = sns.color_palette()
fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(10,10))
for i, item in enumerate(df_category):
    plt.subplot(2,1,(i+1))
    #ax=df[item].value_counts().plot(kind = 'bar')
    ax=sns.countplot(item,data = df_category)
    plt.xlabel(str(item),fontsize=14)
    plt.ylabel('Count',fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.title("Churn by "+ str(item))
    i=i+1
    plt.tight_layout()
plt.show()
plt.figure(1 , figsize = (10 , 8))
sns.barplot(train["employmentLength"].value_counts(dropna=False),
            train["employmentLength"].value_counts(dropna=False).keys())
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('employmentLength',fontsize=14)
plt.show()
for i in train[['issueDate', 'earliesCreditLine']]:
    print(train[i].value_counts())
    print()

'''
2016-03-01    29066
2015-10-01    25525
2015-07-01    24496
2015-12-01    23245
2014-10-01    21461
              ...
2007-08-01       23
2007-07-01       21
2008-09-01       19
2007-09-01        7
2007-06-01        1
Name: issueDate, Length: 139, dtype: int64

Aug-2001    5567
Aug-2002    5403
Sep-2003    5403
Oct-2001    5258
Aug-2000    5246
            ...
Jan-1946       1
Nov-1953       1
Aug-1958       1
Jun-1958       1
Oct-1957       1
Name: earliesCreditLine, Length: 720, dtype: int64
'''
label=train.isDefault
label.value_counts()/len(label)

'''
0    0.800488
1    0.199513
Name: isDefault, dtype: float64
'''
sns.countplot(label)
train_loan_fr = train.loc[train['isDefault'] == 1]
train_loan_nofr = train.loc[train['isDefault'] == 0]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
# 目标变量为1时候grade的分布
train_loan_fr.groupby("grade").size().plot.bar(ax=ax1)
# 目标变量为0时候grade的分布
train_loan_nofr.groupby("grade")["grade"].count().plot.bar(ax=ax2)
# 目标变量为1时候employmentLength的分布
train_loan_fr.groupby("employmentLength").size().plot.bar(ax=ax3)
# 目标变量为0时候employmentLength的分布
train_loan_nofr.groupby("employmentLength")["employmentLength"].count().plot.bar(ax=ax4)
plt.xticks(rotation=90);
train_positve = train[train['isDefault'] == 1]
train_negative = train[train['isDefault'] != 1]
f, ax = plt.subplots(len(numerical_feature),2,figsize = (10,80))
for i,col in enumerate(numerical_feature):
    sns.distplot(train_positve[col],ax = ax[i,0],color = "blue")
    ax[i,0].set_title("positive")
    sns.distplot(train_negative[col],ax = ax[i,1],color = 'red')
    ax[i,1].set_title("negative")
plt.subplots_adjust(hspace = 1)
# 去掉标签
X_missing = train.drop(['isDefault'],axis=1)

# 查看缺失情况
missing = X_missing.isna().sum()
missing = pd.DataFrame(data={'特征': missing.index,'缺失值个数':missing.values})
#通过~取反，选取不包含数字0的行
missing = missing[~missing['缺失值个数'].isin([0])]
# 缺失比例
missing['缺失比例'] =  missing['缺失值个数']/X_missing.shape[0]
missing

'''
           特征        缺失值个数 缺失比例
7      employmentTitle      1  0.000001
8     employmentLength  46799  0.058499
14            postCode      1  0.000001
16                 dti    239  0.000299
22  pubRecBankruptcies    405  0.000506
24           revolUtil    531  0.000664
29               title      1  0.000001
31                  n0  40270  0.050338
32                  n1  40270  0.050338
33                  n2  40270  0.050338
34                  n3  40270  0.050338
35                  n4  33239  0.041549
36                  n5  40270  0.050338
37                  n6  40270  0.050338
38                  n7  40270  0.050338
39                  n8  40271  0.050339
40                  n9  40270  0.050338
41                 n10  33239  0.041549
42                 n11  69752  0.087190
43                 n12  40270  0.050338
44                 n13  40270  0.050338
45                 n14  40270  0.050338
'''
# 可视化
(train.isnull().sum()/len(train)).plot.bar(figsize = (20,6),color=['#d6ecf0','#a3d900','#88ada6','#ffb3a7','#cca4e3','#a1afc9'])
f, ax = plt.subplots(1,1, figsize = (20,20))
cor = train[numerical_feature].corr()
sns.heatmap(cor, annot = True, linewidth = 0.2, linecolor = "white", ax = ax, fmt =".1g" )

train.duplicated().sum()
label = 'isDefault'
Y_label = train['isDefault']
numerical_feature.remove(label)
# 训练集
train[numerical_feature] = train[numerical_feature].fillna(train[numerical_feature].median())
# 测试集
test[numerical_feature] = test[numerical_feature].fillna(train[numerical_feature].median())
train[category_feature]
'''
       grade subGrade employmentLength   issueDate earliesCreditLine
0          E       E2          2 years  2014-07-01          Aug-2001
1          D       D2          5 years  2012-08-01          May-2002
2          D       D3          8 years  2015-10-01          May-2006
3          A       A4        10+ years  2015-08-01          May-1999
4          C       C2              NaN  2016-03-01          Aug-1977
     ...      ...              ...         ...               ...
799995     C       C4          7 years  2016-07-01          Aug-2011
799996     A       A4        10+ years  2013-04-01          May-1989
799997     C       C3        10+ years  2015-10-01          Jul-2002
799998     A       A4        10+ years  2015-02-01          Jan-1994
799999     B       B3          5 years  2018-08-01          Feb-2002

[800000 rows x 5 columns]
'''
# 训练集
train[category_feature] = train[category_feature].fillna(train[category_feature].mode())
# 测试集
test[category_feature] = test[category_feature].fillna(train[category_feature].mode())
train.isnull().sum()
'''

id                        0
loanAmnt                  0
term                      0
interestRate              0
installment               0
grade                     0
subGrade                  0
employmentTitle           0
employmentLength      46799
homeOwnership             0
annualIncome              0
verificationStatus        0
issueDate                 0
isDefault                 0
purpose                   0
postCode                  0
regionCode                0
dti                       0
delinquency_2years        0
ficoRangeLow              0
ficoRangeHigh             0
openAcc                   0
pubRec                    0
pubRecBankruptcies        0
revolBal                  0
revolUtil                 0
totalAcc                  0
initialListStatus         0
applicationType           0
earliesCreditLine         0
title                     0
policyCode                0
n0                        0
n1                        0
n2                        0
n3                        0
n4                        0
n5                        0
n6                        0
n7                        0
n8                        0
n9                        0
n10                       0
n11                       0
n12                       0
n13                       0
n14                       0
issueDateDT               0
dtype: int64
'''
train.employmentLength
'''
0           2 years
1           5 years
2           8 years
3         10+ years
4               NaN

799995      7 years
799996    10+ years
799997    10+ years
799998    10+ years
799999      5 years
Name: employmentLength, Length: 800000, dtype: object
'''
from sklearn.tree import DecisionTreeClassifier

empLenNotNullInd = train.employmentLength.notnull() # 不是空的行，返回True
columns = ['postCode','regionCode','employmentTitle','annualIncome'] # 用四个特征来预测employmentLength
train_empLen_X  = train.loc[empLenNotNullInd,columns]
train_empLen_y = train.employmentLength[empLenNotNullInd]

DTC = DecisionTreeClassifier() # 实例化
DTC.fit(train_empLen_X ,train_empLen_y) # 训练
print(DTC.score(train_empLen_X ,train_empLen_y))# 0.9809320486828881
# 预测
for data in [train,test]:
    empLenIsNullInd = data.employmentLength.isnull()
    test_empLen_X  = data.loc[empLenIsNullInd,columns]
    empLen_pred = DTC.predict(test_empLen_X)
    data.employmentLength[empLenIsNullInd] = empLen_pred
train.isnull().any().sum()
train['employmentLength'][:20]
'''
0       2 years
1       5 years
2       8 years
3     10+ years
4       5 years
5       7 years
6       9 years
7        1 year
8       5 years
9       6 years
10    10+ years
11      3 years
12      2 years
13    10+ years
14      2 years
15      2 years
16      9 years
17     < 1 year
18    10+ years
19      9 years
Name: employmentLength, dtype: object
'''
train['employmentLength'].value_counts(dropna=False).sort_index()
'''
1 year        55034
10+ years    276853
2 years       76435
3 years       68888
4 years       50893
5 years       54038
6 years       39517
7 years       37175
8 years       37903
9 years       31463
< 1 year      71801
Name: employmentLength, dtype: int64
'''
def find_outliers_by_3segama(data,fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return data
data_train = train.copy()
for fea in numerical_feature:
    data_train = find_outliers_by_3segama(data_train,fea)
    print(data_train[fea+'_outliers'].value_counts())
    print(data_train.groupby(fea+'_outliers')['isDefault'].sum())
    print('*'*10)
train['issueDate']
'''
0         2014-07-01
1         2012-08-01
2         2015-10-01
3         2015-08-01
4         2016-03-01
             ...
799995    2016-07-01
799996    2013-04-01
799997    2015-10-01
799998    2015-02-01
799999    2018-08-01
Name: issueDate, Length: 800000, dtype: object
'''
train.shape # (800000, 47)
import datetime
# 转化成时间格式 issueDateDT特征表示数据日期离数据集中日期最早的日期（2007-06-01）的天数
train['issueDate'] = pd.to_datetime(train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
train['issueDateDT'] = train['issueDate'].apply(lambda x: x-startdate).dt.days

train.shape # (800000, 48)
train[['issueDate','issueDateDT']]
#转化成时间格式
test['issueDate'] = pd.to_datetime(train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
test['issueDateDT'] = test['issueDate'].apply(lambda x: x-startdate).dt.days
plt.figure(1 , figsize = (10 , 8))
plt.hist(train['issueDateDT'], label='train');
plt.hist(test['issueDateDT'], label='test');
plt.legend();
plt.title('Distribution of issueDateDT dates');
#train 和 test issueDateDT 日期有重叠 所以使用基于时间的分割进行验证是不明智的
train[['issueDate','earliesCreditLine']]
'''
         issueDate earliesCreditLine
0       2014-07-01          Aug-2001
1       2012-08-01          May-2002
2       2015-10-01          May-2006
3       2015-08-01          May-1999
4       2016-03-01          Aug-1977
           ...               ...
799995  2016-07-01          Aug-2011
799996  2013-04-01          May-1989
799997  2015-10-01          Jul-2002
799998  2015-02-01          Jan-1994
799999  2018-08-01          Feb-2002

[800000 rows x 2 columns]
'''
train_earliesCreditLine_year = train['earliesCreditLine'].apply(lambda x:x[-4:]).astype('int64')
test_earliesCreditLine_year = test['earliesCreditLine'].apply(lambda x:x[-4:]).astype('int64')

train_issueDate_year = train['issueDate'].astype('str').apply(lambda x:x[:4]).astype('int64')
test_issueDate_year = test['issueDate'].astype('str').apply(lambda x:x[:4]).astype('int64')

train['CreditLine'] = train_issueDate_year - train_earliesCreditLine_year
test['CreditLine'] = test_issueDate_year - test_earliesCreditLine_year

train = train.drop(['earliesCreditLine','issueDate'],axis=1)
test = test.drop(['earliesCreditLine','issueDate'],axis=1)
train['CreditLine']
'''
0         13
1         10
2          9
3         16
4         39
          ..
799995     5
799996    24
799997    13
799998    21
799999    16
Name: CreditLine, Length: 800000, dtype: int64
'''
train.shape ## (800000, 47)
def employmentLength_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
for data in [train, test]:
    data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
    data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
    data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)
train['employmentLength'][:20]
'''
0       2 years
1       5 years
2       8 years
3     10+ years
4       5 years
5       7 years
6       9 years
7        1 year
8       5 years
9       6 years
10    10+ years
11      3 years
12      2 years
13    10+ years
14      2 years
15      2 years
16      9 years
17     < 1 year
18    10+ years
19      9 years
Name: employmentLength, dtype: object
'''
train['employmentLength']
'''
0          2
1          5
2          8
3         10
4          5
          ..
799995     7
799996    10
799997    10
799998    10
799999     5
Name: employmentLength, Length: 800000, dtype: int64
'''
a2z = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
a2z_code = np.arange(1,27)
a2z_mapping = dict(zip(a2z, a2z_code))

for data in [train,test]:
    data.loc[:,['grade','subGrade']] = data.loc[:,['grade','subGrade']].applymap(lambda g:g.replace(g[0], str(a2z.index(g[0])+1))).astype('int')
train[['grade','subGrade']]
train[['grade','subGrade']]
train[['homeOwnership','verificationStatus','purpose']]
'''
        homeOwnership  verificationStatus  purpose
0                   2                   2        1
1                   0                   2        0
2                   0                   2        0
3                   1                   1        4
4                   1                   2       10
              ...                 ...      ...
799995              1                   0        0
799996              0                   2        4
799997              1                   2        0
799998              0                   2        4
799999              0                   0        4

[800000 rows x 3 columns]
'''
train.shape# (800000, 47)
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder(sparse=False)
oh.fit(train[['homeOwnership','verificationStatus','purpose']])
OneHot1 = oh.transform(train[['homeOwnership','verificationStatus','purpose']])
OneHot2 = oh.transform(test[['homeOwnership','verificationStatus','purpose']])

OneHot1.shape# (800000, 23)
'''
array([[0., 0., 1., ..., 0., 0., 0.],
       [1., 0., 0., ..., 0., 0., 0.],
       [1., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 1., 0., ..., 0., 0., 0.],
       [1., 0., 0., ..., 0., 0., 0.],
       [1., 0., 0., ..., 0., 0., 0.]])
'''

train = pd.concat([train, pd.DataFrame(OneHot1)], axis=1)
test = pd.concat([test, pd.DataFrame(OneHot2)], axis=1)

train = train.drop(['homeOwnership','verificationStatus','purpose'],axis=1)
test = test.drop(['homeOwnership','verificationStatus','purpose'],axis=1)

train.shape# (800000, 67)
# 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
train['loanAmnt_bin1'] = np.floor_divide(train['loanAmnt'], 1000)
## 通过对数函数映射到指数宽度分箱
train['loanAmnt_bin2'] = np.floor(np.log10(train['loanAmnt']))
train['loanAmnt_bin3'] = pd.qcut(train['loanAmnt'], 10, labels=False)
train=train.drop(["id"],axis=1)
train.shape # (800000, 66)
test=test.drop(["id"],axis=1)
test.shape # (200000, 65)
train.corr()["isDefault"].sort_values()
train=train.drop(["initialListStatus","n5","n11","n12","n8","postCode","policyCode"],axis=1)
test=test.drop(["initialListStatus","n5","n11","n12","n8","postCode","policyCode"],axis=1)

train.shape# (800000, 59)
# 显示相关性高于0.6的变量
def getHighRelatedFeatureDf(corr_matrix, corr_threshold):
    highRelatedFeatureDf = pd.DataFrame(corr_matrix[corr_matrix>corr_threshold].stack().reset_index())
    highRelatedFeatureDf.rename({'level_0':'feature_x', 'level_1':'feature_y', 0:'corr'}, axis=1, inplace=True)
    highRelatedFeatureDf = highRelatedFeatureDf[highRelatedFeatureDf.feature_x != highRelatedFeatureDf.feature_y]
    highRelatedFeatureDf['feature_pair_key'] = highRelatedFeatureDf.loc[:,['feature_x', 'feature_y']].apply(lambda r:'#'.join(np.sort(r.values)), axis=1)
    highRelatedFeatureDf.drop_duplicates(subset=['feature_pair_key'],inplace=True)
    highRelatedFeatureDf.drop(['feature_pair_key'], axis=1, inplace=True)
    return highRelatedFeatureDf

getHighRelatedFeatureDf(train.corr(),0.6)

'''
              feature_x           feature_y      corr
2              loanAmnt         installment  0.953369
5          interestRate               grade  0.953269
6          interestRate            subGrade  0.970847
11                grade            subGrade  0.993907
22   delinquency_2years                 n13  0.658946
24         ficoRangeLow       ficoRangeHigh  1.000000
28              openAcc            totalAcc  0.700796
29              openAcc                  n2  0.658807
30              openAcc                  n3  0.658807
31              openAcc                  n4  0.618207
32              openAcc                  n7  0.830624
33              openAcc                  n8  0.646342
34              openAcc                  n9  0.660917
35              openAcc                 n10  0.998717
37               pubRec  pubRecBankruptcies  0.644402
44             totalAcc                  n5  0.623639
45             totalAcc                  n6  0.678482
46             totalAcc                  n8  0.761854
47             totalAcc                 n10  0.697192
53                   n1                  n2  0.807789
54                   n1                  n3  0.807789
55                   n1                  n4  0.829016
56                   n1                  n7  0.651852
57                   n1                  n9  0.800925
61                   n2                  n3  1.000000
62                   n2                  n4  0.663186
63                   n2                  n7  0.790337
64                   n2                  n9  0.982015
65                   n2                 n10  0.655296
70                   n3                  n4  0.663186
71                   n3                  n7  0.790337
72                   n3                  n9  0.982015
73                   n3                 n10  0.655296
79                   n4                  n5  0.717936
80                   n4                  n7  0.742157
81                   n4                  n9  0.639867
82                   n4                 n10  0.614658
86                   n5                  n7  0.618970
87                   n5                  n8  0.838066
97                   n7                  n8  0.774955
98                   n7                  n9  0.794465
99                   n7                 n10  0.829799
105                  n8                 n10  0.640729
113                  n9                 n10  0.660395
'''
col = ['installment','ficoRangeHigh','openAcc','n3','n9']
for data in [train,test]:
    data.drop(col,axis=1,inplace=True)
train.shape # (800000, 54)
train.var().sort_values()
col = ['applicationType']
for data in [train,test]:
    data.drop(col,axis=1,inplace=True)
train.shape  # (800000, 53)
label.value_counts()/len(label)
'''
0    0.800488
1    0.199513
Name: isDefault, dtype: float64
'''
import imblearn
from imblearn.over_sampling import SMOTE
over_samples = SMOTE(random_state=1234)
train_over,label_over = over_samples.fit_sample(train, label)

train_over.to_csv('train_over.csv',index=False)
label_over.to_csv('label_over.csv',index=False)

print(label_over.value_counts()/len(label_over))
print(train_over.shape)

from imblearn.under_sampling import RandomUnderSampler
under_samples = RandomUnderSampler(random_state=1234)
train_under, label_under = under_samples.fit_sample(train,label)

train_under.to_csv('train_under.csv',index=False)
label_under.to_csv('label_under.csv',index=False)

print(label_under.value_counts()/len(label_under))
print(train_under.shape)
X = train.drop(['isDefault'], axis=1)
y = train.loc[:,'isDefault']

kf = KFold(n_splits=5, shuffle=True, random_state=525)
X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2)
import lightgbm as lgb

cv_scores = []
for i, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train, y_train, X_val, y_val = X.iloc[train_index], y.iloc[train_index], X.iloc[val_index], y.iloc[val_index]

    train_matrix = lgb.Dataset(X_train, label=y_train)
    valid_matrix = lgb.Dataset(X_val, label=y_val)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'learning_rate': 0.1,
        'metric': 'auc',
        'min_child_weight': 1e-3,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 525,
        'nthread': 8,
        'silent': True,
    }

    model = lgb.train(params, train_set=train_matrix, num_boost_round=20000, valid_sets=valid_matrix, verbose_eval=1000,
                      early_stopping_rounds=200)
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    cv_scores.append(roc_auc_score(y_val, val_pred))
    print(cv_scores)

print("lgb_scotrainre_list:{}".format(cv_scores))
print("lgb_score_mean:{}".format(np.mean(cv_scores)))
print("lgb_score_std:{}".format(np.std(cv_scores)))
from sklearn import metrics
from sklearn.metrics import roc_auc_score

al_pre_lgb = model.predict(X_val, num_iteration=model.best_iteration)
fpr, tpr, threshold = metrics.roc_curve(y_val, val_pred)
roc_auc = metrics.auc(fpr, tpr)
print('AUC：{}'.format(roc_auc))

plt.figure(figsize=(8, 8))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.4f' % roc_auc)
plt.ylim(0,1)
plt.xlim(0,1)
plt.legend(loc='best')
plt.title('ROC')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# 画出对角线
plt.plot([0,1],[0,1],'r--')
plt.show()
X = train.drop(['isDefault'], axis=1)
y = train.loc[:,'isDefault']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3)
from xgboost.sklearn import XGBClassifier
clf1 = XGBClassifier(n_jobs=-1)
clf1.fit(Xtrain,Ytrain)
clf1.score(Xtest,Ytest)
from sklearn.metrics import roc_curve, auc

predict_proba = clf1.predict_proba(Xtest)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Ytest, predict_proba[:,1])
auc(false_positive_rate, true_positive_rate)
gra=GradientBoostingClassifier()
xgb=XGBClassifier()
lgb=LGBMClassifier()
models=[gra,xgb,lgb]
model_names=["gra","xgb","lgb"]

#交叉验证看看上述3个算法评分
for i,model in enumerate(models):
    score=cross_val_score(model,X,y,cv=5,scoring="accuracy",n_jobs=-1)
    print(model_names[i],np.array(score).round(3),round(score.mean(),3))
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

# 其余参数
other_params = {'learning_rate': 0.1,
                'n_estimators': 100,
                'max_depth': 5,
                'min_child_weight': 1,
                'seed': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1}

# 待调参数
param_test1 = {
 'max_depth':list(range(4,9,2)),
 'min_child_weight':list(range(1,6,2))
}

xgb1 = XGBClassifier(**other_params)
# 网格搜索
gs1 = GridSearchCV(xgb1,param_test1,cv = 5,scoring = 'roc_auc',n_jobs = -1,verbose=2)
best_model1=gs1.fit(Xtrain,Ytrain)
print('最优参数：',best_model1.best_params_)
print('最佳模型得分：',best_model1.best_score_)
other_params = {'learning_rate': 0.1,
                'n_estimators': 100,
                'max_depth': 4,
                'min_child_weight': 5,
                'seed': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1}


param_test = {
'gaama':[0,0.05,0.1,0.2,0.3]
}

xgb = XGBClassifier(**other_params)
gs = GridSearchCV(xgb,param_test,cv = 5,scoring = 'roc_auc',n_jobs = -1,verbose=2)
best_model=gs.fit(Xtrain,Ytrain)
print('最优参数：',best_model.best_params_)
print('最佳模型得分：',best_model.best_score_)
other_params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

param_test = {
'subsample':[0.6,0.7,0.8,0.9],
'colsample_bytree':[0.6,0.7,0.8,0.9]
}

xgb = XGBClassifier(**other_params)
gs = GridSearchCV(xgb,param_test,cv = 5,scoring = 'roc_auc',n_jobs = -1,verbose=2)
best_model=gs.fit(Xtrain,Ytrain)
print('最优参数：',best_model.best_params_)
print('最佳模型得分：',best_model.best_score_)
other_params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                    'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

param_test = {
'reg_alpha': [4,5,6,7],
'reg_lambda': [0,0.01,0.05, 0.1]
}

xgb = XGBClassifier(**other_params)
gs = GridSearchCV(xgb,param_test,cv = 5,scoring = 'roc_auc',n_jobs = -1,verbose=2)
best_model=gs.fit(Xtrain,Ytrain)
print('最优参数：',best_model.best_params_)
print('最佳模型得分：',best_model.best_score_)
other_params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 4, 'min_child_weight': 5, 'seed': 0,
                    'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 5, 'reg_lambda': 0.01}

param_test = {
'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2],
'n_estimators': [100,200,300,400,500]
}

xgb = XGBClassifier(**other_params)
gs = GridSearchCV(xgb,param_test,cv = 5,scoring = 'roc_auc',n_jobs = -1,verbose=2)
best_model=gs.fit(Xtrain,Ytrain)
print('最优参数：',best_model.best_params_)
print('最佳模型得分：',best_model.best_score_)
from xgboost.sklearn import XGBClassifier

clf = XGBClassifier(
    learning_rate=0.05,
    n_estimators=400,
    max_depth=4,
    min_child_weight=5,
    seed=0,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0,
    reg_alpha=5,
    reg_lambda=0.01,
    n_jobs=-1)

clf.fit(Xtrain, Ytrain)
clf.score(Xtest, Ytest)
from sklearn.metrics import roc_curve, auc

predict_proba = clf.predict_proba(Xtest)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Ytest, predict_proba[:,1])
auc(false_positive_rate, true_positive_rate)
from xgboost import plot_importance
plot_importance(clf)
plt.show()

# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.model_selection import train_test_split,cross_val_score	#划分数据 交叉验证
#
# clf1 = XGBClassifier(learning_rate=0.1, n_estimators=150, max_depth=3, min_child_weight=2, subsample=0.7,
#                      colsample_bytree=0.6, objective='binary:logistic')
# clf2 = RandomForestClassifier(n_estimators=50, max_depth=1, min_samples_split=4,
#                               min_samples_leaf=63,oob_score=True)
# clf3 = SVC(C=0.1)
#
# # 硬投票
# eclf = VotingClassifier(estimators=[
# 							('xgb', clf1),
# 							('rf', clf2),
# 							('svc', clf3)], voting='hard')
# # 比较模型融合效果
# for clf, label in zip([clf1, clf2, clf3, eclf], ['XGBBoosting', 'Random Forest', 'SVM', 'Ensemble']):
#     scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
from mlxtend.classifier import StackingClassifier

gra = GradientBoostingClassifier()
xgb = XGBClassifier()
lgb = LGBMClassifier()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[gra, xgb, lgb],
                          use_probas=True,
                          meta_classifier=lr)

sclf.fit(Xtrain, Ytrain)
pre = sclf.predict_proba(Xtest)[:, 1]
fpr, tpr, thresholds = roc_curve(Ytest, pre)
score = auc(fpr, tpr)
print(score)
