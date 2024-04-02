# -*- coding: utf-8 -*-

#%% XGBoost model

'''
======================================XGB-sklearnAPI======================================
'''    
 
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


### Features and labels

Features = pd.read_excel('Features several correct.xlsx',sheet_name='Sheet4').iloc[:,2:]
Labels = pd.read_excel('CN cycling function.xlsx',sheet_name='Sheet1').iloc[:,2:]

Labels=np.log(Labels+1)


### RFE and model

all_cor=[]
# all_r2_score=[]
# all_rmse=[]
# all_mape=[]

j=35
i=7
for i in range(2,20):
    
    x = Features.copy()
    y = pd.DataFrame(Labels.iloc[:,j])
    f_names=x.columns.values.tolist()
    l_names=y.columns.values.tolist()
    
    data = pd.concat([x,y],axis=1)
    data.iloc[:,-1].replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
       
    model = XGBR(n_estimators=100,random_state=0)
    selector = RFE(model, n_features_to_select=i, step=1).fit(x, np.ravel(y.values))
    sup = selector.support_ 
    feature_imp_rank_RFE = pd.concat([pd.DataFrame(x.columns,columns=['Feature']),pd.DataFrame(selector.ranking_,columns=['Importance'])],axis=1)
    colist = np.flatnonzero(sup) 
    x = x.iloc[:,colist]
    f_names = x.columns
        
    data = pd.concat([x,y],axis=1)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    f_names=x.columns.values.tolist()
    l_names=y.columns.values.tolist()
    
    kf = KFold(n_splits=10,shuffle=True,random_state=0) 
    o=[]
    corlist_test=[]
    corlist_train=[]
    rmselist_train=[]
    rmselist_test=[]
    r2list_train=[]
    r2list_test=[]
    mapelist_train=[]
    mapelist_test=[]
    
    for train_index, test_index in kf.split(x):
        
        x_train=x.iloc[train_index,:]
        x_train.columns=f_names
        y_train=y.iloc[train_index,:]        
        x_test=x.iloc[test_index,:]
        x_test.columns=f_names
        y_test=y.iloc[test_index,:]
        
        model=XGBR(n_estimators=30,learning_rate=0.15,max_depth=7,objective='reg:squarederror',random_state=0)
        model.fit(x_train,np.ravel(y_train.values))
        
        y_train_pred=pd.DataFrame(model.predict(x_train).reshape(y_train.shape),index=train_index)
        y_test_pred=pd.DataFrame(model.predict(x_test).reshape(y_test.shape),index=test_index)
        
        cor_train=np.corrcoef(y_train[l_names[0]],y_train_pred[0])
        cor_test=np.corrcoef(y_test[l_names[0]],y_test_pred[0])        
        r2_train=r2_score(y_train[l_names[0]],y_train_pred[0])
        r2_test=r2_score(y_test[l_names[0]],y_test_pred[0])
        rmse_train=np.sqrt(mean_squared_error(y_train[l_names[0]],y_train_pred[0]))
        rmse_test=np.sqrt(mean_squared_error(y_test[l_names[0]],y_test_pred[0]))
        mape_train = mean_absolute_percentage_error(y_train[l_names[0]],y_train_pred[0])
        mape_test = mean_absolute_percentage_error(y_test[l_names[0]],y_test_pred[0])
        
        corlist_train.append(cor_train[1,0])
        corlist_test.append(cor_test[1,0])
        r2list_train.append(r2_train)
        r2list_test.append(r2_test)
        rmselist_train.append(rmse_train)
        rmselist_test.append(rmse_test)        
        mapelist_train.append(mape_train)
        mapelist_test.append(mape_test)
        
        o.append(y_train[l_names[0]])
        o.append(y_train_pred[0])
        o.append(y_test[l_names[0]])
        o.append(y_test_pred[0])
        #pbar.update()           
        #imp.append(model.feature_importances_)
        #.columns = ['niche breath_pred']
        #plot_data = pd.concat([y_test,y_test_pred], axis=1)
        #sns.regplot(x='niche breath',y='niche breath_pred',data=plot_data)                                               
        #plt.show()
        
    print('cor:',np.mean(corlist_test))
    print('r2_score:',np.mean(r2list_test))
    print("RMSE:",np.mean(rmselist_test))
    print("MAPE:",np.mean(mapelist_test))
    print('-----------------------------------------')
    
    all_cor.append(np.mean(corlist_test))
    # all_r2_score.append(np.mean(r2list_test))
    # all_rmse.append(np.mean(rmselist_test))
    # all_mape.append(np.mean(mape_test))
    

    

"""
----------------------------------Adjustment of hyperparameters-------------------------------------
"""    

import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product
from tqdm import tqdm, trange


Features = pd.read_excel('Features several correct.xlsx',sheet_name='Sheet4').iloc[:,2:]
Labels = pd.read_excel('CN cycling function.xlsx',sheet_name='Sheet1').iloc[:,2:]
Labels=np.log(Labels+1)

def Try(label,n_estimators,learning_rate,max_depth,n_features):
        
    x = Features.copy()
    y = pd.DataFrame(Labels.iloc[:,label])
    f_names=x.columns.values.tolist()
    l_names=y.columns.values.tolist()
    
    data = pd.concat([x,y],axis=1)
    data.iloc[:,-1].replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    
    Rmodel = XGBR(n_estimators=100,random_state=0)
    selector = RFE(Rmodel, n_features_to_select=n_features, step=1).fit(x, np.ravel(y.values))
    sup = selector.support_ 
    colist = np.flatnonzero(sup) 
    x = x.iloc[:,colist]
    f_names = x.columns
        
    data = pd.concat([x,y],axis=1)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    f_names=x.columns.values.tolist()
    l_names=y.columns.values.tolist()
    
    kf = KFold(n_splits=10,shuffle=True,random_state=0) 
    #o=[]
    corlist_test=[]
    mapelist_test=[]
    #rmsel_train=[]
    #rmsel_test=[]
    
    for train_index, test_index in kf.split(x):
        
        x_train=x.iloc[train_index,:]
        x_train.columns=f_names
        y_train=y.iloc[train_index,:]
        x_test=x.iloc[test_index,:]
        x_test.columns=f_names
        y_test=y.iloc[test_index,:]
        
        model=XGBR(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,objective='reg:squarederror',random_state=0)
        model.fit(x_train,np.ravel(y_train.values))
        
        y_train_pred=pd.DataFrame(model.predict(x_train).reshape(y_train.shape),index=train_index)
        y_test_pred=pd.DataFrame(model.predict(x_test).reshape(y_test.shape),index=test_index)
        
        #cor_train=np.corrcoef(y_train[l_names[0]],y_train_pred[0])
        cor_test=np.corrcoef(y_test[l_names[0]],y_test_pred[0])        
        #r2_train=r2_score(y_train[l_names[0]],y_train_pred[0])
        #r2_test=r2_score(y_test[l_names[0]],y_test_pred[0])
        #rmse_train=np.sqrt(mean_squared_error(y_train[l_names[0]],y_train_pred[0]))
        #rmse_test=np.sqrt(mean_squared_error(y_test[l_names[0]],y_test_pred[0]))
        #mape_train = mean_absolute_percentage_error(y_train[l_names[0]],y_train_pred[0])
        mape_test = mean_absolute_percentage_error(y_test[l_names[0]],y_test_pred[0])
                

        #corlist_train.append(corr_train[1,0])
        corlist_test.append(cor_test[1,0])
        mapelist_test.append(mape_test)
        #rmsel_train.append(rmse_train)
        #rmsel_test.append(rmse_test)
        #o.append(y_train[l_names[0]])
        #o.append(y_train_pred[0])
        #o.append(y_test[l_names[0]])
        #o.append(y_test_pred[0])

    print(np.mean(corlist_test),np.mean(mapelist_test))
    #print('r2_score:',r2_score(y_test_pred[0],y_test[l_names[0]]))
    return(np.mean(corlist_test),np.mean(mapelist_test))
    #return(r2_score(y_test_pred[0],y_test[l_names[0]]))

label = 35
n_estimators = list(range(30,300,10))
learning_rate = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2]
max_depth = [3,4,5,6,7,8,9]
n_features = [9,8,7]
loop_val = [n_estimators,learning_rate,max_depth,n_features]

all_cor=[]
all_mape=[]
all_p=[] 

n = len(n_estimators) * len(learning_rate) * len(max_depth) * len(n_features)

with tqdm(total=n) as pbar:
    for i in product(*loop_val):
        cor,mape=Try(label,i[0],i[1],i[2],i[3])
        all_cor.append(cor)
        all_mape.append(mape)
        all_p.append(i) 
        pbar.update(1)

params = pd.DataFrame(all_p)
cor = pd.DataFrame(all_cor)
mape=pd.DataFrame(all_mape)
result = pd.concat([params,cor,mape],axis=1)
result.columns=['n_estimators','learning_rate','max_depth','n_features','cor','MAPE']



"""
----------------------------------ECE and CI-------------------------------------
"""   

import time
import pandas as pd
from sklearn.datasets import make_regression
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.stats
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import ShuffleSplit
import os

alpha = 0.975

# log cosh quantile is a regularized quantile loss function
def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err)**2
        return grad, hess
    return _log_cosh_quantile

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def ci_t(data, confidence):
    sample_mean = np.mean(data)
    sample_std = np.std(data)
    sample_size = len(data)
    df = len(data) - 1

    alpha = (1 - confidence) / 2
    t_score = scipy.stats.t.isf(alpha, df)

    ME = t_score * sample_std / np.sqrt(sample_size)

    lower_limit = sample_mean - ME
    upper_limit = sample_mean + ME

    return (lower_limit, upper_limit)

def bootstrap_mean(data):
    return np.mean(np.random.choice(data, size=len(data)))


def draw_bootstrap(data, times=1):
    bs_mean = np.empty(times)

    for i in range(times):
        bs_mean[i] = bootstrap_mean(data)

    return bs_mean

def expected_calibration_error(y, proba, bins='fd'):
    import numpy as np
    bin_count, bin_edges = np.histogram(proba, bins=bins)
    n_bins = len(bin_count)

    bin_edges[0] -= 1e-8  # because left edge is not included
    bin_id = np.digitize(proba, bin_edges, righ=True) - 1

    bin_ysum = np.bincount(bin_id, weights=y, minlength=n_bins)
    bin_probasum = np.bincount(bin_id, weights=proba, minlength=n_bins)
    bin_ymean = np.divide(bin_ysum, bin_count, out=np.zeros(n_bins), where=bin_count > 0)
    bin_probamean = np.divide(bin_probasum, bin_count, out=np.zeros(n_bins), where=bin_count > 0)

    ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(proba)
    return ece



X = pd.read_excel('Features several correct.xlsx',sheet_name='Sheet4').iloc[:,2:]
y = pd.read_excel('N cycling function.xlsx',sheet_name='Sheet1').iloc[:,2:]
Labels['denitrification']=np.log(Labels['denitrification']+1)  
  
j=6
X = Features.copy()
y = pd.DataFrame(Labels.iloc[:,j])
f_names=x.columns.values.tolist()
l_names=y.columns.values.tolist()

data = pd.concat([X,y],axis=1)
data.iloc[:,-1].replace(0,np.nan,inplace=True)
data.dropna(axis=0,subset=[l_names[0]],inplace=True)
X = data.iloc[:,:-1]
y = pd.DataFrame(data.iloc[:,-1])

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score  # 用来划分训练集和测试集
from sklearn.metrics import mean_absolute_error,r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

#features = list(X_train.columns)
# print("输出表头：",features)

#model = xgb.XGBRegressor(learning_rate = 0.1,
#                         n_estimators = 80,
#                         max_depth = 10,
#                         min_child_weight = 3,
#                         subsample = 0.9,
#                         colsample_bytree = 0.7,
#                         seed = 2)

#normal predict


multi_xgb = XGBR(n_estimators=200,learning_rate=0.02,max_depth=7,objective='reg:squarederror',random_state=0)
model = multi_xgb.fit(X_train, y_train)
y_pred = multi_xgb.predict(X_test)

print("R2 test:", model.score(X_test, y_test))
print("R2 train:", model.score(X_train, y_train))
print("RMSE:", metrics.mean_squared_error(y_test, y_pred)**0.05)
MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
print("CI:", ci_t(y_pred, 0.95))
volume = y_pred
size = len(volume)
bs_mean = draw_bootstrap(volume, 10000)
plt.hist(bs_mean, bins=27, density=True, stacked=True, rwidth=0.9)
plt.show()
print("95% CI level:", np.percentile(bs_mean, [2.5, 97.5]))

#ece = expected_calibration_error(y_pred, MAPE)
#print('ECE:',ece)

#over predict
multi_xgb = XGBR(n_estimators=200,learning_rate=0.02,max_depth=7,objective=log_cosh_quantile(alpha),random_state=0)
model = multi_xgb.fit(X_train, y_train)
y_pred_upper = multi_xgb.predict(X_test)

#under predict
multi_xgb = XGBR(n_estimators=200,learning_rate=0.02,max_depth=7,objective=log_cosh_quantile(1-alpha),random_state=0)
model = multi_xgb.fit(X_train, y_train)
y_pred_lower = multi_xgb.predict(X_test)
y_true = np.array(y_test.values)

#index = res['upper_bound'] < 0
#print(res[res['upper_bound'] < 0])
#print(X_test[index])
#max_length = 350

fig = plt.figure()
plt.plot(list(y_true), 'gx', label=u'real value')
plt.plot(y_pred_upper, 'y_', label=u'Q up')
plt.plot(y_pred_lower, 'b_', label=u'Q low')
index = np.array(range(0, len(y_pred_upper)))
plt.fill(np.concatenate([index, index[::-1]]),
         np.concatenate([y_pred_upper, y_pred_lower[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% prediction interval')
plt.xlabel('$Data$')
plt.ylabel('$Index$')
plt.legend(loc='upper right')
plt.show()

res = pd.DataFrame({'lower_bound' : y_pred_lower, 'true': y_test['denitrification'], 'upper_bound': y_pred_upper})
count = res[(res.true >= res.lower_bound) & (res.true <= res.upper_bound)].shape[0]
total = res.shape[0]
p=count/total
#print(f'pref = {count} / {total}')
print('Within CI Percent: {:.2%}'.format(count/total))
ece=(alpha-model.score(X_test, y_test))*p
print(ece)


#%% XGBoost output

import numpy as np
import pandas as pd
import os
import xgboost as xgb
from xgboost import XGBRegressor as XGBR
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

parameter = pd.read_excel('Hyperparameters.xlsx')

Labels = pd.read_excel('CN cycling function.xlsx').iloc[:,2:]
Labels=np.log(Labels+1)

'''
Features = pd.read_excel('Features several correct.xlsx',sheet_name='Sheet4')
Label_names=parameter.iloc[:,0]
writer3=pd.ExcelWriter('Features selected.xlsx')
j=0
for j in list(range(0,35)):
    river_no=Features.iloc[:,[0,1]]
    feature_selected=Features.loc[:,parameter.iloc[j,:][13:].dropna().tolist()]
    feature=pd.concat([river_no,feature_selected],axis=1)
    feature.to_excel(writer3,sheet_name=Label_names[j])
    
writer3.save()
writer3.close()
'''

writer1=pd.ExcelWriter('op.xlsx')
writer2=pd.ExcelWriter('cor.xlsx')

j_index=list(range(0,35))
j_index.remove(2)
j=j_index[0]

for j in j_index:

    eta = parameter.loc[:,['eta']].iloc[j,0]
    n_round = int(parameter.loc[:,['n_round']].iloc[j,0])
    max_depth = int(parameter.loc[:,['max_depth']].iloc[j,0])

    y = pd.DataFrame(Labels.iloc[:,j])
    l_names=y.columns.values.tolist()
    
    Features = pd.read_excel('Features selected.xlsx',sheet_name=l_names[0],index_col=0).iloc[:,2:]
    x = Features.copy()
    f_names = x.columns
    
    data = pd.concat([x,y],axis=1)
    data.iloc[:,-1].replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    
    kf = KFold(n_splits=10,shuffle=True,random_state=0) 
    
    train_pred=[]
    test_pred=[]
    corlist_train=[]
    corlist_test=[]
    rmselist_train=[]
    rmselist_test=[]
        
    for train_index, test_index in kf.split(x):
        x_train=x.iloc[train_index,:]
        x_train.columns=f_names
        y_train=y.iloc[train_index,:]
        
        x_test=x.iloc[test_index,:]
        x_test.columns=f_names
        y_test=y.iloc[test_index,:]
        
        model=XGBR(n_estimators=n_round,learning_rate=eta,max_depth=max_depth,objective='reg:squarederror',random_state=0)
        model.fit(x_train,np.ravel(y_train.values))
        
        y_train_pred=pd.DataFrame(model.predict(x_train).reshape(y_train.shape),index=train_index)
        y_test_pred=pd.DataFrame(model.predict(x_test).reshape(y_test.shape),index=test_index)
        
        cor_train=np.corrcoef(y_train[l_names[0]],y_train_pred[0])
        cor_test=np.corrcoef(y_test[l_names[0]],y_test_pred[0])    
        rmse_train=np.sqrt(mean_squared_error(y_train[l_names[0]],y_train_pred[0]))
        rmse_test=np.sqrt(mean_squared_error(y_test[l_names[0]],y_test_pred[0]))
        
        corlist_train.append(cor_train[1,0])
        corlist_test.append(cor_test[1,0])
        rmselist_train.append(rmse_train)
        rmselist_test.append(rmse_test)
        
        train_pred.append(y_train_pred[0])
        test_pred.append(y_test_pred[0])
    print(np.mean(corlist_test))
    
    train_pred=pd.DataFrame(train_pred).T
    train_pred.index=y.index
    test_pred=pd.DataFrame(test_pred).T
    test_pred.index=y.index
    
    obs_pre_df=pd.concat([y,train_pred,test_pred],axis=1)
    obs_pre_df.columns=('Observation','train1','train2','train3','train4','train5',
                            'train6','train7','train8','train9','train10',
                            'test1','test2','test3','test4','test5',
                            'test6','test7','test8','test9','test10')
    obs_pre_df.to_excel(writer1,sheet_name=l_names[0])
    
    cor_df=pd.DataFrame({'train':corlist_train,'test':corlist_test,'rmse_train':rmselist_train,'rmse_test':rmselist_test})
    cor_df.to_excel(writer2,sheet_name=l_names[0])
    
writer1.save()
writer1.close()
writer2.save()
writer2.close()


#%% Causal interferance

'''
--------------------------Causal relationship between features and labels-------------------------
'''

from econml.solutions.causal_analysis import CausalAnalysis
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

writer1=pd.ExcelWriter('Causal relationship_1.xlsx')

Labels = pd.read_excel('CN cycling function.xlsx').iloc[:,2:]
Labels=np.log(Labels+1)
Label_names = Labels.columns

j=6
for j in [8,9,1,6,10,7,11,12]:
    print(j)
    y=pd.DataFrame(Labels.iloc[:,j])
    l_names=y.columns.values.tolist()
    
    Features = pd.read_excel('Features selected.xlsx',sheet_name=l_names[0],index_col=0).iloc[:,2:]
    x = Features.copy()
    f_names = x.columns
    
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x))
    x.columns=f_names
        
    '''
    data = pd.concat([x,y],axis=1)
    data.replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    data.replace(np.nan,0,inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    '''
    
    ca = CausalAnalysis(
        feature_inds=f_names,
        categorical=[],
        classification=False,
        nuisance_models="automl",
        heterogeneity_model="linear",
        random_state=123,
        n_jobs=-1
        )
    ca.fit(x, np.ravel(y))
    
    global_summ = ca.global_causal_effect(alpha=0.05)
    global_summ=global_summ.sort_values(by="p_value")
    
    global_summ.to_excel(writer1,sheet_name=l_names[0])

writer1.save()
writer1.close()


'''
--------------------------Causal relationship among features---------------------------
'''

from econml.solutions.causal_analysis import CausalAnalysis
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

l_names=['nitrite_ammonification','denitrification','nitrogen_fixation','nitrate_reduction']

j=1
result=pd.DataFrame()
for j in range(len(l_names)):
    
    l_name=l_names[j]
    print(l_name)
    Features = pd.read_excel('Features selected.xlsx',sheet_name=l_name,index_col=0).iloc[:,2:]
    f_names = Features.columns
    
    scaler = StandardScaler()
    Features = pd.DataFrame(scaler.fit_transform(Features))
    Features.columns=f_names
    n_f=len(f_names)
    
    i=1
    for i in range(n_f):
        x=Features.drop(f_names[i],axis=1)
        y=pd.DataFrame(Features.iloc[:,i])
        print(y.columns[0])
        
        ca = CausalAnalysis(
            feature_inds=x.columns,
            categorical=[],
            classification=False,
            nuisance_models="automl",
            heterogeneity_model="linear",
            n_jobs=-1,
            random_state=123)
        ca.fit(x,np.ravel(y))
        
        global_summ = ca.global_causal_effect(alpha=0.05)
        global_summ = global_summ.sort_values(by="p_value")
        
        global_summ.insert(0,'Function',l_name)
        global_summ.insert(0,'label_feature',y.columns[0])
        
        result=pd.concat([result,global_summ])
        
result.to_excel('Causal relationship_2.xlsx')



#%% SHAP

import numpy as np
import pandas as pd
import os
from pdpbox import pdp, get_dataset, info_plots
from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import train_test_split
import matplotlib as plt
import shap

writer1=pd.ExcelWriter('shap values.xlsx')

parameter = pd.read_excel('Hyperparameters.xlsx')

Labels = pd.read_excel('CN cycling function.xlsx').iloc[:,2:]
Labels=np.log(Labels+1)
Label_names = Labels.columns

j_index=list(range(0,35))
j_index.remove(2)
j=6
i=0
for j in j_index:
    
    n_estimators=int(parameter.loc[:,['n_round']].iloc[j,0])
    learning_rate=parameter.loc[:,['eta']].iloc[j,0]
    max_depth = int(parameter.loc[:,['max_depth']].iloc[j,0])

    y = pd.DataFrame(Labels.iloc[:,j])
    l_names=y.columns.values.tolist()
    Features = pd.read_excel('Features selected.xlsx',sheet_name=l_names[0],index_col=0).iloc[:,2:]
    x = Features.copy()
    f_names = x.columns
    
    data = pd.concat([x,y],axis=1)
    data.iloc[:,-1].replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])

    model=XGBR(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,random_state=0)
    model.fit(x,np.ravel(y.values))
        
    
    explainer=shap.TreeExplainer(model) 
    shap_values1 = explainer.shap_values(x) 
    shap_values2 = explainer(x) 
    # print(shap_values1.shape)
    
    y_base = explainer.expected_value
    # print(y_base)
    
    # fig = plt.gcf()
    shap.plots.heatmap(shap_values2, max_display=14,show=False)    
    # outpath='plot/shap/heatmap_shap.pdf'
    # fig.savefig(outpath, dpi=300, bbox_inches = 'tight')
 
    fig = shap.summary_plot(shap_values1, x, show=False)
    # fig.savefig('plot/shap/summary_shap.pdf', bbox_inches = 'tight')  
    
    fig = shap.summary_plot(shap_values1, x, plot_type='bar', show=False)
    # fig.savefig('plot/shap/mean_shap.pdf',dpi=300, bbox_inches = 'tight')
    
    shap.force_plot(explainer.expected_value, shap_values1, x)
    # fig.savefig('plot/shap/force_shap.pdf')

    shap.plots.beeswarm(shap_values2)

    shap.dependence_plot('DOC', shap_values1, x, interaction_index=None)  
        
    shap_interaction_values = explainer.shap_interaction_values(x)
    shap.summary_plot(shap_interaction_values, x, show=False)
    # plt.savefig('plot/interaction_shap.pdf',dpi=300, bbox_inches = 'tight')
    
    shap.dependence_plot('DOC', shap_values1, x, interaction_index='Niche_B')
    
    shap_values=pd.DataFrame(shap_values1,index=x.index,columns=f_names)
    shap_values.to_excel(writer1,sheet_name=l_names[0])
    
writer1.save()
writer1.close()


#%% pdp

'''
-----------------------------Single featrue----------------------------------------
'''
import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor as XGBR
import copy


parameter = pd.read_excel('Hyperparameters.xlsx')


Labels = pd.read_excel('CN cycling function.xlsx').iloc[:,2:]
Labels=np.log(Labels+1)
Label_names = Labels.columns


j_index=[1,6,7,8,9,14,17,18,19,20,21,31]


j=6
i=0
k=0
for j in j_index:
    
    n_estimators=int(parameter.loc[:,['n_round']].iloc[j,0])
    learning_rate=parameter.loc[:,['eta']].iloc[j,0]
    max_depth = int(parameter.loc[:,['max_depth']].iloc[j,0])

    y = pd.DataFrame(Labels.iloc[:,j])
    l_names=y.columns.values.tolist()
    Features = pd.read_excel('Features selected.xlsx',sheet_name=l_names[0],index_col=0).iloc[:,2:]
    x = Features.copy()
    f_names = x.columns
    
    data = pd.concat([x,y],axis=1)
    data.iloc[:,-1].replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])

    model=XGBR(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,random_state=0)
    model.fit(x,np.ravel(y.values))
    
    grid_resolution=100    
    
    for i in range(len(f_names)):
        f_range=np.linspace(data[f_names[i]].quantile(q=0.05),data[f_names[i]].quantile(q=0.95),grid_resolution)
        
        pdp_result=pd.DataFrame(np.zeros(shape=(grid_resolution,2)))
        pdp_result.columns=[f_names[i],l_names[0]]
        
        for k in range(len(f_range)):
            pdp_x=copy.deepcopy(x)
            pdp_x[f_names[i]]=f_range[k]
            pdp_pred=model.predict(pdp_x).mean()
            pdp_result.iloc[k,0]=f_range[k]
            pdp_result.iloc[k,1]=pdp_pred
            
        pdp_result.to_csv('plot/pdp-myself/single/'+f_names[i]+'-'+l_names[0]+'.csv')
        


'''
-----------------------------Two featrues----------------------------------------
'''  
     
import numpy as np
import pandas as pd
import os
from xgboost import XGBRegressor as XGBR
import copy
import itertools

parameter = pd.read_excel('Hyperparameters.xlsx')

Labels = pd.read_excel('CN cycling function.xlsx').iloc[:,2:]
Labels=np.log(Labels+1)
Label_names = Labels.columns

j_index=[6,7,9,12]

j=6

f_s=[0,1,2,3,4,5,
     [['DOC','NO3_N'],['DOC','Chao1'],['DOC','Niche_B'],['NO3_N','Chao1'],['NO3_N','Niche_B'],['Chao1','Niche_B']],    
     [['PM10','Chao1'],['PM10','Niche_B'],['Chao1','Niche_B']],
     8,
     [['Chao1','pH']],
     10,11,
     [['Shannon','Niche_B'],['Shannon','K'],['Shannon','Al'],['Niche_B','K'],['Niche_B','Al'],['K','Al']]]

for j in j_index:
    
    n_estimators=int(parameter.loc[:,['n_round']].iloc[j,0])
    learning_rate=parameter.loc[:,['eta']].iloc[j,0]
    max_depth = int(parameter.loc[:,['max_depth']].iloc[j,0])
       
    y = pd.DataFrame(Labels.iloc[:,j])
    l_names=y.columns.values.tolist()
    Features = pd.read_excel('Features selected.xlsx',sheet_name=l_names[0],index_col=0).iloc[:,2:]
    x = Features.copy()
    f_names = x.columns
    
    data = pd.concat([x,y],axis=1)
    data.iloc[:,-1].replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    
    model=XGBR(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,random_state=0)
    model.fit(x,np.ravel(y.values))
    
    print(Label_names[j])
    print(f_names.values)
    
    f=f_s[j][0]
    for f in f_s[j]:            
        grid_resolution=100
              
        f1=f[0]
        f2=f[1]
                
        range_1=np.linspace(data[f1].quantile(q=0.05),data[f1].quantile(q=0.95),grid_resolution)        
        range_2=np.linspace(data[f2].quantile(q=0.05),data[f2].quantile(q=0.95),grid_resolution) 
        
        pdp_result=pd.DataFrame(np.zeros(shape=(grid_resolution*grid_resolution,3)))
        pdp_result.columns=[f1,f2,l_names[0]]
        
        k=0        
        for i in itertools.product(range_1,range_2):
            pdp_x=copy.deepcopy(x)
            pdp_x[f1]=i[0]
            pdp_x[f2]=i[1]
            pdp_pred=model.predict(pdp_x).mean()
            
            pdp_result.iloc[k,0]=i[0]
            pdp_result.iloc[k,1]=i[1]
            pdp_result.iloc[k,2]=pdp_pred
            
            k=k+1
            
            if k > grid_resolution*grid_resolution:
                break
            
        pdp_result.to_csv('plot/pdp-myself/two/'+f1+'-'+f2+'-'+l_names[0]+'.csv') 


#%% RF

import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


Features = pd.read_excel('Features several correct.xlsx',sheet_name='Sheet4').iloc[:,2:]
Labels = pd.read_excel('CN cycling function.xlsx',sheet_name='Sheet1').iloc[:,2:]
Labels=np.log(Labels+1)


all_cor=[]
all_r2_score=[]
all_rmse=[]
all_mape=[]
all_l_names=[]

l_list=[8,9,1,6,10,7,11,12]

j=l_list[0]
for j in l_list:   
    x = Features.copy()
    y = pd.DataFrame(Labels.iloc[:,j])
    f_names=x.columns.values.tolist()
    l_names=y.columns.values.tolist()   
    
    data = pd.concat([x,y],axis=1)
    data.iloc[:,-1].replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
          
    model = RandomForestRegressor(random_state=0)
    selector = RFE(model, n_features_to_select=4, step=1).fit(x, np.ravel(y.values))
    sup = selector.support_ 
    colist = np.flatnonzero(sup) 
    x = x.iloc[:,colist]
    f_names = x.columns
        
    data = pd.concat([x,y],axis=1)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    f_names=x.columns.values.tolist()
    l_names=y.columns.values.tolist()
    
    kf = KFold(n_splits=10,shuffle=True,random_state=0) 
    o=[]
    corlist_test=[]
    corlist_train=[]
    rmselist_train=[]
    rmselist_test=[]
    r2list_train=[]
    r2list_test=[]
    mapelist_train=[]
    mapelist_test=[]
    
    for train_index, test_index in kf.split(x):
               
        x_train=x.iloc[train_index,:]
        x_train.columns=f_names
        y_train=y.iloc[train_index,:]        
        x_test=x.iloc[test_index,:]
        x_test.columns=f_names
        y_test=y.iloc[test_index,:]
                
        model=RandomForestRegressor(random_state=0)
        model.fit(x_train,np.ravel(y_train.values))
                
        y_train_pred=pd.DataFrame(model.predict(x_train).reshape(y_train.shape),index=train_index)
        y_test_pred=pd.DataFrame(model.predict(x_test).reshape(y_test.shape),index=test_index)
        
        cor_train=np.corrcoef(y_train[l_names[0]],y_train_pred[0])
        cor_test=np.corrcoef(y_test[l_names[0]],y_test_pred[0])        
        r2_train=r2_score(y_train[l_names[0]],y_train_pred[0])
        r2_test=r2_score(y_test[l_names[0]],y_test_pred[0])
        rmse_train=np.sqrt(mean_squared_error(y_train[l_names[0]],y_train_pred[0]))
        rmse_test=np.sqrt(mean_squared_error(y_test[l_names[0]],y_test_pred[0]))
        mape_train = mean_absolute_percentage_error(y_train[l_names[0]],y_train_pred[0])
        mape_test = mean_absolute_percentage_error(y_test[l_names[0]],y_test_pred[0])
        
        corlist_train.append(cor_train[1,0])
        corlist_test.append(cor_test[1,0])
        r2list_train.append(r2_train)
        r2list_test.append(r2_test)
        rmselist_train.append(rmse_train)
        rmselist_test.append(rmse_test)        
        mapelist_train.append(mape_train)
        mapelist_test.append(mape_test)
        
        o.append(y_train[l_names[0]])
        o.append(y_train_pred[0])
        o.append(y_test[l_names[0]])
        o.append(y_test_pred[0])
        #pbar.update()           
        #imp.append(model.feature_importances_)
        #.columns = ['niche breath_pred']
        #plot_data = pd.concat([y_test,y_test_pred], axis=1)
        #sns.regplot(x='niche breath',y='niche breath_pred',data=plot_data)                                               
        #plt.show()
        
    print('cor:',np.mean(corlist_test))
    print('r2_score:',np.mean(r2list_test))
    print("RMSE:",np.mean(rmselist_test))
    print("MAPE:",np.mean(mapelist_test))
    print('-----------------------------------------')
    
    all_cor.append(np.mean(corlist_test))
    all_r2_score.append(np.mean(r2list_test))
    all_rmse.append(np.mean(rmselist_test))
    all_mape.append(np.mean(mape_test))
    all_l_names.append(l_names[0])

result=pd.DataFrame({'function':all_l_names,'cor':all_cor,'r2_score':all_r2_score,'rmse':all_rmse,'mape':all_mape})
result.to_excel('RF.xlsx',index=False)


#%% SVM

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import os 


Features = pd.read_excel('Features several correct.xlsx',sheet_name='Sheet4').iloc[:,2:]
Labels = pd.read_excel('CN cycling function.xlsx',sheet_name='Sheet1').iloc[:,2:]
Labels=np.log(Labels+1)

all_cor=[]
all_r2_score=[]
all_rmse=[]
all_mape=[]
all_l_names=[]

l_list=[8,9,1,6,10,7,11,12]

j=l_list[0]
for j in l_list: 
    x = Features.copy()
    f_names = x.columns
    y = pd.DataFrame(Labels.iloc[:,j])
    l_names=y.columns.values.tolist()
    
    data = pd.concat([x,y],axis=1)
    data.iloc[:,-1].replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    
    ''' 
    model = MLPRegressor(solver='adam', hidden_layer_sizes=(50, 50), activation='tanh', max_iter=5000)
    model=MLPRegressor()
    selector = RFE(model, n_features_to_select=4, step=1).fit(x, np.ravel(y.values))
    sup = selector.support_ 
    colist = np.flatnonzero(sup) 
    x = x.iloc[:,colist]
    f_names = x.columns
    '''  
    
    data = pd.concat([x,y],axis=1)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    f_names=x.columns.values.tolist()
    l_names=y.columns.values.tolist()
    
    kf = KFold(n_splits=10,shuffle=True,random_state=0) 
    o=[]
    corlist_test=[]
    corlist_train=[]
    rmselist_train=[]
    rmselist_test=[]
    r2list_train=[]
    r2list_test=[]
    mapelist_train=[]
    mapelist_test=[]
    
    for train_index, test_index in kf.split(x):
        
        x_train=x.iloc[train_index,:]
        x_train.columns=f_names
        y_train=y.iloc[train_index,:]        
        x_test=x.iloc[test_index,:]
        x_test.columns=f_names
        y_test=y.iloc[test_index,:]
        
        model=svm.SVR()
        model.fit(x_train,np.ravel(y_train.values))

        y_train_pred=pd.DataFrame(model.predict(x_train).reshape(y_train.shape),index=train_index)
        y_test_pred=pd.DataFrame(model.predict(x_test).reshape(y_test.shape),index=test_index)
        
        cor_train=np.corrcoef(y_train[l_names[0]],y_train_pred[0])
        cor_test=np.corrcoef(y_test[l_names[0]],y_test_pred[0])        
        r2_train=r2_score(y_train[l_names[0]],y_train_pred[0])
        r2_test=r2_score(y_test[l_names[0]],y_test_pred[0])
        rmse_train=np.sqrt(mean_squared_error(y_train[l_names[0]],y_train_pred[0]))
        rmse_test=np.sqrt(mean_squared_error(y_test[l_names[0]],y_test_pred[0]))
        mape_train = mean_absolute_percentage_error(y_train[l_names[0]],y_train_pred[0])
        mape_test = mean_absolute_percentage_error(y_test[l_names[0]],y_test_pred[0])

        corlist_train.append(cor_train[1,0])
        corlist_test.append(cor_test[1,0])
        r2list_train.append(r2_train)
        r2list_test.append(r2_test)
        rmselist_train.append(rmse_train)
        rmselist_test.append(rmse_test)        
        mapelist_train.append(mape_train)
        mapelist_test.append(mape_test)
        
        o.append(y_train[l_names[0]])
        o.append(y_train_pred[0])
        o.append(y_test[l_names[0]])
        o.append(y_test_pred[0])
        #pbar.update()           
        #imp.append(model.feature_importances_)
        #.columns = ['niche breath_pred']
        #plot_data = pd.concat([y_test,y_test_pred], axis=1)
        #sns.regplot(x='niche breath',y='niche breath_pred',data=plot_data)                                               
        #plt.show()
        
    print('cor:',np.mean(corlist_test))
    print('r2_score:',np.mean(r2list_test))
    print("RMSE:",np.mean(rmselist_test))
    print("MAPE:",np.mean(mapelist_test))
    print('-----------------------------------------')
    
    all_cor.append(np.mean(corlist_test))
    all_r2_score.append(np.mean(r2list_test))
    all_rmse.append(np.mean(rmselist_test))
    all_mape.append(np.mean(mape_test))
    all_l_names.append(l_names[0])

result=pd.DataFrame({'function':all_l_names,'cor':all_cor,'r2_score':all_r2_score,'rmse':all_rmse,'mape':all_mape})
result.to_excel('SVM.xlsx',index=False)





# %% ANN

import numpy as np
import pandas as pd
from sklearn import neural_network
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import os 

scaler = preprocessing.StandardScaler()
Features_0 = pd.read_excel('Features several correct.xlsx',sheet_name='Sheet4').iloc[:105,2:]
Labels_0 = pd.read_excel('CN cycling function.xlsx',sheet_name='Sheet1').iloc[:105,2:]
Features = pd.DataFrame(scaler.fit_transform(Features_0))
Labels =  pd.DataFrame(scaler.fit_transform(Labels_0))
Features.columns=Features_0.columns
Labels.columns=Labels_0.columns  

all_cor=[]
all_r2_score=[]
all_rmse=[]
all_mape=[]
all_l_names=[]

l_list=[8,9,1,6,10,7,11,12]

j=l_list[0]
for j in l_list: 
    x = Features.copy()
    f_names = x.columns
    y = pd.DataFrame(Labels.iloc[:,j])
    l_names=y.columns.values.tolist()
    
    data = pd.concat([x,y],axis=1)
    data.iloc[:,-1].replace(0,np.nan,inplace=True)
    data.dropna(axis=0,subset=[l_names[0]],inplace=True)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    
    '''  
    model = MLPRegressor(solver='adam', hidden_layer_sizes=(50, 50), activation='tanh', max_iter=5000)
    model=MLPRegressor()
    selector = RFE(model, n_features_to_select=4, step=1).fit(x, np.ravel(y.values))
    sup = selector.support_ 
    #feature_imp_rank_RFE = pd.concat([pd.DataFrame(x.columns,columns=['Feature']),pd.DataFrame(selector.ranking_,columns=['Importance'])],axis=1)
    colist = np.flatnonzero(sup) 
    x = x.iloc[:,colist]
    f_names = x.columns
    '''  
    data = pd.concat([x,y],axis=1)
    x = data.iloc[:,:-1]
    y = pd.DataFrame(data.iloc[:,-1])
    f_names=x.columns.values.tolist()
    l_names=y.columns.values.tolist()

    kf = KFold(n_splits=10,shuffle=True,random_state=0) 
    o=[]
    corlist_test=[]
    corlist_train=[]
    rmselist_train=[]
    rmselist_test=[]
    r2list_train=[]
    r2list_test=[]
    mapelist_train=[]
    mapelist_test=[]
    
    for train_index, test_index in kf.split(x):
        
        x_train=x.iloc[train_index,:]
        x_train.columns=f_names
        y_train=y.iloc[train_index,:]        
        x_test=x.iloc[test_index,:]
        x_test.columns=f_names
        y_test=y.iloc[test_index,:]

        model=neural_network.MLPRegressor(hidden_layer_sizes=(10),
                               activation='relu',
                               solver='adam',
                               alpha=0.0001,
                               batch_size='auto',
                               learning_rate='constant',
                               learning_rate_init=0.001,
                               power_t=0.5,max_iter=200,tol=1e-4)

        model.fit(x_train,np.ravel(y_train.values))

        y_train_pred=pd.DataFrame(model.predict(x_train).reshape(y_train.shape),index=train_index)
        y_test_pred=pd.DataFrame(model.predict(x_test).reshape(y_test.shape),index=test_index)

        cor_train=np.corrcoef(y_train[l_names[0]],y_train_pred[0])
        cor_test=np.corrcoef(y_test[l_names[0]],y_test_pred[0])        
        r2_train=r2_score(y_train[l_names[0]],y_train_pred[0])
        r2_test=r2_score(y_test[l_names[0]],y_test_pred[0])
        rmse_train=np.sqrt(mean_squared_error(y_train[l_names[0]],y_train_pred[0]))
        rmse_test=np.sqrt(mean_squared_error(y_test[l_names[0]],y_test_pred[0]))
        mape_train = mean_absolute_percentage_error(y_train[l_names[0]],y_train_pred[0])
        mape_test = mean_absolute_percentage_error(y_test[l_names[0]],y_test_pred[0])

        corlist_train.append(cor_train[1,0])
        corlist_test.append(cor_test[1,0])
        r2list_train.append(r2_train)
        r2list_test.append(r2_test)
        rmselist_train.append(rmse_train)
        rmselist_test.append(rmse_test)        
        mapelist_train.append(mape_train)
        mapelist_test.append(mape_test)
        
        o.append(y_train[l_names[0]])
        o.append(y_train_pred[0])
        o.append(y_test[l_names[0]])
        o.append(y_test_pred[0])
        #pbar.update()           
        #imp.append(model.feature_importances_)
        #.columns = ['niche breath_pred']
        #plot_data = pd.concat([y_test,y_test_pred], axis=1)
        #sns.regplot(x='niche breath',y='niche breath_pred',data=plot_data)                                               
        #plt.show()
        
    print('cor:',np.mean(corlist_test))
    print('r2_score:',np.mean(r2list_test))
    print("RMSE:",np.mean(rmselist_test))
    print("MAPE:",np.mean(mapelist_test))
    print('-----------------------------------------')
    
    all_cor.append(np.mean(corlist_test))
    all_r2_score.append(np.mean(r2list_test))
    all_rmse.append(np.mean(rmselist_test))
    all_mape.append(np.mean(mape_test))
    all_l_names.append(l_names[0])

result=pd.DataFrame({'function':all_l_names,'cor':all_cor,'r2_score':all_r2_score,'rmse':all_rmse,'mape':all_mape})
result.to_excel('SVM.xlsx',index=False)
























