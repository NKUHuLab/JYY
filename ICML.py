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
























