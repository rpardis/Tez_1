# Program Name: Test_Rekordz4.py
from sklearn.model_selection import\
    cross_val_score,\
    RepeatedStratifiedKFold
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import time
from datetime import datetime
import winsound
#%% Dialog
def zman(tn,t0):
    print('__________________________________')
    T03=tn-t0
    H03=int(T03/3600)
    M03=int((T03-H03*3600)/60)
    S03=int(T03-H03*3600-M03*60)
    print('Elapsed %i H %i M %i s'%(H03,M03,S03))
    print(datetime.now().strftime("%D & %H:%M:%S"))
isdti=int(input('\nUse DTIs?  >>>  '))
if isdti==1:
    hh_X=pd.read_excel(open('Mjazat_STD.xls', 'rb'), sheet_name='X_D', header=0)
else:
    hh_X=pd.read_excel(open('Mjazat_STD.xls', 'rb'), sheet_name='X_R', header=0)

data1_X=pd.DataFrame.to_numpy(hh_X)
data_X=np.array(data1_X)
[r_X,s_X]=np.shape(data_X)
X0_X = data_X[:,0:s_X-1]
X=X0_X.astype('float')
Y0_X = data_X[:,s_X-1]
y=Y0_X.astype('int')
# NR=int(input('\n\nCV n_repeats  (df=999)>>>  '))
#%%
a=np.zeros([12])
aa=np.array([
[6.34305783e-01,8.50927836e-01,1.03277360e-01,4.71614568e-01,2.36607837e-01,5.64430977e-01,1.66484723e+01,8.19128915e-01,1.29173641e+02,6.45513957e-01,4.45980245e-01,7.83060329e-01]
])

[s1,s2]=np.shape(aa)
Accuz=np.zeros(s1)
for jj in range(s1):
    T00=time.time()
    a[:]=aa[jj,:]
    Test_params = {'learning_rate': a[0], 'scale_pos_weight':a[1] , 'colsample_bylevel':a[2], 'colsample_bytree': a[3], 'gamma':a[4] , 'max_delta_step': a[5], 'max_depth': int(a[6]), 'min_child_weight': a[7], 'n_estimators': int(a[8]), 'reg_alpha': a[9], 'reg_lambda': a[10], 'subsample': a[11]}
    Test_Model=XGBClassifier(use_label_encoder=False,\
    colsample_bylevel=Test_params['colsample_bylevel'],\
    colsample_bytree=Test_params['colsample_bytree'],\
    gamma=Test_params['gamma'],\
    learning_rate=Test_params['learning_rate'],\
    max_delta_step=Test_params['max_delta_step'],\
    max_depth=Test_params['max_depth'],\
    min_child_weight=Test_params['min_child_weight'],\
    n_estimators=Test_params['n_estimators'],\
    reg_alpha=Test_params['reg_alpha'],\
    reg_lambda=Test_params['reg_lambda'],\
    subsample=Test_params['subsample'],\
    scale_pos_weight=Test_params['scale_pos_weight'],\
    eval_metric='logloss',objective ='binary:logistic')
    
    cv0 = RepeatedStratifiedKFold(n_splits=10, n_repeats=33, random_state=1)
    #%% Output measures
    Accuz[jj] = cross_val_score(Test_Model, X, y, cv=cv0,scoring='accuracy').mean()
    zman(time.time(),T00)
    print('\t%i: Accuracy = %.3f' %(jj,Accuz[jj])) 
    winsound.Beep(5000, 500)
    time.sleep(13)
am=np.argmax(Accuz)
a[:]=aa[am,:]
Test_params = {'learning_rate': a[0], 'scale_pos_weight':a[1] , 'colsample_bylevel':a[2], 'colsample_bytree': a[3], 'gamma':a[4] , 'max_delta_step': a[5], 'max_depth': int(a[6]), 'min_child_weight': a[7], 'n_estimators': int(a[8]), 'reg_alpha': a[9], 'reg_lambda': a[10], 'subsample': a[11]}
winsound.Beep(600, 5000)
Goon=int(input('\n\nGo on accurate accuracy? (df=1) >>>  '))
if Goon==1:
    Test_Model=XGBClassifier(use_label_encoder=False,\
    colsample_bylevel=Test_params['colsample_bylevel'],\
    colsample_bytree=Test_params['colsample_bytree'],\
    gamma=Test_params['gamma'],\
    learning_rate=Test_params['learning_rate'],\
    max_delta_step=Test_params['max_delta_step'],\
    max_depth=Test_params['max_depth'],\
    min_child_weight=Test_params['min_child_weight'],\
    n_estimators=Test_params['n_estimators'],\
    reg_alpha=Test_params['reg_alpha'],\
    reg_lambda=Test_params['reg_lambda'],\
    subsample=Test_params['subsample'],\
    scale_pos_weight=Test_params['scale_pos_weight'],\
    eval_metric='logloss',objective ='binary:logistic')
    cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    acmac = cross_val_score(Test_Model, X, y, cv=cv1,scoring='accuracy').mean()
    print('\t\t%i>>> Best accuracy = %.3f' %(am,acmac))
    print(Test_params)