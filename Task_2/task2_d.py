import numpy as np
import sklearn as skl
import pandas as pd
import sklearn.preprocessing
from sklearn import linear_model
from scipy import special
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

Vitals_labels = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
Test_labels = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

Vitals = ['RRate', 'ABPm', 'SpO2', 'Heartrate']
Test = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total',
         'Lactate', 'TroponinI', 'SaO2',
         'Bilirubin_direct', 'EtCO2']

#%% md

#To generate the final table. Values must be binary. cutn ass mutherfocker
#use the 12 entries of heartrate to regress to the label asked. try linear first then go poly. ignore drugs for now

## TASK 0 #######################################################################################################

#%%

df = pd.read_csv("train_features.csv")
df_train= df.groupby(['pid'],sort=False).mean()
df_train_test = df_train[Test]
df_train_vitals=df_train[Vitals]
df_final=pd.concat([df_train_test,df_train_vitals],axis=1)
#df_final = df_final.fillna(0)
#print(df_final)
df_final = df_final.notnull().astype('int')
#print(df_final)

labels = pd.read_csv("train_labels.csv")
labels_sebsis = labels[["LABEL_Sepsis"]].to_numpy()
labels_hr = labels[["LABEL_Heartrate"]].to_numpy()
labels_spo2 = labels[["LABEL_SpO2"]].to_numpy()
labels_abpm = labels[["LABEL_ABPm"]].to_numpy()
labels_rrate = labels[["LABEL_RRate"]].to_numpy()
#scaler0 = skl.preprocessing.StandardScaler()
#scaler0.fit(labels)
#labels = scaler0.transform(labels)
heartrate = df[["Heartrate"]].to_numpy()

heartrate = heartrate.reshape((18995,12))
#print(heartrate[1,:])

## this is a mess if the entire row has no value. it may be usefull later in the process tho.
#row_mean = np.nanmean(heartrate,axis=1)
#inds = np.where(np.isnan(heartrate))
#heartrate[inds] = np.take(row_mean,inds[1])
#print(heartrate[1,:])

#lets find what stupid patient doesnt have a heartrate


#cause some fuckers didnt get a measurement.
heartrate = np.nan_to_num(heartrate,nan=0)

#lets use concatenated full data set TRAIN DATA
fit_df = df_train.fillna(0).to_numpy()
fit_df = np.delete(fit_df,0,1) #remove patient ID
fit_df = np.delete(fit_df,0,1) #remove time spent in ICU
scaler1 = skl.preprocessing.StandardScaler()
scaler1.fit(fit_df)
fit_df = scaler1.transform(fit_df)
scaler2 = skl.preprocessing.StandardScaler()
scaler2.fit(heartrate)
heartrate = scaler2.transform(heartrate)
fit_df = np.concatenate((fit_df, heartrate), axis=1)
heartrate_train = heartrate

#SAME FOR TEST DATA
test_df_full = pd.read_csv("test_features.csv")
test_df = test_df_full.groupby(['pid'],sort=False).mean()
test_df = test_df.fillna(0).to_numpy()
pid_test = test_df_full[["pid"]].to_numpy()
pid_test = pd.DataFrame({'pid': pid_test[:,0]}).drop_duplicates().reset_index(drop=True)
test_df = np.delete(test_df,0,1)
test_df = np.delete(test_df,0,1)
labels_test = pd.read_csv("test_features.csv")
heartrate = test_df_full[["Heartrate"]].fillna(0).to_numpy()
heartrate = heartrate.reshape((12664,12))
test_df = scaler1.transform(test_df)
heartrate = scaler2.transform(heartrate)
test_df = np.concatenate((test_df,heartrate), axis=1)
heartrate_test = heartrate

#TESTusing localoutliers - create mask for training data
#something = np.array([10,20, 50,100,150,200])
#for item in something:
#    value = item
#    lof = skl.neighbors.LocalOutlierFactor(n_neighbors=value)  # try out different factors
#    yhat = lof.fit_predict(fit_df)
#    mask = yhat != -1
#    longmask = np.repeat(mask, 12, axis=0)
#    train_vital_means = df_train[Vitals]
#    print(np.count_nonzero(mask))

#using localoutliers - create mask for training data
lof = skl.neighbors.LocalOutlierFactor(n_neighbors= 150) #try out different factors
yhat = lof.fit_predict(fit_df)

mask = yhat != -1
longmask = np.repeat(mask,12,axis=0)


train_vital_means = df_train[Vitals]

print(np.count_nonzero(mask))

## TASK 1 #######################################################################################################

#%% using all meds is much better than just using one. however the norming doesnt work yet. it must be done for

# every single column. and scaling factors must be saved to be then applied to test data

#clf = skl.linear_model.RidgeClassifierCV(alphas=[0.01, 0.01, 0.1, 1, 10, 100, 1000, 2000,5000], cv=15, normalize=True)
clf = skl.svm.LinearSVC(dual=False, class_weight='balanced') #check out what balanced does
#clf = skl.linear_model.LogisticRegressionCV(multi_class='multinomial')

#lets do this for base excess and then see how it works
train_data = df[Test].fillna(0).to_numpy()
train_labels = labels[Test_labels].fillna(0).to_numpy()
test_data = test_df_full[Test].fillna(0).to_numpy()
predict_data = np.zeros((12664,10))

#using localoutliers
#lof = skl.neighbors.LocalOutlierFactor(n_neighbors= 100) #try out different factors
#yhat = lof.fit_predict(train_data)

#mask = yhat != -1
#train_data, test_data = train_data[mask,:], train_labels[mask]
#print(train_data.shape)

train_data = train_data[longmask,:]
train_labels = train_labels[mask,:]

for i in range(0,10):
    scaler = skl.preprocessing.StandardScaler()
    scaler.fit(train_data[:,i].reshape(-1,1))
    temp = scaler.transform(train_data[:,i].reshape(-1,1))
    train_data[:,i] = np.reshape(temp, newshape=16233*12)
    temp = scaler.transform(test_data[:,i].reshape(-1,1))
    test_data[:,i] = np.reshape(temp, newshape=12664*12)


for j in range(0,10):
    clf.fit(train_data[:,j].reshape(16233,12),train_labels[:,j])
    predict_data[:,j] = special.expit(clf.decision_function(test_data[:,j].reshape(12664,12)))
    #print(predict_data)
    #print(np.max(predict_data))

df_task1 = pd.DataFrame(data=predict_data, columns= Test_labels)

## TASK 2 #######################################################################################################

clf = skl.svm.LinearSVC(dual=False, class_weight='balanced')


X = fit_df
Y = labels_sebsis

clf.fit(X,Y)
predict_sepsis = special.expit(clf.decision_function(test_df))

df_task2 = pd.DataFrame({'LABEL_Sepsis': predict_sepsis})

## TASK 3 #######################################################################################################

#lets try and remove some outliers from the dataset: Top Res: 8.679543614001695

#%% Outlier removed

data_mean, data_std = np.mean(fit_df), np.std(fit_df)

#using isolationforest
#iso = skl.ensemble.IsolationForest(contamination=0.1)
#yhat = iso.fit_predict(fit_df)

#Remove outliers from data
fit_df, labels_hr, labels_spo2, labels_abpm, labels_rrate = fit_df[mask,:], labels_hr[mask], labels_spo2[mask], labels_abpm[mask], labels_rrate[mask]

#%%

clf = skl.linear_model.RidgeCV(alphas=[0.01, 0.01, 0.1, 1, 10, 100, 1000, 2000,5000], cv=15)

#%% md

# now let's predict some stuff :D
# poly = 18
# linearRegression = 9
# never fucking use linear svr. takes fucking forever

#%%

#predict heart rate:
clf.fit(fit_df,labels_hr)
predict_train = clf.predict(fit_df)
print(np.sqrt(np.mean((predict_train-labels_hr)**2)))
predict_heartrate = clf.predict(test_df)

#predict abpm:
clf.fit(fit_df,labels_abpm)
predict_abpm = clf.predict(test_df)

#predict rrate:
clf.fit(fit_df,labels_rrate)
predict_rrate = clf.predict(test_df)

#predict spo2:
clf.fit(fit_df,labels_spo2)
predict_sp02 = clf.predict(test_df)

#%% md

#For some reason linear regression works the best. but i still dont think it's acurate enough

## TASK 4 #######################################################################################################

df_task3 = pd.DataFrame({'LABEL_RRate': predict_rrate[:,0], 'LABEL_ABPm': predict_abpm[:,0], 'LABEL_SpO2': predict_sp02[:,0],'LABEL_Heartrate': predict_heartrate[:,0]})
print(df_task3.head)

#print(fit_df)
#print(labels-predict_train)
#print(predict_train)
#print(predict_heartrate)

#%% md

## Final Concatenate

#%%

final_frickin_df = pd.concat([pid_test ,df_task1, df_task2, df_task3], axis=1)
compression_opts = dict(method='zip', archive_name='result.csv')
final_frickin_df.to_csv('result.zip', index=False, compression=compression_opts)
