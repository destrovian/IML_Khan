import numpy as np
import sklearn as skl
import pandas as pd
import sklearn.preprocessing
from sklearn import linear_model
from scipy import special
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

np.random.seed(69) #fixing seed for reproducability     

Vitals_labels = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
Test_labels = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

Vitals = ['RRate', 'ABPm', 'SpO2', 'Heartrate']
Test = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total',
         'Lactate', 'TroponinI', 'SaO2',
         'Bilirubin_direct', 'EtCO2']

#%% md

#To generate the final table. Values must be binary.
#use the 12 entries of heartrate to regress to the label asked. try linear first then go poly. ignore drugs for now

## TASK 0 #######################################################################################################

#%%

df = pd.read_csv("train_features.csv")
df_train= df.groupby(['pid'],sort=False).mean()
df_train_test = df_train[Test]
df_train_vitals=df_train[Vitals]
df_final=pd.concat([df_train_test,df_train_vitals],axis=1)
df_final = df_final.notnull().astype('int')

labels = pd.read_csv("train_labels.csv")
labels_sebsis = labels[["LABEL_Sepsis"]].to_numpy()
labels_hr = labels[["LABEL_Heartrate"]].to_numpy()
labels_spo2 = labels[["LABEL_SpO2"]].to_numpy()
labels_abpm = labels[["LABEL_ABPm"]].to_numpy()
labels_rrate = labels[["LABEL_RRate"]].to_numpy()
heartrate = df[["Heartrate"]].to_numpy()

heartrate = heartrate.reshape((18995,12))

#lets find what patient doesnt have a heartrate
#cause some people didnt get a measurement.
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

print("Task 1 finished with no Errors")

#SAME FOR TEST DATA
test_df_full = pd.read_csv("test_features.csv")
test_df = test_df_full.groupby(['pid'],sort=False).mean()
test_df = test_df.fillna(0).to_numpy()
pid_test = test_df_full[["pid"]].to_numpy()
pid_test = pd.DataFrame({'pid': pid_test[:,0]}).drop_duplicates().reset_index(drop=True)
test_df = np.delete(test_df,0,1) #remove patient ID
test_df = np.delete(test_df,0,1) #remove time spent in ICU
labels_test = pd.read_csv("test_features.csv")
heartrate = test_df_full[["Heartrate"]].fillna(0).to_numpy()
heartrate = heartrate.reshape((12664,12))
test_df = scaler1.transform(test_df)
heartrate = scaler2.transform(heartrate)
test_df = np.concatenate((test_df,heartrate), axis=1)
heartrate_test = heartrate

#using localoutliers - create mask for training data
lof = skl.neighbors.LocalOutlierFactor(n_neighbors= 150) #try out different factors
yhat = lof.fit_predict(fit_df)

mask = yhat != -1
longmask = np.repeat(mask,12,axis=0)

train_vital_means = df_train[Vitals]

#Remove outliers from data
fit_df, labels_hr, labels_spo2, labels_abpm, labels_rrate, labels_sebsis = \
    fit_df[mask,:], labels_hr[mask], labels_spo2[mask], labels_abpm[mask], labels_rrate[mask], labels_sebsis[mask]

#expand fit_df with the vitals asked for in the exercise
append_rrate = df[["RRate"]].to_numpy()
append_rrate = np.nan_to_num(append_rrate,nan=0)
append_rrate = append_rrate[longmask,:].reshape(np.count_nonzero(mask),12)
append_abpm = df[["ABPm"]].to_numpy()
append_abpm = np.nan_to_num(append_abpm,nan=0)
append_abpm = append_abpm[longmask,:].reshape(np.count_nonzero(mask),12)
append_sp02 = df[["SpO2"]].to_numpy()
append_sp02 = np.nan_to_num(append_sp02,nan=0)
append_sp02 = append_sp02[longmask,:].reshape(np.count_nonzero(mask),12)

scaler_rrate = skl.preprocessing.StandardScaler()
scaler_rrate.fit(append_rrate)
append_rrate = scaler_rrate.transform(append_rrate)

scaler_abpm = skl.preprocessing.StandardScaler()
scaler_abpm.fit(append_abpm)
append_abpm = scaler_abpm.transform(append_abpm)

scaler_sp02 = skl.preprocessing.StandardScaler()
scaler_sp02.fit(append_sp02)
append_sp02 = scaler_sp02.transform(append_sp02)

fit_df = np.concatenate((fit_df, append_rrate, append_abpm, append_sp02), axis=1)

#do the same stuff with the test_data
append_rrate = test_df_full[["RRate"]].to_numpy()
append_rrate = np.nan_to_num(append_rrate,nan=0)
append_rrate = append_rrate.reshape(12664,12)
append_abpm = test_df_full[["ABPm"]].to_numpy()
append_abpm = np.nan_to_num(append_abpm,nan=0)
append_abpm = append_abpm.reshape(12664,12)
append_sp02 = test_df_full[["SpO2"]].to_numpy()
append_sp02 = np.nan_to_num(append_sp02,nan=0)
append_sp02 = append_sp02.reshape(12664,12)

append_rrate = scaler_rrate.transform(append_rrate)
append_abpm = scaler_abpm.transform(append_abpm)
append_sp02 = scaler_sp02.transform(append_sp02)



test_df = np.concatenate((test_df, append_rrate, append_abpm, append_sp02), axis=1)

## TASK 1 #######################################################################################################

#%% using all meds is much better than just using one. however the norming doesnt work yet. it must be done for

# every single column. and scaling factors must be saved to be then applied to test data

#clf = skl.linear_model.RidgeClassifierCV(alphas=[0.01, 0.01, 0.1, 1, 10, 100, 1000, 2000,5000], cv=15, normalize=True)
clf = skl.svm.LinearSVC(dual=False, class_weight='balanced')

#lets do this for base excess and then see how it works
train_data = df[Test].fillna(0).to_numpy()
train_labels = labels[Test_labels].fillna(0).to_numpy()
test_data = test_df_full[Test].fillna(0).to_numpy()
predict_data = np.zeros((12664,10))

#using localoutliers

train_labels = train_labels[mask,:]

for j in range(0,10):
    clf.fit(fit_df,train_labels[:,j])
    predict_data[:,j] = special.expit(clf.decision_function(test_df))

df_task1 = pd.DataFrame(data=predict_data, columns= Test_labels)
print("Task 1 finished with no Errors")

## TASK 2 #######################################################################################################

clf = skl.svm.LinearSVC(dual=False, class_weight='balanced')


X = fit_df
Y = labels_sebsis.ravel()

clf.fit(X,Y)
predict_sepsis = special.expit(clf.decision_function(test_df))

df_task2 = pd.DataFrame({'LABEL_Sepsis': predict_sepsis})
print("Task 2 finished with no Errors")

## TASK 3 #######################################################################################################

#lets try and remove some outliers from the dataset: Top Res: 8.679543614001695

#%% Outlier removed

data_mean, data_std = np.mean(fit_df), np.std(fit_df)

#%%

clf = skl.linear_model.RidgeCV(alphas=[0.01, 0.01, 0.1, 1, 10, 100, 1000, 2000,5000], cv=15)

#try expanding the dataset with the measurements of the vitals and fit again

#predict heart rate:
clf.fit(fit_df,labels_hr)
predict_train = clf.predict(fit_df)
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

#For some reason linear regression works the best. but i still dont think it's acurate enough
print("Task 3 finished with no Errors")

## TASK 4 #######################################################################################################

df_task3 = pd.DataFrame({'LABEL_RRate': predict_rrate[:,0], 'LABEL_ABPm': predict_abpm[:,0], 'LABEL_SpO2': predict_sp02[:,0],'LABEL_Heartrate': predict_heartrate[:,0]})

#%% md

## Final Concatenate

#%%

final_frickin_df = pd.concat([pid_test ,df_task1, df_task2, df_task3], axis=1)
final_frickin_df = final_frickin_df.round(8)
compression_opts = dict(method='zip', archive_name='result.csv')
final_frickin_df.to_csv('result.zip', index=False, compression=compression_opts)

print("Successfully completed all tasks.")
