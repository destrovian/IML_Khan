"""
Created on Thu Apr 15 08:40:02 2021

@author: alicefritzsche
"""
""" ###load packages """
import numpy as np
import pandas as pd
import math



from sklearn.linear_model import LinearRegression

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import scale
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from scipy.special import expit

"""### Load data sets """
df_train = pd.read_csv('/Users/alicefritzsche/Desktop/Task2/task2_k49am2lqi/train_features.csv', header=0)
data = df_train.to_numpy()
X_train = data[:,2:] #cut off 'pid' and 'Time' as they are not features
#df_train.head(n=10) #show first patient

df_labels = pd.read_csv('/Users/alicefritzsche/Desktop/Task2/task2_k49am2lqi/train_labels.csv', header=0)
data = df_labels.to_numpy()
y_train = data[:,1:] #cut off pid
#df_labels.head(n=10) #show first patient

df_test = pd.read_csv('/Users/alicefritzsche/Desktop/Task2/task2_k49am2lqi/test_features.csv', header=0)
data = df_test.to_numpy()
X_test = data[:,2:] #cut off 'pid' and 'Time' as they are not features
#df_test.head(n=10) #show first patient


#Features are the same for the train and test data
features = np.size(X_train, 1) 
feature_names = list(df_train.columns[2:features+2]) #cut of 'pid' and 'Time'
#All the labels from the training data:
labels = np.size(y_train,1) 
label_names = list(df_labels.columns[1:labels+1]) #cut off 'pid' 


"""### Data processing """
#X_train_mean is the matrix where for each feature the mean value is calculated. 
#The entry is NaN if all the values of a feature is NaN.
rows_train = np.size(X_train, 0)
patients_train = round(rows_train/12)
X_train_mean = np.zeros((patients_train,features))

for i in range(0,rows_train,12):
    #Obtain mean of columns
    X_train_mean[round(i/12),:] = np.nanmean(X_train[i:i+12,:], axis=0)

df = pd.DataFrame(X_train_mean, columns = feature_names)
#print(df.head())
#df.shape
#calculate mean value of each feature for all the patients and fill up the NaNs with this value. Overfitting?
X_train_no_nan = X_train_mean

for i in range(features):
    #Obtain mean of columns
    if np.isnan(X_train_mean[:,i]).all(): #if a feature has no values for all patients
      print("drop feature") #drop feature. In this dataset this does not happen
    else:
      mean = np.nanmean(X_train_mean[:,i], axis=0) #mean value of ith feature
      nan_indices = np.argwhere(np.isnan(X_train_mean[:,i])) #all indices of one feature where the value is NaN
      for j in nan_indices:
        X_train_no_nan[j,i] = mean #replace NaN with mean value

df= pd.DataFrame(X_train_no_nan, columns = feature_names)
X_train_no_nan = df.values
#print(df.head(n=10))

#Standardize train set
scaler = StandardScaler()
scaler.fit(X_train_no_nan)
X_train_no_nan = scaler.transform(X_train_no_nan)


#same data processing for test data:
rows_test = np.size(X_test, 0)
patients_test = round(rows_test/12)
X_test_mean = np.zeros((patients_test,features))

for i in range(0,rows_test,12):
    #Obtain mean of columns
    X_test_mean[round(i/12),:] = np.nanmean(X_test[i:i+12,:], axis=0)

df= pd.DataFrame(X_test_mean, columns = feature_names)
#df.head(n=10)
X_test_no_nan = X_test_mean

for i in range(features):
    #Obtain mean of columns
    if np.isnan(X_test_mean[:,i]).all(): #if a feature has no values for all patients
      print("drop feature") #drop feature. In this dataset this does not happen
    else:
      mean = np.nanmean(X_test_mean[:,i], axis=0) #mean value of ith feature
      nan_indices = np.argwhere(np.isnan(X_test_mean[:,i])) #all indices of one feature where the value is NaN
      for j in nan_indices:
        X_test_no_nan[j,i] = mean #replace NaN with mean value

df= pd.DataFrame(X_test_no_nan, columns = feature_names)
#df.head(n=10)
#Standardize test set
scaler = StandardScaler()
scaler.fit(X_test_no_nan)
X_test_no_nan = scaler.transform(X_test_no_nan)

    
"""### drop outliers"""

labels_sepsis = df_labels[["LABEL_Sepsis"]].to_numpy()
labels_hr = df_labels[["LABEL_Heartrate"]].to_numpy()
labels_spo2 = df_labels[["LABEL_SpO2"]].to_numpy()
labels_abpm = df_labels[["LABEL_ABPm"]].to_numpy()
labels_rrate = df_labels[["LABEL_RRate"]].to_numpy()




# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train_no_nan)


# select all rows that are not outliers
mask = yhat != -1

X_train_no_nan, labels_hr, labels_spo2, labels_abpm, labels_rrate, labels_sepsis = \
    X_train_no_nan[mask,:], labels_hr[mask], labels_spo2[mask], labels_abpm[mask], labels_rrate[mask], labels_sepsis[mask]







"""### Sub-Task 1 """
#extract labels we need
n_labels = 10
y_train_new = y_train[:,0:10] #the first 10 labels
#df= pd.DataFrame(y_train_new, columns = label_names[0:10])
#print(df.head(n=10))
#print(y_train_new)

#df= pd.DataFrame(X_train_no_nan, columns = feature_names)
#df.head(n=10)

clf = LinearSVC(dual=False, class_weight='balanced')
X_predict = np.zeros((patients_test, n_labels))
for l in range(0,10):
  #clf_svm = SVC(random_state=42) #seed
  clf.fit(X_train_no_nan, y_train_new[:,l]) #fit the SVM for lth label
  X_predict[:,l] = expit(clf.decision_function(X_test_no_nan))
  

#prediction into dataframe
df_task1 = pd.DataFrame(data = X_predict, columns = label_names[0:10])
#print(df_task1)



  
"""### Sub_Task 2 """


clf =  LinearSVC(dual=False, class_weight='balanced')
  
y_Sepsis = df_labels[["LABEL_Sepsis"]].to_numpy() #extract label sepsis

clf.fit(X_train_no_nan,y_Sepsis)
#get sigmoid fct with expit and try to predict with decision_function
predict_Sepsis = expit(clf.decision_function(X_test_no_nan)) #sigmoid fct


#prediction into dataframe
df_task2 = pd.DataFrame({'LABEL_Sepsis': predict_Sepsis})
#print(df_task2)


"""### Sub_Task 3 """

clf = RidgeCV(cv=20)

#predict RRate
y_RRate = df_labels[["LABEL_RRate"]].to_numpy()
clf.fit(X_train_no_nan, y_RRate)
predict_RRate = clf.predict(X_test_no_nan)

#predict ABPm
y_ABPm = df_labels[["LABEL_ABPm"]].to_numpy()
clf.fit(X_train_no_nan, y_ABPm)
predict_ABPm = clf.predict(X_test_no_nan)

#predict SpO2
y_Sp02 = df_labels[["LABEL_SpO2"]].to_numpy()
clf.fit(X_train_no_nan, y_Sp02)
predict_Sp02 = clf.predict(X_test_no_nan)

#predict Heartrate
y__Heartrate = df_labels[["LABEL_Heartrate"]].to_numpy()
clf.fit(X_train_no_nan, y__Heartrate)
predict_Heartrate = clf.predict(X_test_no_nan)

#prediction into dataframe
df_task3 = pd.DataFrame({'LABEL_RRate': predict_RRate[:,0], 'LABEL_ABPm': predict_ABPm[:,0], 'LABEL_SpO2': predict_Sp02[:,0],'LABEL_Heartrate': predict_Heartrate[:,0]})
#print(df_task3.head)




#final submission 
pid_test = df_test[["pid"]].to_numpy()
pid_test = pd.DataFrame({'pid': pid_test[:,0]}).drop_duplicates().reset_index(drop=True)
final_df = pd.concat([pid_test ,df_task1, df_task2, df_task3], axis=1)
print(final_df)
final_df.to_csv('prediction.zip', index=False, float_format='%.3f', compression='zip')
    

    


