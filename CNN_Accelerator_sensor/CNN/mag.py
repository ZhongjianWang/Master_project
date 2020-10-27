import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset and preprocessing 
df = pd.read_csv('Magdata.csv')
df.columns = ['timestamp', 'status', 'x', 'y', 'z']
df['events'] = df['status'].apply(lambda x: 0 if 'pan' in x.lower() else 1)
X = df['x'].values
y = df['events'].values

XR = np.arange(int(len(X)/90))
YR = np.arange(int(len(X)/90))

start = end = 0
for i in range(0, int(len(X)/90)):
    start = end
    end   = start + 90
    result = 0
    for a in range(start, end):
        #print("%.6f"%X[i])

        if (X[end-1]-X[a])>=0:
            result = result + 1
        else:
            result = result - 1
    XR[i] = result
print("X_data:")   
print(XR)


start = end = 0
for i in range(0, int(len(y)/90)):
    start = end
    end   = start + 90
    result = 0
    for a in range(start, end):
        #print("%.6f"%X[i])
        if y[a] == 0:
            result = result + 1
        else:
            result = result - 1
    if result > 0 :
        YR[i] = 0
    else:
        YR[i] = 1
print("y_data:")   
print(YR)


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XR, YR, test_size = 0.25, random_state = 0)
X_train = X_train.reshape(-1, 1)
print("X_train_reshaped:")
print(X_train)   
y_train = y_train.reshape(-1, 1)
print("y_train_reshaped:")
print(y_train)   
X_test = X_test.reshape(-1, 1)
print("X_test_reshaped:")
print(X_test) 

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print("X_train_scaled:")
print(X_train) 
X_test = sc.transform(X_test)
print("X_test_sclaed:")
print(X_test) 
    
#Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
    
import joblib
joblib.dump(classifier, 'finalized_model.pkl')
joblib.dump(sc,'finalized_scaler.pkl')

loaded_model = joblib.load('finalized_model.pkl')
#Predicting the Test set results  
y_pred = loaded_model.predict(X_test)
print("y_pred:")
print(y_pred)

(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))    

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


