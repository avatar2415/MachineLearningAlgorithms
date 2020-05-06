import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
#print(data)

#-------------------feature names-----------------------------------------------------
feature=data['feature_names']
#print(feature)

#-------------------convert the dataset into dataframes-------------------------------
data=pd.DataFrame(np.c_[data['data'],data['target']],columns=np.append(data['feature_names'],['target']))
#print(data)

#-------------------seperate the features and labels----------------------------------
X=data.iloc[:,:30]
y=data.iloc[:,30]
#print(X,y)

#------------------split the train and the test data-----------------------------------
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

#------------------train the model-----------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=25)
train=knn.fit(X_train,y_train)

#-----------------prediction on the test data-------------------------------------------
prediction=knn.predict(X_test)
prediction_array_into_dataframes=pd.DataFrame(data=prediction)
#print(prediction_array_into_dataframes)

#----------------score----------------------------------------------------------------
score=train.score(X_test,y_test)
print(score)