# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the 
   required module from sklearn.
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Yazhini G
RegisterNumber: 212222220060

import pandas as pd
data=pd.read_csv("dataset/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evalution","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
## Output:

## Data Head:
![279483572-48930891-5bec-4e6e-93d2-fb25bc91af62](https://github.com/Yazhini-G/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120244201/0c330578-f471-419d-a7d5-14fdfa625602)

## Information:
![279483550-c106ec2e-8339-4e11-86f5-bf22a5a34ece](https://github.com/Yazhini-G/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120244201/bcd81662-cf1e-4b3f-a4d5-92dc27ea7143)

## Null dataset:
![279483520-7087c1e7-ab1e-432b-be43-41d666396fea](https://github.com/Yazhini-G/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120244201/b695b1bc-fe6f-4684-af98-de46a0b30aaa)

## Value_counys:
![279483489-fdbcfdee-c460-4cf3-8daf-48c22d96ec96](https://github.com/Yazhini-G/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120244201/39bf458e-0697-493a-a5b4-1347f47eeace)

## Data Type Convertion:
![279483466-d6d6383d-d9e8-4293-942f-06d3fc057911](https://github.com/Yazhini-G/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120244201/33ff816b-f418-442a-b69c-05deb62bfa58)

## Data Info:
![279483438-935b78bd-5e9b-4a47-9211-3f5a9a8de033](https://github.com/Yazhini-G/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120244201/c72ced00-745a-490b-8bea-9e738ab6ace3)

## Accuracy:
![279483410-35f6b2b0-3be3-4b9d-9dc8-9ad11115ff9d](https://github.com/Yazhini-G/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120244201/b289385a-ede4-4c6c-9261-e8c306d19d75)

## Data Prediction:
![279483393-2f27d27c-cf7e-4d35-8552-9c8f43cb7ae9](https://github.com/Yazhini-G/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120244201/ae2391b5-16d6-4a60-9366-18dff8b30d43)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
