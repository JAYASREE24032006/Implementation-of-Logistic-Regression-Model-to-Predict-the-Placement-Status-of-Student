# EX-5 : IMPLEMENTATION OF LOGISTIC REGRESSION MODEL TO PREDICT THE PLACEMENT STATUS OF STUDENT

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1.Import the required packages and print the present data
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices. 

## PROGRAM:
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data1=data.copy()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Matrix",accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix",confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print("\nclassification_report")
print(classification_report1)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/be0ab178-dfea-4032-b104-d671513d84f9)


## RESULT:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
