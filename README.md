# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```

<img width="1753" height="747" alt="Screenshot 2025-09-29 111202" src="https://github.com/user-attachments/assets/64ba3042-0851-488b-b760-7aacc85b2fd1" />

```
data.isnull().sum()
```
<img width="420" height="629" alt="Screenshot 2025-09-29 111227" src="https://github.com/user-attachments/assets/72c3691b-de58-4e8e-8ecb-6ba511c413bc" />

```
missing=data[data.isnull().any(axis=1)]
missing
```

<img width="1724" height="546" alt="Screenshot 2025-09-29 111254" src="https://github.com/user-attachments/assets/43e60056-d916-48d8-bd9b-4391755ee483" />

```
data2=data.dropna(axis=0)
data2
```

<img width="1730" height="736" alt="Screenshot 2025-09-29 111320" src="https://github.com/user-attachments/assets/bd0a83e6-0799-477e-a0fc-bed5ba368aa3" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="585" height="288" alt="Screenshot 2025-09-29 111346" src="https://github.com/user-attachments/assets/f3d6cc2a-9f3c-433f-a703-315d6ddab740" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="524" height="518" alt="Screenshot 2025-09-29 111408" src="https://github.com/user-attachments/assets/9a0df24c-b820-475e-862d-845ef15e4c04" />

```
data2
```

<img width="1755" height="533" alt="Screenshot 2025-09-29 111434" src="https://github.com/user-attachments/assets/48c6be8c-e7ad-46d4-ad08-8d36e59480c5" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1784" height="576" alt="Screenshot 2025-09-29 111458" src="https://github.com/user-attachments/assets/c33d221e-982d-4406-95a5-e4576aa86b52" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1563" height="56" alt="Screenshot 2025-09-29 111519" src="https://github.com/user-attachments/assets/fdcaf35c-2376-4ee0-825c-8b4dead579ff" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
<img width="1803" height="53" alt="Screenshot 2025-09-29 111546" src="https://github.com/user-attachments/assets/bc89d69a-1dcc-44f1-87d0-780f4cc2ac41" />

```
y=new_data['SalStat'].values
print(y)
```

<img width="293" height="46" alt="Screenshot 2025-09-29 111607" src="https://github.com/user-attachments/assets/56905dd7-e469-4eb8-a1cb-1949739e3f15" />

```
x=new_data[features].values
print(x)
```

<img width="515" height="179" alt="Screenshot 2025-09-29 111627" src="https://github.com/user-attachments/assets/a94f32c8-934f-49b0-9c5f-374f3a7bf95c" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="451" height="98" alt="Screenshot 2025-09-29 111651" src="https://github.com/user-attachments/assets/8fe794c2-8273-4a70-8577-0b7c27fc6d98" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="254" height="66" alt="Screenshot 2025-09-29 111709" src="https://github.com/user-attachments/assets/65605814-7fed-413c-b774-d2d69ea4e92c" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="315" height="39" alt="Screenshot 2025-09-29 111732" src="https://github.com/user-attachments/assets/305ce053-37da-45dd-ac8b-674e30689f29" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="346" height="47" alt="Screenshot 2025-09-29 111755" src="https://github.com/user-attachments/assets/0a85a70a-fb0f-4af9-bc36-f35dfed69144" />

```
data.shape
```

<img width="202" height="32" alt="Screenshot 2025-09-29 111813" src="https://github.com/user-attachments/assets/c2c5a9b6-2888-4b93-81fc-1a2ded5590ce" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="391" height="86" alt="Screenshot 2025-09-29 111838" src="https://github.com/user-attachments/assets/30d89813-290d-4863-988a-0991f1694aab" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="725" height="281" alt="Screenshot 2025-09-29 111900" src="https://github.com/user-attachments/assets/788b4a67-4c54-4f16-bed7-186ea86259a9" />

```
tips.time.unique()
```

<img width="478" height="62" alt="Screenshot 2025-09-29 111918" src="https://github.com/user-attachments/assets/079ed519-2ba5-4a7c-86bc-14dae9f2f33c" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="381" height="120" alt="Screenshot 2025-09-29 111936" src="https://github.com/user-attachments/assets/3ed0e6f8-2a17-4d5b-9e91-338b9c9507c7" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="474" height="77" alt="Screenshot 2025-09-29 111951" src="https://github.com/user-attachments/assets/2e0f2649-b1dd-48d5-8eb6-5a0883774156" />

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.

