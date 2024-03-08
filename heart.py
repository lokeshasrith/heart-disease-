# Importing required tools.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Importing Data
df = pd.read_csv('/content/heart-disease.csv')

# Data Exploration
df.head()
df.tail()
df.info()
df.isna().sum()
df.describe()

# Analysing the 'target' variable
df["target"].describe()
df["target"].unique()

# Checking correlation between columns
print(df.corr()["target"].abs().sort_values(ascending=False))

# Exploratory Data Analysis (EDA)
# ... (continue with the EDA)

# Feature Engineering
categorical_val = []
continous_val = []

for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

categorical_val.remove('target')
dfs = pd.get_dummies(df, columns=categorical_val)
sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dfs[col_to_scale] = sc.fit_transform(dfs[col_to_scale])

# Model Fitting
X = dfs.drop("target", axis=1)
y = dfs["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Logistic Regression
lr = LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                        fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                        max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                        random_state=1234, solver='lbfgs', tol=0.0001, verbose=0,
                        warm_start=False)
model1 = lr.fit(X_train, y_train)
prediction1 = model1.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, prediction1)
sns.heatmap(cm, annot=True, cmap='winter', linewidths=0.3, linecolor='black', annot_kws={"size": 20})
TP = cm[0][0]
TN = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print('Testing Accuracy for Logistic Regression:', (TP + TN) / (TP + TN + FN + FP))
print('Testing Sensitivity for Logistic Regression:', (TP / (TP + FN)))
print('Testing Specificity for Logistic Regression:', (TN / (TN + FP)))
print('Testing Precision for Logistic Regression:', (TP / (TP + FP)))
