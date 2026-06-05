import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#local path
df = pd.read_parquet("C:\Python\PyEx\employees_raw.parquet")

#Viya path
#df = pd.read_parquet("/data/home/shared/parquet/employees_raw.parquet")


subset = df[["salary", "satisfaction_score", "project_completion_rate"]]

y = df[["left_company"]]
x = subset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, np.ravel(y_train))

y_proba = model.predict_proba(X_test)[:,1]

preds = y_proba.round(2) 

print(preds[:15])
