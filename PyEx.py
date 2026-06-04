import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_parquet("C:\Python\PyEx\employees_raw.parquet")

#print("Hej")

#print(df.head(15))

subset = df[["salary", "satisfaction_score", "project_completion_rate"]]

#print(subset.head(15))

y = df[["left_company"]]
x = subset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, np.ravel(y_train))

preds = model.predict(X_test)

print(preds[:15])
