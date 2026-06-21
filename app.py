import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Placement_data.csv")

print("Columns:")
print(data.columns)

for c in data.columns:
    if data[c].dtype == "object":
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c])

x = data.drop(["status", "salary"], axis=1)
y = data["status"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

m = RandomForestClassifier(random_state=42)
m.fit(x_train, y_train)

p = m.predict(x_test)

print("\nAccuracy =", accuracy_score(y_test, p))

print("\nActual Values:")
print(list(y_test[:10]))

print("\nPredicted Values:")
print(list(p[:10]))

from sklearn.metrics import confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, p))

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, p))

for i, j in zip(x.columns, m.feature_importances_):
    print(i, round(j, 3))

import matplotlib.pyplot as plt

plt.bar(x.columns, m.feature_importances_)
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.show()