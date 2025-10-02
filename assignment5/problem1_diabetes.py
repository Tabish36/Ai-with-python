import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

features = ["bmi","s5"]
lr = LinearRegression()
lr.fit(X_train[features], y_train)
score_base = r2_score(y_test, lr.predict(X_test[features]))

results = {}
for f in X.columns:
    if f not in features:
        lr.fit(X_train[features+[f]], y_train)
        results[f] = r2_score(y_test, lr.predict(X_test[features+[f]]))

best = max(results, key=results.get)
score_best = results[best]

lr.fit(X_train, y_train)
score_all = r2_score(y_test, lr.predict(X_test))

print("Problem 1: Diabetes")
print("Base model (bmi+s5) R2:", score_base)
print("Best added variable:", best, "with R2:", score_best)
print("All variables model R2:", score_all)

"""
I trained a regression using bmi and s5 first. 
After that I added each feature separately and compared the R2 scores. 
The one with the highest test R2 was chosen as the best variable to add. 
Then I also fitted the model with all the features. 
The results showed that more features can help, but the increase is not always large. 
"""
