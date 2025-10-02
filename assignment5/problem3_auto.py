import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

auto = pd.read_csv("Auto.csv", na_values=["?","NA"]).dropna()

y = auto["mpg"]
X = auto.drop(columns=["mpg","name","origin"])
X = X.apply(pd.to_numeric, errors="coerce").dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

alpha_list = np.logspace(-3,3,25)
ridge_scores = []
lasso_scores = []

for a in alpha_list:
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=a))])
    lasso = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=a, max_iter=10000))])
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    ridge_scores.append(ridge.score(X_test, y_test))
    lasso_scores.append(lasso.score(X_test, y_test))

plt.plot(alpha_list, ridge_scores, label="Ridge R2")
plt.plot(alpha_list, lasso_scores, label="Lasso R2")
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("R2 score")
plt.legend()
plt.show()

best_ridge = alpha_list[np.argmax(ridge_scores)]
best_lasso = alpha_list[np.argmax(lasso_scores)]

print("Problem 3: Car mpg")
print("Best Ridge alpha:", best_ridge, "R2:", max(ridge_scores))
print("Best Lasso alpha:", best_lasso, "R2:", max(lasso_scores))

"""
In this task I predicted mpg using the auto dataset. 
I trained Ridge and Lasso regressions with different alpha values and checked their R2 on the test set. 
The R2 scores were plotted against alpha to visualize performance. 
The alpha that gave the best score was considered optimal. 
Ridge stayed more consistent, while Lasso dropped when alpha got large. 
"""
