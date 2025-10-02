import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("50_Startups.csv")

print("Problem 2: Profit Prediction")
print("Columns are:", data.columns.tolist())

print("Checking correlations:")
print(data.corr(numeric_only=True))

y = data["Profit"]
X = pd.get_dummies(data.drop(columns=["Profit"]), drop_first=True)

chosen_vars = ["R&D Spend","Marketing Spend"]
print("Chosen variables for model:", chosen_vars)

for col in chosen_vars:
    plt.scatter(data[col], y)
    plt.xlabel(col)
    plt.ylabel("Profit")
    plt.title(col + " vs Profit")
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X[chosen_vars], y, test_size=0.2, random_state=21)
reg = LinearRegression().fit(X_train, y_train)

train_score = r2_score(y_train, reg.predict(X_train))
test_score = r2_score(y_test, reg.predict(X_test))

rmse_train = mean_squared_error(y_train, reg.predict(X_train), squared=False)
rmse_test = mean_squared_error(y_test, reg.predict(X_test), squared=False)

print("Train R2:", train_score)
print("Test R2:", test_score)
print("Train RMSE:", rmse_train)
print("Test RMSE:", rmse_test)

"""
I used the startup dataset to find what affects profit. 
The correlation test showed R&D spend and marketing spend are most related. 
I plotted them with profit to confirm they look linear. 
Then I trained a linear regression on these variables and checked the scores. 
The R2 and RMSE values showed the model is working well for both training and testing sets. 
"""
