import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

"""
Step 0 & 1:
For this assignment, I am using the Bank Marketing dataset available from UCI.
The file is named bank.csv and the separator used in the file is a semicolon (;).
I begin by reading the data into pandas and checking the structure to understand what columns exist.
"""

# Load data
df = pd.read_csv("bank.csv", delimiter=';')
print("Dataset loaded successfully!")
print("Shape of data:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types Summary:")
print(df.dtypes)

"""
Step 2:
I am only required to use a few columns for this analysis.
Therefore, I create a new dataframe named df2 that includes:
y, job, marital, default, housing, and poutcome.
"""

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
print("\nSubset dataframe df2 created with selected columns.")
print(df2.head(4))

"""
Step 3:
Since most columns in df2 are categorical, I need to convert them into numeric form.
To do this, I use pandas get_dummies() which converts text data into dummy variables.
"""

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])
print("\nDummy encoding completed. New shape of df3:", df3.shape)

"""
Step 4:
I will now create a heatmap to explore correlations among the numeric variables.
This will help visualize relationships, even though most dummy variables won't correlate strongly.
"""

corr_values = df3.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_values, cmap="YlGnBu")
plt.title("Correlation Heatmap for df3")
plt.show()

"""
Observation:
From the heatmap, I can see that correlations between most variables are quite weak.
This makes sense because each category was split into its own column (dummy variables).
The target variable 'y' has very little correlation with individual predictors.
"""

"""
Step 5:
Now I separate the target column 'y' from the other features.
I will map 'yes' and 'no' in y to 1 and 0 respectively for modeling purposes.
"""

df3['y'] = df3['y'].map({'yes': 1, 'no': 0})
y = df3['y']
X = df3.drop('y', axis=1)
print("\nSeparated target and feature variables.")
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

"""
Step 6:
I split the data into training and testing sets.
I use 75% for training and 25% for testing to match assignment requirements.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\nData successfully divided into training and testing sets.")

"""
Step 7:
Next, I train a Logistic Regression model.
This model works well for binary classification problems like this one.
"""

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

"""
Step 8:
Now I will evaluate how well the Logistic Regression model performs.
I use both confusion matrix and accuracy score for this.
"""

cm_log = confusion_matrix(y_test, y_pred_log)
acc_log = accuracy_score(y_test, y_pred_log)

print("\n--- Logistic Regression Results ---")
print("Confusion Matrix:\n", cm_log)
print(f"Accuracy: {acc_log:.4f}")

sns.heatmap(cm_log, annot=True, fmt='d', cmap='crest')
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""
Step 9:
I will now train a K-Nearest Neighbors (KNN) model.
For this example, I start with k=3 as suggested in the task.
"""

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("\n--- KNN Model Results (k=3) ---")
print("Confusion Matrix:\n", cm_knn)
print(f"Accuracy: {acc_knn:.4f}")

sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Purples')
plt.title("KNN (k=3) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

"""
Step 10:
Finally, I compare both models and summarize which one performed better.
"""

print("\n--- Model Accuracy Comparison ---")
print(f"Logistic Regression Accuracy: {acc_log:.4f}")
print(f"KNN Accuracy (k=3): {acc_knn:.4f}")

"""
Findings:
Both models perform classification, but Logistic Regression performs slightly better in this dataset.
This could be because Logistic Regression handles binary outcomes more efficiently
and KNN might be less effective with many dummy variables (high-dimensional space).
"""
