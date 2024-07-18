# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# 1. Data Preprocessing
# Load the dataset
df = pd.read_csv('data_july3.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# Encode categorical variables if needed (assuming 'target' is categorical)
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# Split the data into features and target
X = df.drop(columns=['target'])
y = df['target']

# Scale/normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Exploratory Data Analysis (EDA)
# Statistical summaries
print(df.describe())

# Data distribution and relationships between features
sns.pairplot(df, hue='target')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# 3. Classification
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, y_pred_log))

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)
print("Decision Tree:\n", classification_report(y_test, y_pred_dt))

# Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Cross-validation
cv_scores_log = cross_val_score(log_reg, X_scaled, y, cv=5)
cv_scores_dt = cross_val_score(dt_clf, X_scaled, y, cv=5)
cv_scores_rf = cross_val_score(rf_clf, X_scaled, y, cv=5)
print("Cross-validation scores (Logistic Regression):", cv_scores_log)
print("Cross-validation scores (Decision Tree):", cv_scores_dt)
print("Cross-validation scores (Random Forest):", cv_scores_rf)

# 4. Regression
# Assuming 'target' is a continuous variable for regression purposes
# For demonstration purposes, we'll transform the target variable back
y_reg = df['target']  # You can revert this line based on the dataset

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
print("Linear Regression:\nR-squared:", r2_score(y_test, y_pred_lin))
print("MSE:", mean_squared_error(y_test, y_pred_lin))

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
y_pred_dt_reg = dt_reg.predict(X_test)
print("Decision Tree Regressor:\nR-squared:", r2_score(y_test, y_pred_dt_reg))
print("MSE:", mean_squared_error(y_test, y_pred_dt_reg))

# Cross-validation
cv_scores_lin = cross_val_score(lin_reg, X_scaled, y, cv=5, scoring='r2')
cv_scores_dt_reg = cross_val_score(dt_reg, X_scaled, y, cv=5, scoring='r2')
print("Cross-validation R-squared scores (Linear Regression):", cv_scores_lin)
print("Cross-validation R-squared scores (Decision Tree Regressor):", cv_scores_dt_reg)

# 5. Confusion Matrix:
# For classification tasks, plot the confusion matrix and compute metrics

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Apply Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Compute confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Plot confusion matrix
plot_confusion_matrix(cm_lr, label_encoder.classes_)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr, average='weighted')
recall = recall_score(y_test, y_pred_lr, average='weighted')
f1 = f1_score(y_test, y_pred_lr, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 6. Cross-Validation:
# Implement k-fold cross-validation for both classification and regression models

# Function to perform k-fold cross-validation
def perform_cross_validation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation score: {scores.mean()}")
    print(f"Standard deviation of cross-validation score: {scores.std()}")

# Logistic Regression cross-validation
print("Logistic Regression Cross-Validation:")
perform_cross_validation(lr, X_scaled, y)

# Apply Linear Regression for regression task
lr_reg = LinearRegression()
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.3, random_state=42)
lr_reg.fit(X_train_reg, y_train_reg)
y_pred_lr_reg = lr_reg.predict(X_test_reg)

# Evaluate the model
r_squared = r2_score(y_test_reg, y_pred_lr_reg)
mse = mean_squared_error(y_test_reg, y_pred_lr_reg)

print(f"R-squared: {r_squared}")
print(f"Mean Squared Error: {mse}")

# Linear Regression cross-validation
print("Linear Regression Cross-Validation:")
perform_cross_validation(lr_reg, X_scaled, y_reg)
