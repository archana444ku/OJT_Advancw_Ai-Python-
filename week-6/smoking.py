import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Example dataset
data = {
    'age': [55, 60, 45, 50, 65],
    'gender': ['male', 'female', 'male', 'female', 'male'],
    'cholesterol': [220, 180, 190, 200, 230],
    'bp': [140, 130, 110, 120, 150],
    'smoking': ['yes', 'no', 'yes', 'no', 'yes'],
    'diabetes': ['no', 'yes', 'no', 'no', 'yes'],
    'exercise': ['yes', 'no', 'yes', 'yes', 'no'],
    'heart_attack': [1, 1, 0, 0, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Print columns of the DataFrame to verify column names
print("Columns in DataFrame:", df.columns)

# Convert categorical variables to numerical using LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking'] = le.fit_transform(df['smoking'])
df['diabetes'] = le.fit_transform(df['diabetes'])
df['exercise'] = le.fit_transform(df['exercise'])

# Separate features and target
X = df.drop('heart_attack', axis=1)
y = df['heart_attack']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print("\nEvaluation Metrics:")
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
