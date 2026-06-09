import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("diabetes.csv")

# Display first 5 rows
print("Dataset Preview:")
print(df.head())

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test prediction with sample patient data
sample_patient = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]

prediction = model.predict(sample_patient)

if prediction[0] == 1:
    print("\nPrediction: Patient is likely Diabetic")
else:
    print("\nPrediction: Patient is likely Not Diabetic")