# -------------------------------------------------------
# Heart Disease Prediction using FS-Ensemble Model
# Description:
# This project compares Logistic Regression, Random Forest,
# and a proposed Feature Selection based Ensemble model
# for heart disease prediction.
# -------------------------------------------------------

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# 2. Load the dataset
# Dataset used: Heart Disease Dataset (UCI / Kaggle)
data = pd.read_csv("heart.csv")

# Display first few rows of the dataset
print("Dataset Preview:")
print(data.head())


# 3. Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())


# 4. Separate features (X) and target variable (y)
X = data.drop("target", axis=1)
y = data["target"]


# 5. Feature Scaling
# Standardization improves model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 6. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# 7. Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)


# 8. Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# 9. Proposed FS-Ensemble Model
# Combines Logistic Regression and Random Forest
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('rf', rf_model)
    ],
    voting='hard'
)

ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)


# 10. Function to evaluate model performance
def evaluate_model(model_name, y_true, y_pred):
    print(f"\n{model_name} Performance:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))


# 11. Evaluate all models
evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("Proposed FS-Ensemble Model", y_test, y_pred_ensemble)


# 12. Accuracy Comparison Visualization
models = ['Logistic Regression', 'Random Forest', 'Proposed FS-Ensemble']
accuracy_scores = [
    accuracy_score(y_test, y_pred_lr) * 100,
    accuracy_score(y_test, y_pred_rf) * 100,
    accuracy_score(y_test, y_pred_ensemble) * 100
]

plt.bar(models, accuracy_scores)
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison of Models")
plt.show()


# 13. Confusion Matrix for Proposed Model
cm = confusion_matrix(y_test, y_pred_ensemble)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Proposed FS-Ensemble Model")
plt.show()


# 14. Feature Importance Visualization (Random Forest)
feature_importance = rf_model.feature_importances_
feature_names = X.columns

plt.barh(feature_names, feature_importance)
plt.xlabel("Importance Score")
plt.title("Feature Importance using Random Forest")
plt.show()


# ------------------- END OF CODE -------------------