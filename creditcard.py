import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE


# 1. Load Dataset

data = pd.read_csv("creditcard.csv")

print("Dataset shape:", data.shape)
print("\nClass distribution:")
print(data["Class"].value_counts())


# 2. Sample data (to avoid crash)

# Take small balanced sample for faster execution
fraud = data[data["Class"] == 1]
normal = data[data["Class"] == 0].sample(n=20000, random_state=42)

data = pd.concat([fraud, normal])

print("\nAfter sampling:")
print(data["Class"].value_counts())


# 3. Feature & Target

X = data.drop("Class", axis=1)
y = data["Class"]


# 4. Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. SMOTE (now manageable)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:", pd.Series(y_train_res).value_counts())


# 7. Models

lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(n_estimators=10, max_depth=20,  random_state=42)

lr.fit(X_train_res, y_train_res)
rf.fit(X_train_res, y_train_res)


# 8. Predictions

lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)


# 9. Reports

print("\n--- Logistic Regression ---")
print(classification_report(y_test, lr_preds))

print("\n--- Random Forest ---")
print(classification_report(y_test, rf_preds))


# 10. Confusion Matrix

cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# 11. ROC Curve

lr_probs = lr.predict_proba(X_test)[:, 1]
rf_probs = rf.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(y_test, lr_probs)
rf_auc = roc_auc_score(y_test, rf_probs)

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

plt.plot(lr_fpr, lr_tpr, label=f"LR (AUC={lr_auc:.2f})")
plt.plot(rf_fpr, rf_tpr, label=f"RF (AUC={rf_auc:.2f})")
plt.legend()
plt.show()
print("Training Logistic Regression...")
lr.fit(X_train_res, y_train_res)
print("Logistic Regression done!")

print("Training Random Forest...")
rf.fit(X_train_res, y_train_res)
print("Random Forest done!")
