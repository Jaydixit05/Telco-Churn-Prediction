import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("PHASE 2: MACHINE LEARNING MODEL TRAINING")
print("=" * 70)

print("\n[1/7] Loading preprocessed data...")

X_train = pd.read_csv('jay/data/X_train.csv')
X_test = pd.read_csv('jay/data/X_test.csv')
y_train = pd.read_csv('jay/data/y_train.csv')
y_test = pd.read_csv('jay/data/y_test.csv')

X_train = X_train.replace((np.inf, -np.inf, np.nan), 0)
X_test = X_test.replace((np.inf, -np.inf, np.nan), 0)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Churn distribution in training: {np.bincount(y_train)}")

print("\n[2/7] Handling class imbalance with SMOTE...")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {np.bincount(y_train)}")
print(f"After SMOTE: {np.bincount(y_train_balanced)}")
print("Classes are now balanced for better model training")

print("\n[3/7] Training Model 1: Logistic Regression...")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_balanced, y_train_balanced)

lr_pred = lr_model.predict(X_test)
lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_pred_proba)

print(f"Accuracy: {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"F1-Score: {lr_f1:.4f}")
print(f"ROC-AUC: {lr_auc:.4f}")

print("\n[4/7] Training Model 2: Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=10,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)

rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1-Score: {rf_f1:.4f}")
print(f"ROC-AUC: {rf_auc:.4f}")

print("\n[5/7] Training Model 3: XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_balanced, y_train_balanced)

xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba)

print(f"Accuracy: {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall: {xgb_recall:.4f}")
print(f"F1-Score: {xgb_f1:.4f}")
print(f"ROC-AUC: {xgb_auc:.4f}")

print("\n[6/7] Comparing all models...")

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [lr_accuracy, rf_accuracy, xgb_accuracy],
    'Precision': [lr_precision, rf_precision, xgb_precision],
    'Recall': [lr_recall, rf_recall, xgb_recall],
    'F1-Score': [lr_f1, rf_f1, xgb_f1],
    'ROC-AUC': [lr_auc, rf_auc, xgb_auc]
})

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)
print(comparison_df.to_string(index=False))

best_model_idx = comparison_df['F1-Score'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_f1 = comparison_df.loc[best_model_idx, 'F1-Score']

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}")
print(f"F1-Score: {best_f1:.4f}")
print("=" * 70)

print("\n[7/7] Saving models...")

joblib.dump(lr_model, 'jay/models/logistic_regression.pkl')
joblib.dump(rf_model, 'jay/models/random_forest.pkl')
joblib.dump(xgb_model, 'jay/models/xgboost.pkl')

if best_model_name == 'Logistic Regression':
    joblib.dump(lr_model, 'jay/models/best_model.pkl')
elif best_model_name == 'Random Forest':
    joblib.dump(rf_model, 'jay/models/best_model.pkl')
else:
    joblib.dump(xgb_model, 'jay/models/best_model.pkl')

print("Models saved:")
print("  - jay/models/logistic_regression.pkl")
print("  - jay/models/random_forest.pkl")
print("  - jay/models/xgboost.pkl")
print(f"  - jay/models/best_model.pkl ({best_model_name})")

print("\n[8/7] Creating visualizations...")

fig, ax = plt.subplots(figsize=(14, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

ax.bar(x - width, comparison_df.iloc[0, 1:].values, width,
       label='Logistic Regression', color='#3498db')
ax.bar(x, comparison_df.iloc[1, 1:].values, width,
       label='Random Forest', color='#2ecc71')
ax.bar(x + width, comparison_df.iloc[2, 1:].values, width,
       label='XGBoost', color='#e74c3c')

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('jay/models/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: jay/models/model_comparison.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models_data = [
    ('Logistic Regression', lr_pred),
    ('Random Forest', rf_pred),
    ('XGBoost', xgb_pred)
]

for idx, (name, predictions) in enumerate(models_data):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[idx],
        cbar=False,
        square=True
    )
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xticklabels(['Stayed (0)', 'Churned (1)'])
    axes[idx].set_yticklabels(['Stayed (0)', 'Churned (1)'])

plt.tight_layout()
plt.savefig('jay/models/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: jay/models/confusion_matrices.png")

fig, ax = plt.subplots(figsize=(10, 8))

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pred_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_pred_proba)

ax.plot(lr_fpr, lr_tpr,
        label=f'Logistic Regression (AUC = {lr_auc:.3f})',
        linewidth=2, color='#3498db')
ax.plot(rf_fpr, rf_tpr,
        label=f'Random Forest (AUC = {rf_auc:.3f})',
        linewidth=2, color='#2ecc71')
ax.plot(xgb_fpr, xgb_tpr,
        label=f'XGBoost (AUC = {xgb_auc:.3f})',
        linewidth=2, color='#e74c3c')
ax.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('jay/models/roc_curves.png', dpi=300, bbox_inches='tight')
print("Saved: jay/models/roc_curves.png")

if best_model_name in ['Random Forest', 'XGBoost']:
    print("\nCalculating feature importance...")

    if best_model_name == 'Random Forest':
        importance = rf_model.feature_importances_
    else:
        importance = xgb_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        feature_importance_df['Feature'],
        feature_importance_df['Importance'],
        color='#3498db'
    )
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(
        f'Top 15 Most Important Features - {best_model_name}',
        fontsize=14,
        fontweight='bold'
    )
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('jay/models/feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: jay/models/feature_importance.png")

print("\n" + "=" * 70)
print("PHASE 2 COMPLETE - MODEL TRAINING FINISHED!")
print("=" * 70)

print("\nWhat we accomplished:")
print("  1. Loaded preprocessed data (5,634 training samples)")
print("  2. Balanced classes using SMOTE")
print("  3. Trained 3 different ML models")
print("  4. Evaluated all models on test set (1,409 samples)")
print("  5. Compared performance metrics")
print(f"  6. Selected best model: {best_model_name}")
print("  7. Saved all models")
print("  8. Created 3-4 visualization charts")

print("\nFiles created:")
print("  Models:")
print("    - jay/models/logistic_regression.pkl")
print("    - jay/models/random_forest.pkl")
print("    - jay/models/xgboost.pkl")
print("    - jay/models/best_model.pkl")
print("  Visualizations:")
print("    - jay/models/model_comparison.png")
print("    - jay/models/confusion_matrices.png")
print("    - jay/models/roc_curves.png")
if best_model_name in ['Random Forest', 'XGBoost']:
    print("    - jay/models/feature_importance.png")

print("\nNext Phase: Build a Streamlit dashboard for predictions!")
print("=" * 70)
