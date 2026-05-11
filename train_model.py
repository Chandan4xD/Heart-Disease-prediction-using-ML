# =============================================================
# Heart Disease Prediction - Model Training Script
# BA College of Engineering and Technology
# Department of CSE | 8th Semester
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Heart Disease Prediction - Model Training")
print("  BA College of Engineering and Technology")
print("=" * 60)

# ----------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------
print("\n[1] Loading Cleveland Heart Disease Dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(url, names=columns, na_values='?')
print(f"   Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------
print("\n[2] Preprocessing...")

# Drop rows with missing values
df.dropna(inplace=True)
print(f"   After removing nulls: {df.shape[0]} rows")

# Binarize target (0 = No Disease, 1 = Disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Convert ca and thal to int
df['ca'] = df['ca'].astype(int)
df['thal'] = df['thal'].astype(int)

print(f"   Target distribution:\n{df['target'].value_counts().to_string()}")

# ----------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA) - Save plots
# ----------------------------------------------------------
print("\n[3] Generating EDA plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Heart Disease - Exploratory Data Analysis", fontsize=16, fontweight='bold')

# Age distribution
axes[0, 0].hist(df['age'], bins=20, color='steelblue', edgecolor='white')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')

# Target count
colors = ['#2ecc71', '#e74c3c']
df['target'].value_counts().plot(kind='bar', ax=axes[0, 1], color=colors, edgecolor='white')
axes[0, 1].set_title('Heart Disease Count')
axes[0, 1].set_xticklabels(['No Disease (0)', 'Disease (1)'], rotation=0)
axes[0, 1].set_ylabel('Count')

# Cholesterol vs Age
axes[1, 0].scatter(df['age'], df['chol'], c=df['target'], cmap='RdYlGn_r', alpha=0.6)
axes[1, 0].set_title('Cholesterol vs Age')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Cholesterol')

# Correlation heatmap
corr = df.corr()
sns.heatmap(corr[['target']].sort_values('target', ascending=False),
            annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1], cbar=True)
axes[1, 1].set_title('Feature Correlation with Target')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   EDA plots saved as 'eda_plots.png'")

# ----------------------------------------------------------
# 4. FEATURE ENGINEERING
# ----------------------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n[4] Train/Test Split: {X_train.shape[0]} train, {X_test.shape[0]} test")

# ----------------------------------------------------------
# 5. TRAIN ALL 5 MODELS
# ----------------------------------------------------------
print("\n[5] Training Models...")

models = {
    'Decision Tree':     DecisionTreeClassifier(random_state=42),
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':               SVC(kernel='rbf', probability=True, random_state=42),
    'KNN':               KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall':    round(recall_score(y_test, y_pred) * 100, 2),
        'F1-Score':  round(f1_score(y_test, y_pred) * 100, 2),
    }
    print(f"   ✓ {name}: Accuracy = {results[name]['Accuracy']}%")

# ----------------------------------------------------------
# 6. RESULTS TABLE
# ----------------------------------------------------------
print("\n[6] Model Comparison:")
results_df = pd.DataFrame(results).T
print(results_df.to_string())

best_model_name = results_df['Accuracy'].idxmax()
print(f"\n   🏆 Best Model: {best_model_name} ({results_df.loc[best_model_name, 'Accuracy']}% Accuracy)")

# ----------------------------------------------------------
# 7. CONFUSION MATRIX PLOT
# ----------------------------------------------------------
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   Confusion matrix saved as 'confusion_matrix.png'")

# ----------------------------------------------------------
# 8. SAVE BEST MODEL & SCALER
# ----------------------------------------------------------
print("\n[7] Saving model and scaler...")
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("   best_model.pkl  ✓")
print("   scaler.pkl      ✓")
print("   results.pkl     ✓")
print("\n✅ Training Complete! Run 'streamlit run app.py' to launch the web app.")
